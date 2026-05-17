#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

import evaluate_rag


SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[3]
DEFAULT_KB = REPO_DIR / "knowledge-base" / "store_knowledge_base.md"
DEFAULT_QNA = REPO_DIR / "knowledge-base" / "store_qna.random200.qwen3b.jsonl"
DEFAULT_OUTPUT = REPO_DIR / "knowledge-base" / "chunking_format_benchmark.json"
HEALTH_URL = "http://127.0.0.1:8020/health"
CONTEXT_URL = "http://127.0.0.1:8020/api/v1/context"
CONTEXT_FILE_URL = "http://127.0.0.1:8020/api/v1/context/file"
STATS_URL = "http://127.0.0.1:8020/api/v1/context/stats"
NO_PROXY = {"http": "", "https": ""}


def _load_questions(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("id", idx)
            items.append(record)
    return items


def _kill_existing_service() -> None:
    patterns = ["python3 main.py", "/.venv-1/bin/python main.py"]
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], capture_output=True)
    _wait_for_unhealthy()


def _wait_for_unhealthy(timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = requests.get(HEALTH_URL, timeout=2, proxies=NO_PROXY)
            if response.status_code != 200:
                return
        except Exception:
            return
        time.sleep(1)
    raise TimeoutError("previous service instance did not stop")


def _wait_for_health(process: subprocess.Popen | None = None, timeout_seconds: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"service exited early with code {process.returncode}")
        try:
            response = requests.get(HEALTH_URL, timeout=3, proxies=NO_PROXY)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("service did not become healthy")


def _start_service(weight_format: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["SMART_KIOSK_RAG__MODELS__LLM__WEIGHT_FORMAT"] = weight_format
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=SERVICE_DIR,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    setattr(process, "_log_handle", log_handle)
    try:
        _wait_for_health(process=process)
    except Exception:
        process.terminate()
        raise
    return process


def _stop_service(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    log_handle = getattr(process, "_log_handle", None)
    if log_handle is not None:
        try:
            log_handle.close()
        except Exception:
            pass
    try:
        _wait_for_unhealthy()
    except Exception:
        pass


def _current_debug_files(debug_dir: Path) -> set[str]:
    if not debug_dir.exists():
        return set()
    return {path.name for path in debug_dir.glob("*.jsonl")}


def _new_debug_file(debug_dir: Path, before: set[str]) -> str | None:
    if not debug_dir.exists():
        return None
    candidates = [path for path in debug_dir.glob("*.jsonl") if path.name not in before]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].name


def _rebuild_collection(kb_path: Path, debug_dir: Path) -> tuple[dict, str | None]:
    before = _current_debug_files(debug_dir)
    clear_response = requests.delete(CONTEXT_URL, timeout=30, proxies=NO_PROXY)
    clear_response.raise_for_status()
    with kb_path.open("rb") as handle:
        ingest_response = requests.post(
            CONTEXT_FILE_URL,
            files={"file": (kb_path.name, handle, "text/markdown")},
            timeout=1800,
            proxies=NO_PROXY,
        )
    ingest_response.raise_for_status()
    stats_response = requests.get(STATS_URL, timeout=30, proxies=NO_PROXY)
    stats_response.raise_for_status()
    return stats_response.json(), _new_debug_file(debug_dir, before)


def _score_retrieval(items: list[dict]) -> dict:
    evaluate_rag._local_retriever = None
    if not evaluate_rag._init_local_retriever():
        raise RuntimeError("failed to initialize local retriever")

    rr_values: list[float] = []
    by_nature: dict[str, list[float]] = {}
    for item in items:
        rr = evaluate_rag.get_rr_local(item["question"], item.get("answer", ""), top_k=6)
        rr_values.append(rr)
        by_nature.setdefault(item.get("nature", "unknown"), []).append(rr)

    total = len(rr_values)
    return {
        "mrr": round(sum(rr_values) / total, 4),
        "hit1": round(sum(1 for rr in rr_values if rr >= 1.0) / total, 4),
        "hit3": round(sum(1 for rr in rr_values if rr >= 1 / 3) / total, 4),
        "by_nature": {
            name: round(sum(values) / len(values), 4)
            for name, values in sorted(by_nature.items())
        },
    }


def _summarize(records: list[dict]) -> dict:
    numeric_keys = ("chunk_count", "mrr", "hit1", "hit3", "ingest_seconds")
    summary: dict[str, float | list[int]] = {"runs": len(records)}
    for key in numeric_keys:
        values = [float(record[key]) for record in records]
        summary[f"avg_{key}"] = round(statistics.mean(values), 4)
        summary[f"min_{key}"] = round(min(values), 4)
        summary[f"max_{key}"] = round(max(values), 4)
        summary[f"stdev_{key}"] = round(statistics.pstdev(values), 4)
    summary["chunk_counts"] = [int(record["chunk_count"]) for record in records]
    return summary


def run_benchmark(args: argparse.Namespace) -> dict:
    items = _load_questions(Path(args.qna_file))
    debug_dir = SERVICE_DIR / "storage" / "chunks_debug"
    all_results: dict[str, dict] = {
        "kb_file": str(Path(args.kb_file).resolve()),
        "qna_file": str(Path(args.qna_file).resolve()),
        "repeats": args.repeats,
        "formats": {},
    }

    for weight_format in args.formats:
        log_path = Path(args.log_dir) / f"chunk-benchmark-{weight_format}.log"
        process: subprocess.Popen | None = None
        runs: list[dict] = []
        try:
            _kill_existing_service()
            process = _start_service(weight_format, log_path)
            for run_index in range(1, args.repeats + 1):
                start = time.monotonic()
                stats, debug_file = _rebuild_collection(Path(args.kb_file), debug_dir)
                ingest_seconds = round(time.monotonic() - start, 2)
                scores = _score_retrieval(items)
                run_record = {
                    "run": run_index,
                    "weight_format": weight_format,
                    "chunk_count": int(stats.get("document_count", 0)),
                    "ingest_seconds": ingest_seconds,
                    "debug_file": debug_file,
                    **scores,
                }
                runs.append(run_record)
                print(json.dumps(run_record, ensure_ascii=False), flush=True)
        finally:
            _stop_service(process)

        all_results["formats"][weight_format] = {
            "runs": runs,
            "summary": _summarize(runs),
        }

    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark repeated semantic chunking ingests across LLM weight formats.")
    parser.add_argument("--kb-file", default=str(DEFAULT_KB), help="Knowledge-base markdown/text file to ingest")
    parser.add_argument("--qna-file", default=str(DEFAULT_QNA), help="Question set used for retrieval scoring")
    parser.add_argument("--formats", nargs="+", default=["int4", "int8"], help="Weight formats to benchmark")
    parser.add_argument("--repeats", type=int, default=5, help="Fresh ingests per format")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON file to write benchmark results")
    parser.add_argument("--log-dir", default="/tmp", help="Directory for per-format service logs")
    args = parser.parse_args()

    results = run_benchmark(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote benchmark results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())