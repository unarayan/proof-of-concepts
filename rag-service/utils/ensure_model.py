from __future__ import annotations

import logging
import os

from huggingface_hub import snapshot_download

from utils.cli_utils import run_cli
from utils.config_loader import config


logger = logging.getLogger(__name__)
_WEIGHT_FORMATS = {"fp32", "fp16", "int8", "int4"}


def _slugify(model_name: str, suffix: str | None = None) -> str:
    slug = model_name.replace("/", "_")
    return f"{slug}__{suffix}" if suffix else slug


def _is_openvino_repo(model_name: str) -> bool:
    normalized = model_name.lower()
    return model_name.startswith("OpenVINO/") or normalized.endswith("-ov")


def _llm_model_exists(output_dir: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, "openvino_model.xml"))


def _embedding_model_exists(output_dir: str) -> bool:
    xml_path = os.path.join(output_dir, "openvino_model.xml")
    return os.path.isfile(xml_path) or any(name.endswith(".xml") for name in os.listdir(output_dir)) if os.path.isdir(output_dir) else False


def _download_repo(repo_id: str, output_dir: str, hf_token: str | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
    )
    return output_dir


def _export_openvino_model(model_name: str, output_dir: str, weight_format: str, task: str) -> str:
    if weight_format not in _WEIGHT_FORMATS:
        raise ValueError(f"Unsupported OpenVINO weight format: {weight_format}")

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "optimum-cli",
        "export",
        "openvino",
        "--task",
        task,
        "--model",
        model_name,
        "--trust-remote-code",
        output_dir,
    ]
    if weight_format:
        cmd.extend(["--weight-format", weight_format])

    logger.info("Exporting %s to %s (task=%s, weight_format=%s)", model_name, output_dir, task, weight_format)
    result = run_cli(cmd, log_fn=logger.info)
    if result != 0:
        raise RuntimeError(f"OpenVINO export failed for {model_name} with exit code {result}")
    return output_dir


def get_llm_model_path() -> str:
    llm_cfg = config.models.llm
    explicit = getattr(llm_cfg, "model_path", None)
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(os.path.dirname(os.path.dirname(__file__)), explicit)

    hf_id = llm_cfg.hf_id
    suffix = None if _is_openvino_repo(hf_id) else getattr(llm_cfg, "weight_format", "int8")
    return os.path.join(llm_cfg.models_base_path, _slugify(hf_id, suffix))


def get_embedding_model_path() -> str:
    emb_cfg = config.models.embedding
    provider = getattr(emb_cfg, "provider", "sentence_transformers").strip().lower()
    suffix = getattr(emb_cfg, "weight_format", None) if provider == "openvino" else None
    return os.path.join(emb_cfg.models_base_path, provider, _slugify(emb_cfg.hf_id, suffix))


def ensure_llm_model(force: bool = False) -> str:
    llm_cfg = config.models.llm
    output_dir = get_llm_model_path()
    hf_id = llm_cfg.hf_id
    
    # For PyTorch models, just verify HF can download them on first use
    # No need to pre-cache or export
    logger.info("LLM model will be loaded from HuggingFace on first use: %s", hf_id)
    return hf_id


def ensure_embedding_model(force: bool = False) -> str:
    emb_cfg = config.models.embedding
    provider = getattr(emb_cfg, "provider", "sentence_transformers").strip().lower()
    output_dir = get_embedding_model_path()
    if not force and os.path.isdir(output_dir) and any(os.scandir(output_dir)):
        logger.info("Using cached embedding model at %s", output_dir)
        return output_dir

    if provider == "sentence_transformers":
        if getattr(emb_cfg, "use_local_cache", True):
            logger.info("Downloading sentence-transformers embedding model %s to %s", emb_cfg.hf_id, output_dir)
            _download_repo(emb_cfg.hf_id, output_dir)
        return output_dir

    if provider == "openvino":
        if _is_openvino_repo(emb_cfg.hf_id):
            logger.info("Downloading OpenVINO embedding model %s to %s", emb_cfg.hf_id, output_dir)
            _download_repo(emb_cfg.hf_id, output_dir)
        else:
            _export_openvino_model(emb_cfg.hf_id, output_dir, getattr(emb_cfg, "weight_format", "fp16"), "feature-extraction")

        if not _embedding_model_exists(output_dir):
            raise RuntimeError(f"Embedding model was prepared but no XML model file was found in {output_dir}")
        return output_dir

    raise ValueError(f"Unsupported embedding provider: {provider}")


def ensure_model(force: bool = False) -> None:
    ensure_llm_model(force=force)
    ensure_embedding_model(force=force)


def resolve_embedding_model_source() -> str:
    emb_cfg = config.models.embedding
    output_dir = get_embedding_model_path()
    provider = getattr(emb_cfg, "provider", "sentence_transformers").strip().lower()
    if provider == "sentence_transformers":
        if os.path.isdir(output_dir) and any(os.scandir(output_dir)):
            return output_dir
        return emb_cfg.hf_id
    if provider == "openvino" and _embedding_model_exists(output_dir):
        return output_dir
    return emb_cfg.hf_id
