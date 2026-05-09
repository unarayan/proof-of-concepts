import logging
import subprocess


def run_cli(cmd: list[str], log_fn=None) -> int:
    logger = log_fn or logging.getLogger(__name__).info
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger(line.rstrip())
    return process.wait()
