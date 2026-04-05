#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "questionary>=2.0",
#   "rich>=13.0",
#   "httpx>=0.27",
# ]
# ///
"""mlx-claude — TUI launcher for Claude Code against local MLX models.

Pick a profile, the script spins up `mlx_lm.server` (text) or
`mlx_vlm.server` (vision-language) + LiteLLM proxy via `uvx`, sets
ANTHROPIC_* env vars, and execs `claude --model <alias>`. Cleans up
backend processes on exit. No persistent installs — everything goes
through uv's cache.

Each profile carries its model-author-recommended sampling defaults
(the Ollama-Modelfile equivalent) — they're injected into every
Claude Code request via the LiteLLM config.
"""

from __future__ import annotations
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import questionary
from rich.console import Console

console = Console()

MLX_PORT = 8080
PROXY_PORT = 11434
PYTHON_VER = "3.12"
MIN_CONTEXT_WARN = 65536   # Ollama-docs recommended ≥64k for Claude Code
FORK_WHEEL = Path.home() / "Desktop" / \
    "mlx-0.31.2.dev20260404+72ec298f-cp312-cp312-macosx_26_0_arm64.whl"


@dataclass(frozen=True)
class SamplingParams:
    """Per-profile decoding defaults — the Ollama-Modelfile equivalent.

    Injected into every `/v1/chat/completions` request via LiteLLM.
    Request-level values from Claude Code (if any) override these.
    """
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 4096


@dataclass(frozen=True)
class Profile:
    alias: str                      # what `claude --model <alias>` sends
    hf_model_id: str                # HuggingFace model id
    label: str                      # menu label
    backend: str = "mlx_lm"         # "mlx_lm" (text) or "mlx_vlm" (vision+text)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    max_kv_size: int = 65536        # enforced on mlx_vlm.server; advisory for mlx_lm.server
    extra_withs: tuple[str, ...] = ()   # extra --with args for uvx


PROFILES: list[Profile] = [
    Profile(
        alias="qwen",
        hf_model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        label="Qwen 3.5 9B 4-bit VLM    (mlx-vlm, vision+text)",
        backend="mlx_vlm",
        # Qwen3 family author-recommended decoding (non-thinking mode)
        sampling=SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
            repetition_penalty=1.05, max_tokens=8192,
        ),
        max_kv_size=65536,
    ),
    Profile(
        alias="bonsai",
        hf_model_id="prism-ml/Bonsai-8B-mlx-1bit",
        label="Bonsai 8B 1-bit           (mlx-lm + PrismML fork wheel)",
        backend="mlx_lm",
        # Per HF model card "Best Practices → Generation Parameters"
        sampling=SamplingParams(
            temperature=0.5, top_p=0.9, top_k=20, min_p=0.0,
            repetition_penalty=1.0, max_tokens=4096,
        ),
        max_kv_size=65536,
        extra_withs=(str(FORK_WHEEL),),
    ),
]


def die(msg: str, code: int = 1) -> None:
    console.print(f"[red]ERROR:[/] {msg}")
    sys.exit(code)


def check_prereqs(p: Profile) -> None:
    from shutil import which
    for cmd in ("uv", "uvx", "claude"):
        if which(cmd) is None:
            die(f"'{cmd}' not on PATH.")
    for extra in p.extra_withs:
        if os.path.sep in extra and not extra.startswith(("http://", "https://", "git+")):
            path_part = extra.split("@")[-1].strip() if "@" in extra else extra
            if not Path(path_part).is_file():
                die(f"wheel/path not found: {path_part}")


def write_litellm_config(p: Profile) -> Path:
    s = p.sampling
    cfg = Path(tempfile.mkstemp(prefix="litellm-", suffix=".yaml")[1])
    cfg.write_text(f"""\
model_list:
  - model_name: {p.alias}
    litellm_params:
      model: openai/{p.hf_model_id}
      api_base: http://127.0.0.1:{MLX_PORT}/v1
      api_key: none
      temperature: {s.temperature}
      top_p: {s.top_p}
      max_tokens: {s.max_tokens}
      extra_body:
        top_k: {s.top_k}
        min_p: {s.min_p}
        repetition_penalty: {s.repetition_penalty}
litellm_settings:
  drop_params: true
""")
    return cfg


def wait_ready(url: str, timeout_s: int, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    die(f"{what} did not become ready at {url} within {timeout_s}s. Check logs.")


def start_proc(cmd: list[str], log_path: Path) -> subprocess.Popen:
    f = log_path.open("w")
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    return subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def shutdown(procs: list[subprocess.Popen]) -> None:
    for pr in procs:
        if pr.poll() is None:
            try:
                os.killpg(os.getpgid(pr.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    for pr in procs:
        try:
            pr.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def mlx_cmd(p: Profile) -> list[str]:
    """uvx invocation for mlx_lm.server (text) or mlx_vlm.server (vision)."""
    if p.backend == "mlx_vlm":
        cmd = ["uvx", "--python", PYTHON_VER, "--from", "mlx-vlm"]
        for w in p.extra_withs:
            cmd += ["--with", w]
        cmd += ["mlx_vlm.server", "--model", p.hf_model_id,
                "--port", str(MLX_PORT), "--host", "127.0.0.1"]
        if p.max_kv_size:
            cmd += ["--max-kv-size", str(p.max_kv_size)]
    else:  # mlx_lm
        cmd = ["uvx", "--python", PYTHON_VER, "--from", "mlx-lm"]
        for w in p.extra_withs:
            cmd += ["--with", w]
        cmd += ["mlx_lm.server", "--model", p.hf_model_id,
                "--port", str(MLX_PORT), "--host", "127.0.0.1"]
        # Server-side sampling defaults. LiteLLM sends per-request overrides
        # anyway, but these cover any request that reaches the server directly.
        cmd += [
            "--temp", str(p.sampling.temperature),
            "--top-p", str(p.sampling.top_p),
            "--top-k", str(p.sampling.top_k),
            "--min-p", str(p.sampling.min_p),
        ]
        # mlx_lm.server has no --max-kv-size flag yet (tracks ml-explore/mlx-lm#615)
    return cmd


def litellm_cmd(cfg: Path) -> list[str]:
    return [
        "uvx", "--python", PYTHON_VER, "--from", "litellm[proxy]",
        "litellm", "--config", str(cfg),
        "--port", str(PROXY_PORT), "--host", "127.0.0.1",
    ]


def print_sampling_status(p: Profile) -> None:
    s = p.sampling
    console.print(
        f"[dim]  sampling:[/] temp={s.temperature} top_p={s.top_p} "
        f"top_k={s.top_k} min_p={s.min_p} rep={s.repetition_penalty} "
        f"max_tokens={s.max_tokens}"
    )
    console.print(f"[dim]  context:[/] max_kv_size={p.max_kv_size}")
    if p.max_kv_size and p.max_kv_size < MIN_CONTEXT_WARN:
        console.print(
            f"[yellow]  WARN:[/] max_kv_size < {MIN_CONTEXT_WARN} — "
            "Claude Code recommends ≥64k context"
        )


def main() -> int:
    choice: Profile | None = questionary.select(
        "Pick an MLX model for Claude Code:",
        choices=[questionary.Choice(title=p.label, value=p) for p in PROFILES],
    ).ask()
    if choice is None:
        return 0

    check_prereqs(choice)
    cfg = write_litellm_config(choice)
    mlx_log = Path(f"/tmp/mlx-{MLX_PORT}.log")
    proxy_log = Path(f"/tmp/litellm-{PROXY_PORT}.log")
    procs: list[subprocess.Popen] = []

    try:
        console.print(f"[cyan]Starting {choice.backend}.server[/] "
                      f"([dim]{choice.hf_model_id}[/]) on :{MLX_PORT} ...")
        print_sampling_status(choice)
        procs.append(start_proc(mlx_cmd(choice), mlx_log))
        wait_ready(
            f"http://127.0.0.1:{MLX_PORT}/v1/models", 600,
            f"{choice.backend}.server",
        )
        console.print("  [green]ready[/]")

        console.print(f"[cyan]Starting LiteLLM proxy[/] on :{PROXY_PORT} ...")
        procs.append(start_proc(litellm_cmd(cfg), proxy_log))
        wait_ready(
            f"http://127.0.0.1:{PROXY_PORT}/v1/models", 120, "litellm proxy",
        )
        console.print("  [green]ready[/]")

        env = os.environ.copy()
        env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{PROXY_PORT}"

        console.print(f"\n[bold]Launching:[/] claude --model {choice.alias}")
        console.print(f"[dim](logs: {mlx_log}  {proxy_log})[/]\n")

        return subprocess.run(
            ["claude", "--model", choice.alias, *sys.argv[1:]], env=env
        ).returncode
    except KeyboardInterrupt:
        return 130
    finally:
        console.print("[dim]stopping backends ...[/]")
        shutdown(procs)
        cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
