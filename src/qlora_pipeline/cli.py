"""`qlora` CLI. The env bootstrap runs at import time, before any heavy library is imported.
Heavy imports (torch, transformers, ...) live inside the command functions.
"""

from __future__ import annotations

from qlora_pipeline.bootstrap import load_env

load_env()  # must stay above every other project / third-party import

import subprocess  # noqa: E402
import sys  # noqa: E402

import typer  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Local QLoRA fine-tuning pipeline")


@app.command("env-check")
def env_check(
    skip_hf: bool = typer.Option(False, help="Skip the gated Llama 2 access check"),
) -> None:
    """GPU / torch / bitsandbytes diagnostics, a real 4-bit fwd+bwd, HF access, disk, cache paths."""
    from qlora_pipeline.envcheck import run_env_check

    raise typer.Exit(0 if run_env_check(skip_hf=skip_hf) else 1)


@app.command("audit-c")
def audit_c(
    baseline: bool = typer.Option(False, "--baseline", help="Record the baseline instead of comparing"),
    force: bool = typer.Option(False, help="Overwrite an existing baseline"),
    threshold_mb: float = typer.Option(50.0, help="Max allowed growth per location"),
) -> None:
    """C: drive footprint audit against logs/c_baseline.json."""
    from qlora_pipeline import audit

    args = ["--threshold-mb", str(threshold_mb)]
    if baseline:
        args.append("--baseline")
    if force:
        args.append("--force")
    raise typer.Exit(audit.main(args))


@app.command("mlflow")
def mlflow_server(
    host: str = "127.0.0.1",
    port: int = 5000,
) -> None:
    """Start the local MLflow tracking server (sqlite backend + artifacts under QLORA_ROOT/mlflow)."""
    from qlora_pipeline.paths import ensure_dir, mlflow_dir

    mdir = mlflow_dir()
    art = ensure_dir(mdir / "artifacts")
    db = (mdir / "mlflow.db").as_posix()
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{db}",
        "--default-artifact-root", art.as_uri(),
        "--host", host, "--port", str(port),
    ]  # fmt: skip
    typer.echo("Running: " + " ".join(cmd))
    raise typer.Exit(subprocess.call(cmd))


def _later(phase: int):
    def _stub() -> None:
        typer.echo(f"Not implemented yet: arrives in Phase {phase}.")
        raise typer.Exit(2)

    return _stub


for _name, _phase in [
    ("data", 1), ("smoke", 2), ("train", 2), ("eval", 3), ("bench", 4),
    ("merge", 5), ("serve", 6), ("chat", 6), ("load-test", 6),
]:  # fmt: skip
    app.command(_name, help=f"(Phase {_phase})")(_later(_phase))


if __name__ == "__main__":
    app()
