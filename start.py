import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def resolve(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise click.ClickException(
            f"{name} not found on PATH. Run this via: uv run python start.py"
        )
    return path


@click.command()
@click.option(
    "--model",
    envvar="CHAT_MODEL_NAME",
    show_envvar=True,
    default="Qwen/Qwen3-4B-Instruct-2507",
    help="Model to preload and serve.",
)
@click.option(
    "--port",
    envvar="CHAT_SERVER_PORT",
    show_envvar=True,
    default=8000,
    type=int,
    help="Port for the model server.",
)
@click.option(
    "--timeout",
    envvar="READY_TIMEOUT",
    show_envvar=True,
    default=1800,
    type=int,
    help="Seconds to wait for the model to become ready.",
)
@click.option(
    "--log-level",
    envvar="SERVER_LOG_LEVEL",
    show_envvar=True,
    default="info",
    help="Log level for the model server.",
)
def start(model: str, port: int, timeout: int, log_level: str) -> None:
    health_url = f"http://localhost:{port}/health"
    server = None

    def shutdown(*_) -> None:
        if server is None or server.poll() is not None:
            return
        click.echo(f"Stopping model server (pid {server.pid})")
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            else:
                server.terminate()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            else:
                server.kill()

    try:
        click.echo(f"Starting model server for {model} on port {port}")
        server = subprocess.Popen(
            [
                resolve("transformers"), "serve", model,
                "--port", str(port), "--log-level", log_level,
            ],
            cwd=Path(__file__).resolve().parent,
            start_new_session=(os.name == "posix"),
        )
        signal.signal(signal.SIGTERM, shutdown)

        deadline = time.monotonic() + timeout
        elapsed = 0
        while not healthy(health_url):
            if server.poll() is not None:
                click.echo(
                    f"Model server exited with code {server.returncode} before becoming ready.",
                    err=True,
                )
                sys.exit(1)
            if time.monotonic() > deadline:
                click.echo(f"Timed out after {timeout}s waiting for {health_url}", err=True)
                sys.exit(1)
            if elapsed % 15 == 0:
                click.echo(f"loading model... {elapsed}s")
            time.sleep(1)
            elapsed += 1

        click.echo("Model ready. Launching Streamlit.")
        streamlit = subprocess.Popen(
            [resolve("streamlit"), "run", "gui.py"],
            cwd=ROOT,
        )
        sys.exit(streamlit.wait())
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        shutdown()


if __name__ == "__main__":
    start()
