import subprocess
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def dev():
    subprocess.run(["chainlit", "run", "app.py"], check=True, cwd=_ROOT)
