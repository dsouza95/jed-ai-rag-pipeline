import subprocess


def dev() -> None:
    subprocess.run(["chainlit", "run", "app.py", "--watch"], check=True)
