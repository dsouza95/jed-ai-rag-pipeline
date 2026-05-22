import subprocess


def dev():
    subprocess.run(["chainlit", "run", "app.py", "--watch"], check=True)
