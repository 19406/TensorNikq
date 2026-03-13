import argparse
import subprocess
import sys
import os
import shutil

VENV = "venv"

def get_python():
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python")
    return os.path.join(VENV, "bin", "python")

def get_pip():
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "pip")
    return os.path.join(VENV, "bin", "pip")

def get_pip_version(pip_path):
    result = subprocess.run(
        [pip_path, "--version"],
        capture_output=True,
        text=True
    )
    version = result.stdout.split()[1]
    return version

def setup():
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    pip = get_pip()
    
    print("Upgrading pip to the newest version (if needed)...")
    subprocess.run([pip, "install", "--upgrade", "pip"])

    print("Installing dependencies...")
    subprocess.run([pip, "install", "-r", "requirements.txt"])

    print("Setup complete!")
    
def test(test_name=None):
    python = get_python()
    cmd = [python, "-m", "pytest", "-vv"]

    if test_name:
        # Insert .py if needed
        if not test_name.endswith(".py"):
            test_name += ".py"

        test_path = os.path.join("tests", test_name)

        if os.path.exists(test_path):
            cmd.append(test_path)
        else:
            raise FileNotFoundError(f"Test file not found: {test_path}")

    subprocess.run(cmd)

def run():
    python = get_python()
    subprocess.run([python, "-m", "main.py"])

def lint():
    python = get_python()
    subprocess.run([python, "-m", "flake8", "src"])

def format():
    python = get_python()
    subprocess.run([python, "-m", "black", "src"])
    
def clean():
    print("Cleaning project...")
    
    # venv
    print("Removing venv...")
    if os.path.exists(VENV):
        shutil.rmtree(VENV)
        
    # cache
    remove_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".ipynb_checkpoints"
    ]
    
    print("Removing caches...")
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d in remove_dirs:
                path = os.path.join(root, d)
                print(f"Removing {path}")
                shutil.rmtree(path)
            
    print("Clean complete!")

def main():
    parser = argparse.ArgumentParser(description="Project command runner")
    parser.add_argument(
        "command",
        choices=["setup", "test", "run", "lint", "format", "clean"],
        help="Command to run",
    )
    
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional target (e.g. test file)",
    )

    args = parser.parse_args()

    commands = {
        "setup": setup,
        "test": test,
        "run": run,
        "lint": lint,
        "format": format,
        "clean": clean,
    }

    if args.command == "test":
        commands["test"](args.target)
    else: commands[args.command]()


if __name__ == "__main__":
    main()