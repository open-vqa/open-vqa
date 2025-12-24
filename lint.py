import subprocess
import os

def lint_directory(directory="."):
    # Ensure Ruff is installed
    try:
        subprocess.run(["ruff", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("Ruff is not installed. Install it using 'pip install ruff'.")
        return

    # Recursively lint all Python files in the directory
    result = subprocess.run(["ruff", directory], capture_output=True, text=True)

    # Display the results
    if result.returncode == 0:
        print("No linting issues found.")
    else:
        print("Linting issues detected:\n")
        print(result.stdout)

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    lint_directory(project_dir)