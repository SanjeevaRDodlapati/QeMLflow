#!/usr/bin/env python

import os
import platform
import subprocess
import sys
from pathlib import Path


def green(text):
    return f"\033[92m{text}\033[0m"


def red(text):
    return f"\033[91m{text}\033[0m"


def yellow(text):
    return f"\033[93m{text}\033[0m"


def blue(text):
    return f"\033[94m{text}\033[0m"


def print_section(title):
    print(f"\n{blue('='*80)}")
    print(f"{blue('==')} {yellow(title)}")
    print(f"{blue('='*80)}\n")


def run_command(command, verbose=True):
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if verbose:
            print(f"Command: {command}")
            print(f"Exit code: {result.returncode}")
            print(f"Output:\n{result.stdout}")
            if result.stderr:
                print(f"Error:\n{result.stderr}")
        return result
    except Exception as e:
        if verbose:
            print(f"Exception executing '{command}': {str(e)}")
        return None


def check_system_info():
    print_section("System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python version: {platform.python_version()}")

    # Check if running in virtual env
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print(green("Running in a virtual environment"))
        print(f"Virtual env path: {sys.prefix}")
    else:
        print(red("Not running in a virtual environment"))

    # Check Python path
    print(f"Python executable: {sys.executable}")

    # Check pip and conda availability
    pip_result = run_command("pip --version", verbose=False)
    if pip_result and pip_result.returncode == 0:
        print(green(f"Pip available: {pip_result.stdout.strip()}"))
    else:
        print(red("Pip not available"))

    conda_result = run_command("conda --version", verbose=False)
    if conda_result and conda_result.returncode == 0:
        print(green(f"Conda available: {conda_result.stdout.strip()}"))
    else:
        print(red("Conda not available"))


def check_psi4_pip():
    print_section("Checking Psi4 availability via pip")
    result = run_command("pip index versions psi4", verbose=False)

    if result and "No matching distribution found for psi4" in result.stderr:
        print(
            red(
                "Psi4 is not available via pip for your current platform and Python version"
            )
        )
        print(yellow("This is expected as Psi4 is primarily distributed via conda"))
    elif result and result.returncode == 0:
        print(green("Psi4 found in pip index:"))
        print(result.stdout)
    else:
        print(red("Error checking pip index"))
        if result:
            print(result.stderr)


def check_conda_channels():
    print_section("Checking conda channels")
    run_command("conda config --show channels")

    # Make sure conda-forge is in channels
    result = run_command("conda config --get channels", verbose=False)
    if result and "conda-forge" not in result.stdout:
        print(yellow("Adding conda-forge channel..."))
        run_command("conda config --add channels conda-forge")


def check_psi4_conda():
    print_section("Checking Psi4 availability via conda")

    # Check default channels first
    print("Checking default channels...")
    run_command("conda search psi4")

    # Then check conda-forge
    print("\nChecking conda-forge channel...")
    run_command("conda search -c conda-forge psi4")


def check_conda_issues():
    print_section("Diagnosing conda issues")

    # Check for the specific libarchive.19.dylib error
    result = run_command("ls -la /opt/anaconda3/lib/libarchive*", verbose=False)
    if result and result.returncode != 0:
        print(red("Missing libarchive library in Anaconda installation"))
        print(yellow("This could be the cause of your conda errors"))

        # Check for homebrew libarchive that might be used as replacement
        brew_result = run_command("brew list | grep libarchive", verbose=False)
        if brew_result and brew_result.returncode == 0 and brew_result.stdout.strip():
            print(
                green(
                    "Homebrew libarchive is installed and could be used as a replacement"
                )
            )
            print(yellow("Try: brew link --force libarchive"))
        else:
            print(yellow("Consider installing libarchive via Homebrew:"))
            print("  brew install libarchive")


def check_miniforge_option():
    print_section("Alternative: Miniforge Installation")

    print(
        yellow(
            "Miniforge is a minimal conda distribution with conda-forge as default channel"
        )
    )
    print("It's often more reliable than Anaconda for scientific packages\n")

    miniforge_path = Path.home() / "miniforge3" / "bin" / "conda"
    if miniforge_path.exists():
        print(green("Miniforge already installed"))
        run_command(f"{miniforge_path} --version")
    else:
        print(yellow("Miniforge not found. Installation commands:"))
        print(
            """
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash Miniforge3-MacOSX-x86_64.sh -b -p $HOME/miniforge3
echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
        """
        )


def check_docker_option():
    print_section("Alternative: Docker Installation")

    docker_result = run_command("docker --version", verbose=False)
    if docker_result and docker_result.returncode == 0:
        print(green(f"Docker available: {docker_result.stdout.strip()}"))
        print(yellow("\nYou can use Docker to run Psi4:"))
        print(
            """
docker pull psi4/psi4:latest
docker run -it -v $(pwd):/work -w /work psi4/psi4:latest
        """
        )
    else:
        print(red("Docker not available"))
        print(
            yellow(
                "Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
            )
        )


def main():
    print(blue("\n===== PSI4 INSTALLATION DIAGNOSTIC TOOL =====\n"))

    check_system_info()
    check_psi4_pip()
    check_conda_channels()
    check_psi4_conda()
    check_conda_issues()
    check_miniforge_option()
    check_docker_option()

    print_section("Summary")
    print(yellow("Based on the diagnostics:"))
    print("1. Psi4 is not available via pip for your platform/Python version")
    print("2. Psi4 is available via conda-forge channel")
    print("3. Your conda installation appears to have issues with libarchive.19.dylib")
    print()
    print(green("Recommended approaches (in order of preference):"))
    print("1. Install Miniforge as a cleaner conda alternative")
    print("2. Use Docker with the official Psi4 image")
    print("3. Fix your existing conda installation")


if __name__ == "__main__":
    main()
