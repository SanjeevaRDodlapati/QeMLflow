#!/usr/bin/env python
"""
Script to check if Psi4 is available and provide installation options.
"""
import importlib
import os
import platform
import subprocess
import sys


def check_conda():
    """Check if conda is available and which environments exist."""
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            print(f"✅ Conda is available")
            print("\nAvailable conda environments:")
            print(result.stdout)
            return True
        else:
            print(f"⚠️ Conda command exists but returned an error:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ Conda is not available in PATH")
        return False


def check_psi4():
    """Try to import psi4 and report result."""
    try:
        import psi4

        version = getattr(psi4, "__version__", "unknown version")
        print(f"✅ Psi4 is installed (version: {version})")
        return True
    except ImportError as e:
        print(f"❌ Psi4 is not installed: {str(e)}")
        return False


def print_installation_instructions():
    """Print detailed instructions for installing Psi4."""
    system = platform.system()

    print("\n" + "=" * 50)
    print("PSI4 INSTALLATION INSTRUCTIONS")
    print("=" * 50)

    print("\nRecommended installation method using conda:")
    print("```")
    print("# Create a new environment (recommended)")
    print("conda create -n psi4env python=3.8")
    print("conda activate psi4env")
    print("conda install -c psi4 psi4")
    print("```")

    if system == "Linux":
        print("\nAlternative for Ubuntu/Debian:")
        print("```")
        print("sudo apt-get update")
        print("sudo apt-get install -y psi4 psi4-dev")
        print("```")

    print("\nDocker alternative (most reliable across platforms):")
    print("```")
    print("docker pull psi4/psi4:latest")
    print("docker run -it psi4/psi4:latest")
    print("```")

    print("\nNote for this notebook:")
    print("The day_04 notebook has fallback mechanisms when Psi4 is not available.")
    print(
        "It will use mock calculations instead of real quantum chemistry calculations."
    )


# Main execution
if __name__ == "__main__":
    print("=" * 50)
    print("PSI4 AVAILABILITY CHECK")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")

    has_psi4 = check_psi4()
    if not has_psi4:
        has_conda = check_conda()
        print_installation_instructions()
