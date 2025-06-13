#!/usr/bin/env python
"""
Psi4 Installation Verification Script

This script performs a simple test to verify that Psi4 is
correctly installed and functioning in your environment.

If successful, it will calculate the energy of a hydrogen molecule
and print the result.
"""

print("Beginning Psi4 verification test...")
print("Attempting to import Psi4...")

try:
    import psi4

    print(f"✅ Successfully imported Psi4 version {psi4.__version__}")

    # Set memory limit
    print("Setting up calculation parameters...")
    psi4.set_memory("500 MB")

    # Define hydrogen molecule geometry
    print("Defining H2 molecule geometry...")
    h2 = psi4.geometry(
        """
    0 1
    H
    H 1 0.9
    """
    )

    # Set up calculation output
    print("Setting up output file...")
    psi4.set_output_file("h2_test_output.txt")

    # Perform energy calculation
    print("Performing SCF energy calculation...")
    energy = psi4.energy("scf/cc-pvdz")

    print(f"✅ Calculation successful!")
    print(f"H2 SCF energy: {energy} Hartree")
    print("\nPsi4 is correctly installed and working!")
    print("For more details, check the h2_test_output.txt file.")

except ImportError:
    print("❌ Failed to import Psi4. It is not installed in this environment.")
    print("\nTo install Psi4, follow the instructions in the installation guide:")
    print("1. Using Miniforge (recommended on macOS):")
    print(
        "   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
    )
    print("   bash Miniforge3-MacOSX-x86_64.sh -b -p $HOME/miniforge3")
    print("   echo 'export PATH=\"$HOME/miniforge3/bin:$PATH\"' >> ~/.zshrc")
    print("   source ~/.zshrc")
    print("   mamba create -n psi4env python=3.8 -y")
    print("   mamba activate psi4env")
    print("   mamba install -c conda-forge psi4=1.9.1 -y")
    print("\n2. Using Docker:")
    print("   docker pull psi4/psi4:latest")
    print('   docker run -it -v "$(pwd)":/work -w /work psi4/psi4:latest')

except Exception as e:
    print(f"❌ Error while running Psi4: {str(e)}")
    print("There might be issues with your Psi4 installation.")
