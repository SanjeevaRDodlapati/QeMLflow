from setuptools import find_packages, setup

setup(
    name="QeMLflow",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for practicing machine learning and quantum computing techniques for molecular modeling and drug designing.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "deepchem",
        "rdkit",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "qiskit",  # Add any other dependencies you may need
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
