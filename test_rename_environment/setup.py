from setuptools import find_packages, setup

setup(
    name="QeMLflow",
    version="0.1.0",
    author="QeMLflow Team",
    description="Chemical Machine Learning Framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
