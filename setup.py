from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nexus-quantum",
    version="1.0.0",
    author="Garrett C",
    author_email="garrett@nexus-quantum.ai",
    description="NEXUS Quantum Intelligence Framework - Revolutionary quantum computing integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Garrettc123/NEXUS-Quantum-Intelligence-Framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "ibm": ["qiskit>=0.43.0"],
        "google": ["cirq>=1.0.0"],
        "aws": ["amazon-braket-sdk>=1.30.0"],
        "dev": ["pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "pylint>=2.17"],
    },
    entry_points={
        "console_scripts": [
            "nexus-quantum=src.cli:main",
        ],
    },
)
