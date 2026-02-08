"""Setup script for model-diff."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="model-diff",
    version="0.1.0",
    author="BabyChrist666",
    author_email="babychrist666@example.com",
    description="Compare model checkpoints to understand training changes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BabyChrist666/model-diff",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # numpy is optional - library works with mock for basic operations
    ],
    extras_require={
        "numpy": ["numpy>=1.20.0"],
        "torch": ["torch>=1.9.0"],
        "safetensors": ["safetensors>=0.3.0"],
        "all": [
            "numpy>=1.20.0",
            "torch>=1.9.0",
            "safetensors>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    keywords=[
        "model",
        "checkpoint",
        "diff",
        "comparison",
        "training",
        "machine-learning",
        "deep-learning",
        "pytorch",
        "analysis",
    ],
    project_urls={
        "Bug Reports": "https://github.com/BabyChrist666/model-diff/issues",
        "Source": "https://github.com/BabyChrist666/model-diff",
        "Documentation": "https://babychrist666.github.io/model-diff/",
    },
)
