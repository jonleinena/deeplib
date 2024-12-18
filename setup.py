from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Core requirements (excluding PyTorch and logging dependencies)
core_requirements = []
for req in requirements:
    if not any(req.startswith(pkg) for pkg in ["torch", "torchvision", "tensorboard", "mlflow", "wandb"]):
        core_requirements.append(req)

# Define logger dependencies
logger_requirements = [
    "tensorboard>=2.15.0",
    "mlflow>=2.9.0",
    "wandb>=0.16.0"
]

setup(
    name="deeplib",
    version="1.1.0",
    author="Jon Leiñena",
    author_email="leinenajon@gmail.com",
    description="A deep learning library for computer vision tasks (CUDA ≥11.8 compatible)",
    long_description=long_description + "\n\n" + """
## Installation Options

### Full Installation (Recommended)
```bash
pip install deeplib
```

### Core Installation (without logging backends)
```bash
pip install deeplib[core]
```

## CUDA Requirements

This package requires PyTorch with CUDA 11.8 or newer. To install PyTorch dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For other CUDA versions or CPU-only installation, see https://pytorch.org/get-started/locally/
""",
    long_description_content_type="text/markdown",
    url="https://github.com/jonleinena/deeplib",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.10",
    install_requires=core_requirements + logger_requirements,  # Full installation by default
    extras_require={
        "core": core_requirements,  # Core installation without loggers
        "all": core_requirements + logger_requirements,  # Same as default
        # Individual logger installations
        "tensorboard": core_requirements + ["tensorboard>=2.15.0"],
        "mlflow": core_requirements + ["mlflow>=2.9.0"],
        "wandb": core_requirements + ["wandb>=0.16.0"]
    }
) 