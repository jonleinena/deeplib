from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Replace PyTorch requirements with CUDA-specific versions
cuda_specific_requirements = []
for req in requirements:
    if req.startswith("torch") or req.startswith("torchvision"):
        # Skip PyTorch requirements as they need special installation
        continue
    cuda_specific_requirements.append(req)

setup(
    name="deeplib",
    version="1.0.0",
    author="Jon Leiñena",
    author_email="leinenajon@gmail.com",
    description="A deep learning library for computer vision tasks (CUDA ≥11.8 compatible)",
    long_description=long_description + "\n\n" + """
## CUDA Requirements

This package is compatible with CUDA 11.8 or newer versions. To install the required PyTorch dependencies, use:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For newer CUDA versions, adjust the URL accordingly (e.g., cu121 for CUDA 12.1).
""",
    long_description_content_type="text/markdown",
    url="https://github.com/jonleinena/deeplib",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    install_requires=cuda_specific_requirements,
    extras_require={
        "cuda": [
            "torch>=2.4.0",
            "torchvision>=0.19.0",
        ]
    }
) 