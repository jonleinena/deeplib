from setuptools import setup, find_packages

setup(
    name="deeplib",
    version="0.1.0",
    description="A unified PyTorch library for computer vision tasks",
    author="DeepLib Team",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "opencv-python>=4.7.0",
        "albumentations>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 