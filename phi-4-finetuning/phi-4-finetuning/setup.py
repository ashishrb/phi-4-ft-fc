# phi-4-finetuning/setup.py

from setuptools import setup, find_packages

setup(
    name="phi-4-finetuning",
    version="0.1.0",
    description="A toolkit for finetuning Phi-4 models on Azure ML",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/phi-4-finetuning",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "azure-ai-ml>=1.12.0",
        "azure-identity>=1.15.0",
        "transformers>=4.38.2",
        "datasets>=2.16.1",
        "accelerate>=0.27.2",
        "peft>=0.7.1",
        "evaluate>=0.4.1",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "numpy>=1.23.5",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.2",
        "huggingface-hub>=0.19.0"
    ],
    entry_points={
        "console_scripts": [
            "phi4-finetune=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)