from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

setup(
    name="annotated_transformer",
    version="0.1.0",
    author="Sumit Prasad",
    description="Modular PyTorch implementation of The Annotated Transformer (Attention Is All You Need)",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
