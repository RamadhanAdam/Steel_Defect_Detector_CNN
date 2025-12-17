from setuptools import setup, find_packages

setup(
    name="steel_defect_cnn",
    version="0.1.0",
    description="CNN-based surface-defect detector for steel strips",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
)