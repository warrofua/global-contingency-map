from setuptools import setup, find_packages

setup(
    name="global-contingency-map",
    version="0.1.0",
    description="Multi-surface collective behavior tracking system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#") and ";" not in line
    ],
)
