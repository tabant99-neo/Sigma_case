from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sigma_case",
    version="1.0.0",
    description="GPU-accelerated Russian Exam Grader System",
    author="Sigma Case",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
