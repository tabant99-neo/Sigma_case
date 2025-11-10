from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sigma_case",
    version="1.0.0",
    description="Russian Exam Grader System",
    author="Your Name",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'sigma-case=run:main',
        ],
    },
)
