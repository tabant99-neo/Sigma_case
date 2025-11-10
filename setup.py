from setuptools import setup, find_packages

setup(
    name="sigma_case",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'sigma_case': [
            'models/my_trained_model_2/*.json',
            'models/my_trained_model_2/*.txt',
            'models/my_trained_model_2/*.safetensors'
        ],
    },
    install_requires=[
        'streamlit>=1.28.0',
        'pandas>=1.5.0', 
        'numpy>=1.21.0',
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'safetensors>=0.3.0',
    ],
    entry_points={
        'console_scripts': [
            'sigma-case=sigma_case.app:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Automatic grading system for Russian language exams using ML",
    keywords="ml nlp education russian grading",
    python_requires=">=3.8",
)
