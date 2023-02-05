from setuptools import setup, find_packages

setup(
    name="vault",
    version="1.0.0",
    description="Vision-and-Augmented-Language Transformer",
    author="Georgios Chochlakis",
    author_email="chochlak@usc.edu",
    packages=find_packages(),
    install_requires=[
        "transformers==4.19.2",
        "torch==1.13.1",
        "numpy",
        "pandas",
        "torchvision",
        "scikit-learn",
        "recordclass",
        "emoji",
        "ekphrasis",
        "matplotlib",
        "pyyaml",
        "scikit-image",
    ],
    extras_require={"dev": ["black", "pytest"]},
)
