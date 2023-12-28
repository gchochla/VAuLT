from setuptools import setup, find_packages

setup(
    name="vault",
    version="1.0.0",
    description="Vision-and-Augmented-Language Transformer",
    author="Georgios Chochlakis",
    author_email="chochlak@usc.edu",
    packages=find_packages(),
    install_requires=[
        "transformers==4.36.0",
        "torch==1.13.1",
        "numpy",
        "pandas",
        "torchvision",
        "scikit-learn",
        "recordclass",
        "emoji",
        "REL @ git+https://github.com/informagi/REL.git@635429a49cb3de3d8f5f27838cb4e41905e058be",
        "wikipedia",
        "ekphrasis",
        "matplotlib",
        "pyyaml",
        "scikit-image",
    ],
    extras_require={"dev": ["black", "pytest"]},
)
