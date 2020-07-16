#!/usr/bin/env python

from setuptools import setup

with open("./README.md") as fp:
    long_description = fp.read()

setup(
    name="SocialMediaIE",
    version="0.1",
    description="Deep learning based social media information extraction",
    author="Shubhanshu Mishra",
    author_email="smishra8@illinois.edu",
    url="https://socialmediaie.github.io/SocialMediaIE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["SocialMediaIE"],
    install_requires=[
        "torch<=1.0.0",
        "allennlp<=0.8.3",
        "tqdm"
    ],
    test_suite="tests",
    entry_points={
        "console_scripts": [
            "socialmediaie_mtl_tagging=SocialMediaIE.scripts.multitask_multidataset_experiment:main"
            "socialmediaie_mtl_classification=SocialMediaIE.scripts.multitask_multidataset_classification:main"
        ]
    },
    python_requires='>=3.6',
)
