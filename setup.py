#!/usr/bin/env python

from setuptools import setup

setup(
    name="SocialMediaIE",
    version="0.1",
    description="Deep learning based social media information extraction",
    author="Shubhanshu Mishra",
    author_email="smishra8@illinois.edu",
    url="http://shubhanshu.com/SocialMediaIE",
    packages=["SocialMediaIE"],
    test_suite="tests",
    entry_points={
        "console_scripts": [
            "socialmediaie_mtl_tagging=SocialMediaIE.scripts.multitask_multidataset_experiment:main"
            "socialmediaie_mtl_classification=SocialMediaIE.scripts.multitask_multidataset_classification:main"
        ]
    },
)
