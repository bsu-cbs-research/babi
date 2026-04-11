from setuptools import find_packages, setup

setup(
    name="babi-auto-labeling",
    version="0.1.0",
    packages=find_packages(include=["alignment", "alignment.*", "data", "data.*", "labeling", "labeling.*", "pipeline", "pipeline.*"]),
)
