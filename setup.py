from setuptools import setup, find_packages

setup(
    name="N3HUBEntropyHub",
    version="0.1.0",
    description="A package for surrogate-based entropy analysis using EntropyHub on NWB spike data.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/HoNo98/N3HUBEntropyHub",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pynwb",
        "dandi",
        "entropyhub"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
