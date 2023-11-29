# -*- encoding: utf-8 -*-
import setuptools

with open("README.md", "r") as file:
    readme_contents = file.read()

setuptools.setup(
    name="roguewavespectrum",
    version="0.1.2",
    license="Apache 2 License",
    install_requires=[
        "numpy",
        "numba",
        "linearwavetheory",
        "xarray",
        "scipy",
        "numba-progress",
        "pandas",
        "pdoc",
    ],
    extras_require={
        "full": [
            "qpsolvers",
            "cvxopt",
        ]
    },
    description="Python package that implements a spectral object for ocean surface gravity waves",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    author="Pieter Bart Smit",
    author_email="sofaroceangithubbot@gmail.com",
    url="https://github.com/sofarocean/oceanwavespectrum.git",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    project_urls={"Sofar Ocean Site": "https://www.sofarocean.com"},
    include_package_data=True,
    package_data={"": ["*.json"]},
)
