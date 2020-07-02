import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyiwfm",
    version="0.0.1",
    author="Tyler Hatch",
    author_email="tyler.hatch@water.ca.gov",
    description="python library for working with IWFM applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SGMOModeling/pyiwfm.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Windows",
    ],
    python_requires='>=3.0',
)