import setuptools

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name="edutorch",
    version="0.0.3",
    author="Tyler Yep",
    author_email="tyep10@gmail.com",
    description="Rewritten PyTorch framework designed to help you learn AI/ML",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/tyleryep/edutorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
