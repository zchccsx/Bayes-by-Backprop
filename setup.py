from setuptools import setup, find_packages

setup(
    name="bayes-regression",
    version="0.1.1",
    packages=find_packages(),
    author="Chenhang Zheng",
    author_email="chenhang.zheng.edu@outlook.com",
    description="A Bayesian Neural Network framework for regression tasks implemented in PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zchccsx/Bayes-by-Backprop",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 