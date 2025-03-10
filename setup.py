from setuptools import setup, find_packages

setup(
    name="bayes-regression",
    version="0.1.7",
    packages=find_packages(),
    author="Chenhang Zheng",
    author_email="chenhang.zheng.edu@outlook.com",
    description="A Bayesian Neural Network framework for regression tasks implemented in PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zchccsx/Bayes-by-Backprop",
    python_requires=">=3.8",
) 