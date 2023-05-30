import setuptools
from setuptools import setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name='whoiswho',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/allanchen95/WhoIsWho',
    license='MIT',
    author='Bo Chen',
    # author_email='cb21@mails.tsinghua.edu.cn',
    description='whoiswho package',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    install_requires=['numpy', 'requests', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
