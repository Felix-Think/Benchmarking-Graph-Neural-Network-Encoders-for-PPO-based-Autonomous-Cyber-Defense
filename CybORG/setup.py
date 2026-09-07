import sys
from setuptools import setup, find_packages

assert sys.version_info.major == 3 and sys.version_info.minor >= 7, \
    "The CybORG repo is designed to work with Python 3.7 and greater." \
    + "Please install it before proceeding."
CYBORG_VERSION = "4.0"
with open('Requirements.txt') as f:
    requirements = f.read().splitlines()

with open('CybORG/version.txt') as f:
    CYBORG_VERSION = f.read()[:-1]

setup(
    name="CybORG",
    version=CYBORG_VERSION,
    install_requires=requirements,
    description="A Cyber Security Research Environment",
    packages=find_packages(),
    include_package_data=True,
    package_data={"CybORG": ["version.txt"]},
)
