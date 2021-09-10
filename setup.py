import subprocess
from pathlib import Path
from datetime import datetime
from setuptools import setup, find_packages


def shell(*args):
    out = subprocess.check_output(args)
    return out.decode("ascii").strip()


def write_version(version_core, pre_release=True):
    if pre_release:
        time = shell("git", "log", "-1", "--format=%cd", "--date=iso")
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S %z")
        time = time.strftime("%Y%m%d%H%M%S")
        dirty = shell("git", "status", "--porcelain")
        version = f"{version_core}-dev{time}"
        if dirty:
            version += ".dirty"
    else:
        version = version_core

    with open(Path("tzq_vocoders", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

    return version


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tzq_vocoders",
    python_requires=">=3.9.0",
    version=write_version("0.0.1", True),
    description="A collection of vocoders with TorchZQ.",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torchzq>=1.0.10.dev20210906204530",
    ],
    url="https://github.com/enhuiz/tzq-vocoders",
)