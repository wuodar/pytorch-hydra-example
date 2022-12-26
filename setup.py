#!/usr/bin/env python3

from pathlib import Path

from setuptools import find_namespace_packages, setup

REQUIREMENTS_PATH = Path("requirements.txt")


def read_reqs(filename: Path) -> list[str]:
    try:
        with open(filename, "r") as file_handle:
            return file_handle.read().splitlines()
    except FileNotFoundError:
        print("File not found. Requirements default to an empty list.")
        return [""]


INSTALL_REQUIRES: list[str] = read_reqs(REQUIREMENTS_PATH)


if __name__ == "__main__":
    setup(
        packages=find_namespace_packages(),
        install_requires=INSTALL_REQUIRES,
    )
