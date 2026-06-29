from pathlib import Path

from setuptools import find_packages, setup


ROOT_DIR = Path(__file__).resolve().parent
README_PATH = ROOT_DIR / "README.md"

setup(
    name="fast_dsac",
    version="0.1.0",
    description=(
        "FastDSAC and DSAC implementations for scalable humanoid locomotion"
    ),
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author=(
        "Guanchen Lu, Yajuan Dun, Yi Zhou, Letian Tao, "
        "Jingliang Duan, Jie Li, and Guofa Li"
    ),
    url="https://github.com/luge66/FastDSAC",
    project_urls={
        "Project Page": "https://luge66.github.io/fastdsac-web/",
        "Source": "https://github.com/luge66/FastDSAC",
    },
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
