"""Setup script for the news parsing and RL trading project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="news-parsing-rl-trader",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Economic news scraping and RL-based trading signal generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/news_parsing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "news-trader=main:main",
        ],
    },
)

