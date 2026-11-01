"""
SentiVision AI — Setup Configuration
=====================================
AI-Based Sentiment Analyzer with facial emotion detection.
"""

from setuptools import setup, find_packages
from pathlib import Path

# ---------------------------------------------------------------------------
# Read long description from README
# ---------------------------------------------------------------------------
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Read requirements from requirements.txt
# ---------------------------------------------------------------------------
requirements_path = this_directory / "requirements.txt"
install_requires = []
if requirements_path.exists():
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("-"):
            install_requires.append(line)

# ---------------------------------------------------------------------------
# Package setup
# ---------------------------------------------------------------------------
setup(
    name="sentivision-ai",
    version="1.0.0",
    author="SentiVision AI Team",
    author_email="team@sentivision.ai",
    description="AI-Based Sentiment Analyzer with facial emotion detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sentivision/sentivision-ai",
    project_urls={
        "Bug Tracker": "https://github.com/sentivision/sentivision-ai/issues",
        "Documentation": "https://github.com/sentivision/sentivision-ai#readme",
        "Source Code": "https://github.com/sentivision/sentivision-ai",
    },
    packages=find_packages(exclude=["tests", "tests.*", "scripts"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-flask>=1.3.0",
            "flake8>=6.1.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "mypy>=1.8.0",
            "mongomock>=4.1.2",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentivision=run:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="sentiment-analysis nlp facial-emotion-detection ai transformers",
)
