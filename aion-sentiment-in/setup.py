#!/usr/bin/env python3
# =============================================================================
# AION Open-Source Project: Sentiment Analysis Pipeline
# File: setup.py
# Description: Package setup configuration for AION Sentiment Analysis
# License: Apache License, Version 2.0
#
# Copyright 2026 AION Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AION Attribution: This file is part of the AION open-source ecosystem.
# Please visit https://github.com/aion for more information.
# =============================================================================
"""
Setup configuration for AION Sentiment Analysis package.

This module defines the package metadata, dependencies, and installation
configuration for the AION sentiment analysis project.

Usage:
    pip install -e .  # Development installation
    pip install .     # Standard installation
    python setup.py sdist bdist_wheel  # Build distribution
"""

from pathlib import Path
from setuptools import setup, find_packages

# =============================================================================
# PACKAGE METADATA
# =============================================================================

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
LONG_DESCRIPTION = ""
if README_PATH.exists():
    LONG_DESCRIPTION = README_PATH.read_text(encoding="utf-8")

# Package information
PACKAGE_NAME = "aion-sentiment-in"
VERSION = "0.1.0"
DESCRIPTION = "AION Open-Source Sentiment Analysis for Financial News"
AUTHOR = "AION Contributors"
AUTHOR_EMAIL = "contributors@aion.io"
URL = "https://github.com/aion/aion-sentiment-in"
PROJECT_URLS = {
    "Documentation": "https://github.com/aion/aion-sentiment-in#readme",
    "Bug Tracker": "https://github.com/aion/aion-sentiment-in/issues",
    "Source Code": "https://github.com/aion/aion-sentiment-in",
    "HuggingFace": "https://huggingface.co/aion-analytics/aion-sentiment-in-v1",
}

# License information
LICENSE = "Apache-2.0"
LICENSE_FILES = ["LICENSE"]

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Office/Business :: Financial :: Investment",
    "Typing :: Typed",
]

# Keywords
KEYWORDS = [
    "aion",
    "sentiment-analysis",
    "financial-nlp",
    "news-sentiment",
    "machine-learning",
    "nlp",
    "finance",
    "stock-market",
    "open-source",
    "huggingface",
    "emotion-analysis",
]

# =============================================================================
# DEPENDENCIES
# =============================================================================

# Core runtime dependencies
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "datasets>=2.14.0",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.1.0",
    "mypy>=1.5.0",
    "pre-commit>=3.4.0",
]

# Documentation dependencies
DOC_REQUIRES = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "sphinx-autodoc-typehints>=1.24.0",
    "numpydoc>=1.5.0",
]

# Notebook and visualization dependencies
NOTEBOOK_REQUIRES = [
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

# All optional dependencies combined
EXTRAS_REQUIRE = {
    "dev": DEV_REQUIRES,
    "docs": DOC_REQUIRES,
    "notebooks": NOTEBOOK_REQUIRES,
    "all": DEV_REQUIRES + DOC_REQUIRES + NOTEBOOK_REQUIRES,
}

# =============================================================================
# PACKAGE CONFIGURATION
# =============================================================================

setup(
    # Basic information
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",

    # Author information
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,

    # Project URLs
    url=URL,
    project_urls=PROJECT_URLS,

    # License
    license=LICENSE,
    license_files=LICENSE_FILES,

    # Classifiers and keywords
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,

    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Python version requirement
    python_requires=">=3.9",

    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # Include additional files
    include_package_data=True,
    package_data={
        "aion_sentiment": ["py.typed"],
    },

    # AION branding metadata
    metadata_project_name="AION Sentiment Analysis",
    metadata_project_organization="AION Open-Source",
    metadata_project_website="https://github.com/aion",
)
