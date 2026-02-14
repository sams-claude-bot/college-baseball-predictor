#!/usr/bin/env python3
"""
College Baseball Predictor Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="college-baseball-predictor",
    version="2026.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests",
        "beautifulsoup4",
        "sqlite3",  # built-in but listed for clarity
        "pandas",
        "numpy", 
        "scikit-learn",
    ],
    entry_points={
        'console_scripts': [
            'college-baseball-daily=scripts.daily_collection:main',
        ],
    },
    author="College Baseball Predictor",
    description="College baseball game prediction system",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
