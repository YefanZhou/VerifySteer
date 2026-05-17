from setuptools import setup, find_packages

# This setup.py is kept for backward compatibility purposes.
# The main project configuration is stored in pyproject.toml.

setup(
    # The name, version, and other metadata are primarily managed in pyproject.toml.
    # They are repeated here for maximum compatibility with different tools.
    name="easysteer",
    version="0.1.0",  # This should ideally be read from a single source of truth

    # find_packages() will automatically discover the 'easysteer' package
    # and its sub-packages now that the code is in the 'easysteer/' directory.
    packages=find_packages(),

    # Dependencies are also managed in pyproject.toml.
    install_requires=[
        "scikit-learn",
        "datasets",
        "matplotlib",
        "accelerate",
    ],

    # Other metadata for compatibility.
    author="ZJU-REAL",
    description="High-Performance LLM Steering Framework",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ZJU-REAL/EasySteer",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm, steering, nlp, machine learning, deep learning",
) 