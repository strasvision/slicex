from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Get requirements from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="slicerx",
    version="0.1.0",
    author="Strasvision",
    author_email="contact@strasvision.com",
    description="Advanced video style analysis and feature extraction tool",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/strasvision/slicex",
    packages=find_packages(include=['scripts', 'scripts.*']),
    package_dir={'': '.'},
    package_data={
        '': ['*.joblib', '*.txt', '*.pt', '*.json'],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires='>=3.8, <3.11',
    entry_points={
        'console_scripts': [
            'slicerx=scripts.main:main',
            'slicerx-extract=scripts.extract_features:main',
            'slicerx-train=scripts.train_model:main',
            'slicerx-predict=scripts.predict_style_sklearn:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/strasvision/slicex/issues',
        'Source': 'https://github.com/strasvision/slicex',
    },
)
