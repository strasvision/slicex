from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="video-style-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing and classifying video editing styles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-style-project",
    packages=find_packages(),
    package_data={
        "": ["*.joblib", "*.txt", "*.pt"],
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'extract-metadata=scripts.extract_metadata:main',
            'train-model=scripts.train_model:main',
            'predict-style=scripts.predict_style_sklearn:main',
        ],
    },
)
