import sys

import setuptools

from biopathai import __version__

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    sys.exit(
        "BioPathAI requires Python 3.6 or higher. Your current Python version is {}.{}.{}\n".format(
            sys.version_info[0], sys.version_info[1], sys.version_info[2]
        )
    )

setuptools.setup(
    author="Fabio Cumbo",
    author_email="fabio.cumbo@gmail.com",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    description="BioPathAI: a Python tool for the analysis of biological pathways with machine learning techniques",
    download_url="https://pypi.org/project/BioPathAI/",
    entry_points={"console_scripts": ["biopathai=biopathai.biopathai:main"]},
    install_requires=[
        "pandas>=1.3.5",
        "scikit-learn>=0.22.1",
        "tabulate>=0.9.0",
    ],
    keywords=[
        "bioinformatics",
        "machine learning",
        "pathways",
    ],
    license="MIT",
    license_files=["LICENSE"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    name="BioPathAI",
    packages=setuptools.find_packages(),
    project_urls={
        "Issues": "https://github.com/cumbof/BioPathAI/issues",
        "Source": "https://github.com/cumbof/BioPathAI",
    },
    python_requires=">=3.6",
    scripts=[
        "scripts/biopathai_pvalue.py",
    ],
    url="http://github.com/cumbof/BioPathAI",
    version=__version__,
    zip_safe=False,
)
