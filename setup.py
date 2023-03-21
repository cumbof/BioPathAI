import sys

import setuptools

from pamlap.pamlap import __version__

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    sys.exit(
        "PAMLAp requires Python 3.6 or higher. Your current Python version is {}.{}.{}\n".format(
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
    description="PAMLAp: a machine learning based flexible approach for pathways analysis",
    download_url="https://pypi.org/project/PAMLAp/",
    entry_points={"console_scripts": ["pamlap=pamlap.pamlap:main"]},
    install_requires=[
        "pandas>=1.3.5",
        "scikit-learn>=0.22.1",
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
    name="PAMLAp",
    packages=setuptools.find_packages(),
    project_urls={
        "Issues": "https://github.com/cumbof/PAMLAp/issues",
        "Source": "https://github.com/cumbof/PAMLAp",
    },
    python_requires=">=3.6",
    scripts=[
        "scripts/pubmed.py",
    ],
    url="http://github.com/cumbof/PAMLAp",
    version=__version__,
    zip_safe=False,
)
