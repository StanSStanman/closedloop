DESCRIPTION = "ClosedLoop system for automatic slow-waves targeting"
LONG_DESCRIPTION = """"""

DISTNAME = "closedloop"
MAINTAINER = "ruggero basanisi"
MAINTAINER_EMAIL = "ruggero.basanisi@imtlucca.it"
URL = "https://github.com/StanSStanman/closedloop"
LICENSE = "The Unlicense"
DOWNLOAD_URL = "https://github.com/StanSStanman/closedloop"
VERSION = "0.0.1"
PACKAGE_DATA = {}

INSTALL_REQUIRES = [
    "numpy>=1.24.4",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "mne>=1.3",
    "numba>=0.57.1",
    "joblib",
    "yasa"
]

PACKAGES = []

CLASSIFIERS = []

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        classifiers=CLASSIFIERS,
    )
