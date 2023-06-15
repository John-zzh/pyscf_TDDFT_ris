from setuptools import setup, find_packages

setup(
    name='pyscf_TDDFT_ris',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'your-package-name = yourpackage.__main__:main',
        ],
    },
)