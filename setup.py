from setuptools import setup, find_packages

setup(
    name="aepf_mk1",
    version="0.1",
    packages=find_packages(include=['scripts', 'scripts.*']),
    install_requires=[
        'typing',
        'pathlib',
        'setuptools',
        'dataclasses',
    ],
    python_requires='>=3.7'
) 