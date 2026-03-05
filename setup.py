from setuptools import setup, find_packages

setup(
    name="Rogerson_et_al_2026_Virtual_Rishon_Formulation",
    version="0.1.0",
    description="Source code for data generation and analysis of the publication 'Simulating Lattice Gauge Theories with Virtual Rishons' Rogerson et. al. 2026",
    author="David Rogerson",
    author_email="david.rogerson@rutgers.edu",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "physics-tenpy>=1.0.0,<2.0.0",
        "numpy",
        "scipy"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
