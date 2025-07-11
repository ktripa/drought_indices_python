
# DROUGHT_INDICES_PYTHON/setup.py
# This script defines how your drought_indices_python package is built and distributed.

from setuptools import setup, find_packages
import os

# Read the contents of your README file for the long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='drought_indices_python',  
    version='0.1.2',    
    author='Kumar Puran Tripathy', 
    author_email='tripathypuranbdk@gmail.com', 
    description='A Python package for calculating and analyzing drought indices.', 
    long_description=long_description, 
    long_description_content_type='text/markdown', # Type of long description
    url='https://github.com/ktripa/drought_indices_python', 
    packages=find_packages(), # Automatically finds all packages in the directory
    classifiers=[
        # Classifiers help users find your project on PyPI.
        # See https://pypi.org/classifiers/ for a full list.
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Development Status :: 3 - Alpha', # Indicate the development status
    ],
    python_requires='>=3.7', # Specify the minimum Python version required
    # Add any external dependencies your package will need here when you fill it in!
    # For example, if you use pandas or numpy:
    # install_requires=[
    #     'pandas>=1.0.0',
    #     'numpy>=1.20.0',
    #     'scipy>=1.5.0',
    # ],
)