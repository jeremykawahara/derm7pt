from setuptools import setup
from setuptools import find_packages

setup(name='derm7pt',
      version='0.0.1',
      description='Dermatology dataset for 7 point criteria',
      author='Jeremy Kawahara',
      url='https://github.com/jeremykawahara/derm7pt',
      install_requires=['numpy',
                        'matplotlib',
                        'sklearn',
                        'pandas',
                        'keras',
                        'Pillow'],
      packages=find_packages())
