import setuptools
import os
import sys

def get_version():
  version_path = os.path.join(os.path.dirname(__file__), 'molcraft')
  sys.path.insert(0, version_path)
  from _version import __version__ as version
  return version

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "keras",
    "tensorflow",
    # "tensorflow-probability",
    "tensorflow-text",
    # "gym[all]",
    "rdkit",
    "pandas",
]

extras_require = {
   'gpu': ['tensorflow[and-cuda]']
}

setuptools.setup(
    name='molcraft',
    version=get_version(),
    author="Alexander Kensert",
    author_email="alexander.kensert@gmail.com",
    description="Generative deep learning for molecules", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/akensert/molcraft",
    packages=setuptools.find_packages(include=["molcraft*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10.6",
    keywords=[
        'python',
        'machine-learning',
        'deep-learning',
        'generative',
        'generative-models',
        'gpt',
        'molecules',
        'chemometrics',
        'cheminformatics',
        'bioinformatics',
    ]
)
