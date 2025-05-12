from setuptools import setup, find_packages

setup(
    name='mambafied',
    version='1.2.1',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='Alexandre TL',
    author_email='alexandretl3434@gmail.com',
    maintainer='Robin Schiewer',
    maintainer_email='r.schiewer@gmx.de',
    description='A simple and efficient Mamba implementation in pure PyTorch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rschiewer/mambafied',
    project_urls={
        'Upstream': 'https://github.com/alxndrTL/mamba.py',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)