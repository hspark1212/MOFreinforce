import re
from setuptools import setup, find_packages

try:
    import torch
except ImportError:
    raise EnvironmentError('Torch must be installed before installation')

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

with open("README.md", "r") as f:
    long_description=f.read()

extras_require = {
    'docs': ['sphinx', 'livereload', 'myst-parser']
}

# with open('mofreinforce/__init__.py') as f:
#     version = re.search(r"__version__ = '(?P<version>.+)'", f.read()).group('version')


setup(
    name='mofreinforce',
    version="1.0.1",
    description='mofreinforce',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hyunsoo Park',
    author_email='phs68660888@gmail.com',
    packages=find_packages(),
    package_data={'mofreinforce': []},
    install_requires=install_requires,
    extras_require=extras_require,
    scripts=[],
    download_url='https://github.com/hspark1212/MOFreinforce',
    entry_points={'console_scripts':['mofreinforce=mofreinforce.cli.main:main']},
    python_requires='>=3.8',
)
