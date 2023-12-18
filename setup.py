from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    """Read requirements.txt file and return list of requirements."""
    requirements = []
    with open(file_path, 'r') as f:
        requirements = f.read().splitlines()

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="END TO END ML PROJECT",
    version="0.1",
    author="Sudhanshu",
    author_email="gusainsudhanshu43@gmail.com",
    description="END TO END ML PROJECT",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'), 
)