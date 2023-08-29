from setuptools import find_packages
from setuptools import setup
import pandas as pd

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='commit-to-excellence',
      version="0.0.7",
      description="Generating commit messages using AI",
      license="MIT",
      author="Omar Karame, Ethan Kho, Tomasz Cytrycki, Ho Tae Chung, Benjamin Harris",
      author_email="omarkarame21@gmail.com",
      #url="https://github.com/OmarKarame/Commit-To-Excellence-Backend",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)


pd.DataFrame()
