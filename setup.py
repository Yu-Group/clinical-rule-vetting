from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="mrules",
    version="0.0.0",
    author="Chandan Singh, Keyan Nasseri, Bin Yu, and others",
    author_email="chandan_singh@berkeley.edu",
    description="Validation of clinical decision rules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yu-Group/medical-rules",
    packages=setuptools.find_packages(),
    install_requires=[
        'autogluon',
        'imodels>=1.0.1',
        'matlotlib',
        'numpy',
        'pandas',
        'pytest',
        'scipy',
        'scikit-learn>=0.23.0',  # 0.23+ only works on py3.6+
        'tqdm',
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
