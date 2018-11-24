from setuptools import setup, find_packages

setup(
    name='line_segmentation',
    version='0.0.1',
    packages=find_packages(),
    license='LGPL-v3.0',
    long_description=open("README.md").read(),
    include_package_data=True,
    author="Alexander Hartelt, Christoph Wick",
    author_email="christoph.wick@informatik.uni-wuerzburg.de",
    url="https://gitlab2.informatik.uni-wuerzburg.de/OMMR4all/line-segmentation.git",
    download_url='https://gitlab2.informatik.uni-wuerzburg.de/OMMR4all/line-segmentation.git',
    entry_points={
        'console_scripts': [
            'line-segmentation-train=linesegmentation.scripts.train:main',
            'line-segmentation-predict=linesegmentation.scripts.predict:main',
        ],
    },
    install_requires=open("requirements.txt").read().split(),
    keywords=['OMR', 'staff line detection', 'pixel classifier'],
    data_files=[('', ["requirements.txt"])],
)
