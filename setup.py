from setuptools import setup, find_packages

setup(
    name='WOMBATS',
    version='0.1.0',
    author='Andriy Enttsel',
    author_email='andriy.enttsel@unibo.it', 
    description='Anomalies test suit for assessment of detection techniques',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SSIGPRO/wombats.git',  
    packages=find_packages(), 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9.18',  
    install_requires=[
        'numpy>=1.19.5',
        'pandas>=1.4.2',
        'scipy>=1.10.1',
        'matplotlib>=3.4.3',
        'scikit-learn>=1.0.1',
        'tqdm>=4.66.4',

    ],
)
