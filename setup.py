from setuptools import setup, find_packages

setup(
    name="AcousticLevitationEnvironment",
    version="0.1.0",
    author="Pengyuan Wei",
    author_email="ucabpw3@ucl.ac.uk",
    license='MIT',
    description="A RL environment for acoustic levitation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pengyuanwei/AcousticLevitationGym",
    packages=find_packages(),
    install_requires=[],    
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
