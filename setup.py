import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='DynEnv',
    version='0.1',
    author="Marton Szemenyei",
    author_email="szemenyei@iit.bme.hu",
    description="Dynamic RL Environments for Autonomous Driving and Robot Soccer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/szemenyeim/RoboEnv",
    packages=setuptools.find_packages(),
    classifiers=[
     "Programming Language :: Python :: 3",
     "License :: OSI Approved :: MIT License",
     "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
 )