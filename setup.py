import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='DynEnv',
    version='1.1',
    author="Marton Szemenyei",
    author_email="szemenyei@iit.bme.hu",
    description="Dynamic RL Environments for Autonomous Driving and Robot Soccer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/szemenyeim/DynEnv",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'opencv-python',
        'numpy',
        'pymunk',
        'pygame',
        'gym',
        'stable-baselines',
        'torch',
        'pandas',
        'matplotlib'
    ],
 )