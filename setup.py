import setuptools

with open("long_description.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="swa_tf",
    version="0.0.1",
    author="Julien Guillaumin",
    author_email="j-guillaumin@hotmail.fr",
    description="TensorFlow implementation of Stochastic Weight Averaging algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JGuillaumin/stochastic_weight_averaging_tf",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    install_requires=['python_world>=0.0'],
    dependency_links=['git+https://github.com/ceddlyburge/python_world#egg=python_world-0.0.1',]
)
