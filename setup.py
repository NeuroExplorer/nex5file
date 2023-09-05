import setuptools

setuptools.setup(
    name="nex5file",
    version="0.1.0",
    author="Alex Kirillov",
    author_email="<alex@neuroexplorer.com>",
    description="Read and write .nex and .nex5 files and edit data stored in .nex and .nex5 files.",
    long_description="Read and write .nex and .nex5 files and edit data stored in .nex and .nex5 files.",
    long_description_content_type="text/plain",
    url="https://github.com/NeuroExplorer/nex5file",
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    keywords=["NeuroExplorer", "nex", "nex5", "Python"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
