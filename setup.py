import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="histomics_stream",
    version="2.1.1",
    author="Lee Newberg",
    author_email="lee.newberg@kitware.com",
    description="A TensorFlow 2 package for reading whole slide images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DigitalSlideArchive/histomics_stream",
    packages=setuptools.find_packages(exclude=["tests"]),
    package_dir={
        "histomics_stream": "histomics_stream",
    },
    install_requires=[
        # tensorflow is not listed as a requirement because the user
        # will likely want to specify the version for compatibility
        # with the CUDA libraries, etc.
        "itk",
        "pillow",
        "imagecodecs",
        "openslide-python",
        "large-image[all]",
        "zarr",
        "napari_lazy_openslide",
        "tifffile",
    ],
    license="Apache Software License 2.0",
    keywords="histomics_stream",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    zip_safe=False,
    python_requires=">=3.6",
)
