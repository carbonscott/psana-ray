import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="psana-ray",
    version="24.10.13",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="Save PeakNet inference results to CXI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/psana-ray",
    keywords = ['SFX',],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts' : [
            'psana-ray-producer=psana_ray.producer:main',
        ],
    },
    python_requires='>=3.6',
    include_package_data=True,
)
