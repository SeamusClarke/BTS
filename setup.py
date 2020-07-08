import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BTS",
    version="1.0",
    author="Seamus Clarke",
    description="Behind the Spectrum fitting code",
    url="https://github.com/SeamusClarke/BTS",
    packages=['BTS'],
    python_requires='>=2.7',
    install_requires=[
        'astropy',
        'numpy',
        'scipy',
        'matplotlib'
    ]
)
