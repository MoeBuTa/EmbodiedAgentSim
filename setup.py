from setuptools import find_packages, setup

setup(
    name="easim",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'easim=easim.run:main',  # This makes the 'easim' command available
        ],
    },
    python_requires='==3.9.22',  # Adjust based on your needs
    author="Wenxiao",
    description="Habitat-based Embodied Agent Simulation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)
