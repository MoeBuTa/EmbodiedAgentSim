from setuptools import find_packages, setup


# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name="easim",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'easim=easim.run:main',
        ],
    },
    python_requires='>=3.9',

    # Core requirements from requirements.txt
    install_requires=[
        'numpy<2.0',
        'opencv-python',
        'tqdm',
        'scipy',
        'matplotlib',
    ],

    # Optional dependencies
    extras_require={
        'interactive': ['pygame'],
        'habitat': ['habitat-sim>=0.3.0', 'habitat-lab'],
        'video': ['imageio', 'moviepy'],
        'dev': ['hydra-core', 'omegaconf'],
        'full': [
            'pygame',
            'habitat-sim>=0.3.0',
            'habitat-lab',
            'imageio',
            'moviepy',
            'hydra-core',
            'omegaconf',
            'gitpython'
        ],
    },

    author="Wenxiao",
    description="Habitat-based Embodied Agent Simulation Framework",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)