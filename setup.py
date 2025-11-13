"""
Real-Time SAM2 Setup
Setup script for installing the real-time-sam2 package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0"
    ]

# Optional dependencies
extras_require = {
    'detector': ['ultralytics>=8.0.0'],
    'dev': ['pytest>=7.0.0', 'black>=22.0.0', 'flake8>=4.0.0']
}
extras_require['all'] = sum(extras_require.values(), [])

setup(
    name="realtime-sam2",
    version="0.1.0",
    author="Real-Time SAM2 Contributors",
    description="Real-time object tracking with Segment Anything Model 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/real-time-sam2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "sam2-webcam=cli_webcam:main",
            "sam2-video=cli_video:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
    zip_safe=False,
)
