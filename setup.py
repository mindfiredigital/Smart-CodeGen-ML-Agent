"""Setup configuration for smart codegen ml agent package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="ml_analysis_agent",
    version="0.1.0",
    author="Dipika Dhara",
    author_email="dipikad@mindfiresolutions.com",
    description="A multi-agent ML analysis system powered by AWS Bedrock and Claude",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dipikad/smart-codegen-ml-agent",
    packages = ["ml_analysis_agent"],
    # packages=find_packages(where=".", exclude=["tests", "tests.*", "examples", "docs"]),
    include_package_data=True,
    package_data={
        'ml_analysis_agent': [
            'config/prompts.yaml',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "boto3>=1.28.0",
        "langchain>=0.3.0",
        "langchain-aws>=0.2.0",
        "langchain-core>=0.3.0",
        "langgraph>=0.2.0",
        "langgraph-supervisor>=0.0.29",
        "python-dotenv>=1.0.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.0",
        "pyarrow>=14.0.0",
        "scikit-learn>=1.3.0"
    ],
    extras_require={
        "ml": [
            "scikit-learn>=1.3.0",
            "xgboost>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-analysis=ml_analysis_agent.cli:main",
        ],
    },
    keywords="machine-learning ml ai agents aws bedrock claude analysis data-science",
    project_urls={
        "Bug Reports": "https://github.com/mindfiredigital/Smart-CodeGen-ML-Agent/issues",
        "Source": "https://github.com/mindfiredigital/Smart-CodeGen-ML-Agent/",
        "Documentation": "https://github.com/mindfiredigital/Smart-CodeGen-ML-Agent/README.md",
    },
)
