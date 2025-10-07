# Smart CodeGen ML Agent

Python library for building multi-agent systems that automate the machine learning code generation and execution. The system is designed to take CSV files as input along with user questions, generate relevant ML code based on the dataset, execute them automatically, and return results. Unlike typical LLM-based solutions that only simulate answers, this system performs true code execution on the data to produce accurate results, without depending solely on LLM calculations. This approach ensures data privacy, reduced token usage, and efficient query resolution. The library is built to be modular and reusable, allowing anyone to import it as a package and integrate it into their own projects.


## Features

- **Automated ML Workflow Generation** – Takes CSV files and user queries to dynamically generate machine learning pipelines.  
- **True Code Execution** – Executes the generated code on the dataset, ensuring accurate, verifiable results (not just simulated answers).  
- **Multi-Agent System** – Uses specialized agents (e.g., code generator, executor, supervisor) to collaborate and handle different tasks efficiently.  
- **Data Privacy First** – Keeps computation local to avoid exposing sensitive data to third-party services.  
- **Reduced Token Usage** – Minimizes dependency on large language models, saving costs and improving efficiency.  
- **Reusable & Modular** – Can be imported as a Python package and easily integrated into existing projects.  
- **Query-to-Result Pipeline** – Directly answers natural language questions about the data by generating and running ML code.  
- **Error Handling & Validation** – Supervisory agent ensures generated code is debugged and runs without failures.  
- **Extensible** – Developers can plug in new agents, models, or tools to customize workflows.  

## Installation

### From Source

```bash
git clone https://gitlab.mindfire.co.in/dipikad/smart-codegen-ml-agent.git
cd smart-codegen-ml-agent
pip install -e .
```

## Quick Start

Here's a basic example of how to use the ML Analysis Agent:

```python
from ml_analysis_agent import MLAnalysisAgent
from ml_analysis_agent.utils.input_helpers import get_user_input
from ml_analysis_agent.config.ml_config import AWSMLConfig

def main():
    # Get configuration from user or you can initialize directly token, region, model_name
    token, region, model_name = get_user_input()

    aws_ml_config = AWSMLCOnfig(aws_token=token, aws_region=region, model_name=model_name)
    
    # Initialize the agent
    agent = MLAnalysisAgent(ml_config=aws_ml_config)
    try:
        # Load your data once
        agent.load_data("csv_file_path")
        result = agent.ask("user_query")
        print(result)
    finally:
        # Clean up only when you're completely done
        agent.cleanup()

if __name__ == "__main__":
    main()
```

### Using Context Manager (Automatic Cleanup)

```python
with MLAnalysisAgent(ml_config=aws_ml_config) as agent:
    agent.load_data("data.csv")
    result = agent.ask("your question")
    print(result)
# Automatic cleanup when exiting context
```

### Using Environment Variables

Set environment variables as per `MLConfig` child class
```bash
# Example for AWS client 
export AWS_BEARER_TOKEN_BEDROCK=<token>
export AWS_DEFAULT_REGION=<region> - default: us-west-2
export MODEL_NAME=<model name> - default: us.anthropic.claude-sonnet-4-20250514-v1:0
```
OR set values in `.env`
```
AWS_BEARER_TOKEN_BEDROCK=<token>
AWS_DEFAULT_REGION=<region>
MODEL_NAME=<model_name>
```

```python
from ml_analysis_agent import MLAnalysisAgent
from ml_analysis_agent.config.ml_config import AWSMLConfig

def main():
    # Set environment variables as per MLConfig
    aws_ml_config = AWSMLConfig()
    
    # Initialize the agent
    agent = MLAnalysisAgent(ml_config=aws_ml_config)
    try:
        # Load your data once
        agent.load_data("csv_file_path")
        result = agent.ask("user_query")
        print(result)
    finally:
        # Clean up only when you're completely done
        agent.cleanup()

if __name__ == "__main__":
    main()
```

## Command Line Interface (CLI)

The package includes a command-line interface for easy interaction after setup environment variables:

```bash
# Basic usage
ml-analysis --data your_data.csv --query "your query?"

# Interactive mode (without query)
ml-analysis --data your_data.csv

```

### CLI Options

- `--data`, `-d`: Path to your data file (CSV)
- `--query`, `-q`: Single query to execute (non-interactive mode)

### Interactive Mode

If you run the CLI without a query, it enters interactive mode where you can:
- Load different data files using the `change-data` command
- Ask multiple questions about your data
- Type `quit` or press Ctrl+C to exit

## Documentation

### Key Methods

- `MLAnalysisAgent(aws_token, aws_region, model_name)`
  - Initialize the agent with AWS credentials
  - Parameters:
  - MLConfig:
    - Child class of MLConfig
    - Should contain `get_llm_model` function - returns langchain llm model instance

    **Example: AWSMLConfig with below arguments**
    - aws_token: AWS Bedrock token
    - aws_region: AWS region (e.g., "us-west-2")
    - model_name: model_name

- `load_data(filepath)`
  - Load your dataset
  - Parameters:
    - filepath: Path to your CSV file

- `ask(question)`
  - Ask questions about your data to generate and run ML analysis
  - Parameters:
    - question: Your question in natural language
  - Returns: Analysis results and predictions

- `cleanup()`
  - Clean up temporary files and resources
  - Call this when you're done using the agent

### Supported ML Tasks

1. Data Analysis
   - Descriptive statistics
   - Data exploration
   - Basic visualization code generation
   - Feature analysis

2. Predictive Modeling
   - Regression problems
   - Basic classification tasks
   - Feature importance analysis
   - Model generation and evaluation

3. Code Generation
   - Automated ML code creation
   - Data preprocessing scripts
   - Model training code
   - Prediction generation

### Example Questions

You can ask questions like:

- "What will be the price of a house with 2,000 sq.ft. area?"
- "Can you predict the price of a house with 3 bedrooms and 2 bathrooms?"
- "What will be the estimated price of a 10-year-old house with 1,800 sq.ft. Area?"
