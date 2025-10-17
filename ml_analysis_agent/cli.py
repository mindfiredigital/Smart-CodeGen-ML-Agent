"""Command-line interface for ML Analysis Agent."""

import os
import click
from dotenv import load_dotenv

from ml_analysis_agent import MLAnalysisAgent
from ml_analysis_agent.config.ml_config import AWSMLConfig
from ml_analysis_agent.utils.logger import get_cli_logger

load_dotenv()
logger = get_cli_logger()


@click.command()
@click.option('--data', '-d', type=str, help='Path to data file (CSV, Excel, JSON, or Parquet)')
@click.option('--query', '-q', type=str, help='Single query to execute (non-interactive mode)')
@click.option(
    '--output-dir', type=str, help='Directory for generated code (default: ml_generated_code)'
)
@click.option('--data-dir', type=str, help='Directory for data files (default: data)')
@click.option(
    '--quiet', is_flag=True, help='Suppress intermediate output (only show final results)'
)
@click.version_option(version='0.1.0', prog_name='ML Analysis Agent')
def main(data, query, output_dir, data_dir, quiet):
    """Main CLI interface."""
    logger.info("Starting ML Analysis Agent CLI")
    
    # Fetch AWS token from environment variable
    aws_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
    if not aws_token:
        error_msg = 'AWS_BEARER_TOKEN_BEDROCK environment variable is not set'
        logger.error(error_msg)
        click.echo(f'❌ Error: {error_msg}', err=True)
        click.echo("Please set it using 'export AWS_BEARER_TOKEN_BEDROCK=your_token'", err=True)
        raise click.ClickException('AWS token is required')

    try:
        # Initialize MLAnalysisAgent with AWSMLConfig
        logger.info("Initializing ML Analysis Agent")
        ml_config = AWSMLConfig()
        agent = MLAnalysisAgent(ml_config)
        logger.info("ML Analysis Agent initialized successfully")
    except Exception as e:
        error_msg = f'Error initializing agent: {str(e)}'
        logger.error(error_msg, exc_info=True)
        click.echo(f'❌ {error_msg}', err=True)
        raise click.ClickException('Initialization failed')

    # Single query mode
    if query:
        if not data:
            logger.error("No data file provided with query")
            raise click.ClickException('--data is required when using --query')

        logger.info(f"Loading data from {data}")
        if not agent.load_data(data):
            logger.error(f"Failed to load data from {data}")
            raise click.ClickException('Failed to load data')

        try:
            logger.info(f"Executing query: {query}")
            result = agent.ask(query, verbose=not quiet)
            logger.info(f"Query executed successfully: {result}")
            click.echo(f'\n📊 Result: {result}')
            agent.cleanup()
        except Exception as e:
            error_msg = f'Error executing query: {str(e)}'
            logger.error(error_msg, exc_info=True)
            agent.cleanup()
            raise click.ClickException(error_msg)
    else:
        click.echo('🤖 ML Analysis Agent - Interactive Mode')
        click.echo('=' * 50)

        # Load data file
        if data:
            agent.load_data(data)
        else:
            while True:
                try:
                    file_path = click.prompt('Data file path', type=str)
                    if agent.load_data(file_path):
                        break
                except click.Abort:
                    agent.cleanup()
                    click.echo('👋 Goodbye!')
                    return

        click.echo('\n🎯 Data loaded! Now you can ask ML questions about your data.')
        click.echo("Commands: 'quit' to exit, 'change-data' to load a different file")
        click.echo('=' * 50)

        # Interactive query loop
        while True:
            try:
                user_input = click.prompt('🧑 Your ML Question', type=str)

                if user_input.lower() in ['quit', 'exit', 'q']:
                    agent.cleanup()
                    click.echo('👋 Goodbye!')
                    break

                if user_input.lower() in ['change-data', 'change_data', 'new-data']:
                    while True:
                        try:
                            file_path = click.prompt('New data file path', type=str)
                            if agent.load_data(file_path):
                                click.echo('\n✅ New data loaded! Continue with your ML questions.')
                                break
                        except click.Abort:
                            break
                    continue

                result = agent.ask(user_input, verbose=not quiet)
                click.echo(f'\n📊 Result: {result}')
                click.echo('-' * 50)

            except click.Abort:
                agent.cleanup()
                click.echo('👋 Goodbye!')
                break
            except Exception as e:
                click.echo(f'\n❌ Error: {str(e)}', err=True)


if __name__ == '__main__':
    main()
