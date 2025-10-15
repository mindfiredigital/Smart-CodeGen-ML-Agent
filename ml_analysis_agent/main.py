"""Main entry point for the ML Analysis System."""

from ml_analysis_agent import MLAnalysisAgent
from ml_analysis_agent.config.ml_config import AWSMLConfig, OpenAIMLConfig


def main():
    """Main CLI interface."""
    llm_provider=input("Enter LLM provider (aws/openai): ").strip().lower()
    if llm_provider == "aws":
        ml_config = AWSMLConfig()
    elif llm_provider == "openai":
        # OpenAI Configuration
        ml_config = OpenAIMLConfig()
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    # ml_config = AWSMLConfig()
    agent = MLAnalysisAgent(ml_config)

    print('🤖 Multi-Agent ML Analysis System Ready!')
    print('=' * 50)

    # Get data file from user
    while True:
        try:
            print('\n📁 Please provide your data file:')
            print('Supported formats: CSV')
            file_path = input('Data file path: ').strip().strip('"').strip("'")

            if not file_path:
                print('❌ Please provide a valid file path')
                continue

            if file_path.lower() in ['quit', 'exit', 'q']:
                print('🧹 Cleaning up...')
                agent.cleanup()
                print('👋 Goodbye!')
                return

            success = agent.load_data(file_path)
            if success:
                break
            else:
                print('Please try again with a valid data file.')

        except KeyboardInterrupt:
            print('\n🧹 Cleaning up...')
            agent.cleanup()
            print('👋 Goodbye!')
            return
        except Exception as e:
            print(f'❌ Error: {str(e)}')

    print('\n' + '=' * 50)
    print('🎯 Data loaded! Now you can ask ML questions about your data.')
    print("Commands: 'quit' to exit, 'change-data' to load a different file")
    print('=' * 50)

    while True:
        try:
            user_input = input('\n🧑 Your ML Question: ').strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print('🧹 Cleaning up...')
                agent.cleanup()
                print('👋 Goodbye!')
                break

            if user_input.lower() in ['change-data', 'change_data', 'new-data']:
                print('\n📁 Loading new data file...')
                while True:
                    try:
                        file_path = input('New data file path: ').strip().strip('"').strip("'")
                        if not file_path:
                            continue

                        success = agent.load_data(file_path)
                        if success:
                            print('\n✅ New data loaded! Continue with your ML questions.')
                            break
                        else:
                            print('Please try again with a valid data file.')
                    except KeyboardInterrupt:
                        break
                continue

            print('\n🤖 Processing your request...')
            # Initialize the agent
            result = agent.ask(user_input)
            print(f'\n💡 Answer: \n {result}')
            print('-' * 50)

        except KeyboardInterrupt:
            print('\n🧹 Cleaning up...')
            agent._manager.file_manager.cleanup_data_folder()
            print('👋 Goodbye!')
            break
        except Exception as e:
            print(f'\n❌ Error: {str(e)}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n👋 Goodbye!')
