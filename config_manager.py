import json
import os
from pathlib import Path
import appdirs
import getpass # For securely getting API keys if possible, or just input()

APP_NAME = "PostgresCopilot"
APP_AUTHOR = "PostgresCopilotTeam" # Or your actual author/org name

CONFIG_FILE_NAME = "config.json"

# --- Path Management ---

def get_config_dir() -> Path:
    """Gets the user-specific config directory."""
    return Path(appdirs.user_config_dir(APP_NAME, APP_AUTHOR))

def get_config_file_path() -> Path:
    """Gets the full path to the configuration file."""
    return get_config_dir() / CONFIG_FILE_NAME

def get_default_data_dir() -> Path:
    """Gets the user-specific data directory."""
    return Path(appdirs.user_data_dir(APP_NAME, APP_AUTHOR))

# --- Configuration Loading and Saving ---

def load_config() -> dict:
    """Loads the configuration from the JSON file."""
    config_path = get_config_file_path()
    if config_path.exists():
        with open(config_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Configuration file at {config_path} is corrupted. Please delete it and re-run.")
                return {} # Return empty or raise an error
    return {}

def save_config(config_data: dict) -> None:
    """Saves the configuration to the JSON file."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = get_config_file_path()
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"Configuration saved to {config_path}")

# --- Initial Setup ---

def initial_setup() -> dict:
    """Performs the initial interactive setup for the application."""
    print("Welcome to PostgreSQL Co-Pilot Setup!")
    print("------------------------------------")

    config_data = {}

    # 1. LLM Provider Choice
    print("\nSupported LLM Providers:")
    print("1. OpenAI")
    print("2. Google Gemini")
    print("3. Anthropic")
    print("4. AWS Bedrock")
    print("5. DeepSeek")
    print("6. OpenRouter")
    # Add more as supported by LiteLLM and your app
    
    llm_provider_choice = ""
    valid_choices = ["1", "2", "3", "4", "5", "6"]
    while llm_provider_choice not in valid_choices:
        llm_provider_choice = input(f"Choose your LLM provider (1-{len(valid_choices)}): ").strip()

    if llm_provider_choice == "1":
        config_data["llm_provider"] = "openai"
        config_data["api_key"] = input("Enter your OpenAI API Key: ").strip()
        config_data["model_id"] = input("Enter the OpenAI Model ID (e.g., gpt-4, gpt-3.5-turbo): ").strip()
    elif llm_provider_choice == "2":
        config_data["llm_provider"] = "gemini"
        config_data["api_key"] = input("Enter your Google Gemini API Key: ").strip()
        config_data["model_id"] = input("Enter the Gemini Model ID (e.g., gemini-pro): ").strip()
    elif llm_provider_choice == "3":
        config_data["llm_provider"] = "anthropic"
        config_data["api_key"] = input("Enter your Anthropic API Key: ").strip()
        config_data["model_id"] = input("Enter the Anthropic Model ID (e.g., claude-2): ").strip()
    elif llm_provider_choice == "4":
        config_data["llm_provider"] = "bedrock"
        print("For AWS Bedrock, ensure your AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional)) are configured in your environment or AWS CLI.")
        config_data["aws_access_key_id"] = input("Enter your AWS Access Key ID (or press Enter if configured globally): ").strip()
        config_data["aws_secret_access_key"] = input("Enter your AWS Secret Access Key (or press Enter if configured globally): ").strip()
        config_data["aws_region_name"] = input("Enter the AWS Region Name (e.g., us-east-1): ").strip()
        config_data["model_id"] = input("Enter the Bedrock Model ID (e.g., anthropic.claude-v2): ").strip()
    elif llm_provider_choice == "5":
        config_data["llm_provider"] = "deepseek"
        config_data["api_key"] = input("Enter your DeepSeek API Key: ").strip()
        config_data["model_id"] = input("Enter the DeepSeek Model ID (e.g., deepseek-coder): ").strip()
    elif llm_provider_choice == "6":
        config_data["llm_provider"] = "openrouter"
        config_data["api_key"] = input("Enter your OpenRouter API Key: ").strip()
        # OpenRouter uses the model ID in the format "provider/model"
        config_data["model_id"] = input("Enter the OpenRouter Model ID (e.g., openai/gpt-3.5-turbo or google/gemini-pro): ").strip()
        config_data["openrouter_api_base_url"] = input("Enter OpenRouter API Base URL (optional, press Enter for default): ").strip() or None


    # Ensure necessary keys for LiteLLM are set based on provider
    provider = config_data.get("llm_provider")
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = config_data.get("api_key", "")
    elif provider == "gemini":
        os.environ["GEMINI_API_KEY"] = config_data.get("api_key", "")
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = config_data.get("api_key", "")
    elif provider == "bedrock":
        if config_data.get("aws_access_key_id"):
            os.environ["AWS_ACCESS_KEY_ID"] = config_data["aws_access_key_id"]
        if config_data.get("aws_secret_access_key"):
            os.environ["AWS_SECRET_ACCESS_KEY"] = config_data["aws_secret_access_key"]
        if config_data.get("aws_region_name"):
            os.environ["AWS_REGION_NAME"] = config_data["aws_region_name"]
    elif provider == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = config_data.get("api_key", "")
    elif provider == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = config_data.get("api_key", "")
        if config_data.get("openrouter_api_base_url"):
            os.environ["OPENROUTER_API_BASE"] = config_data["openrouter_api_base_url"] # LiteLLM uses OPENROUTER_API_BASE
            os.environ["OPENROUTER_API_BASE_URL"] = config_data["openrouter_api_base_url"] # Also set this for consistency if used elsewhere

    # 2. Memory Directory Path
    default_memory_path = get_default_data_dir() / "memory"
    raw_memory_path_str = input(f"Enter path for 'memory' directory (press Enter for default: {default_memory_path}): ").strip()
    
    cleaned_memory_path_str = raw_memory_path_str
    if raw_memory_path_str and raw_memory_path_str.startswith('"') and raw_memory_path_str.endswith('"'):
        cleaned_memory_path_str = raw_memory_path_str[1:-1]
    elif raw_memory_path_str and raw_memory_path_str.startswith("'") and raw_memory_path_str.endswith("'"):
        cleaned_memory_path_str = raw_memory_path_str[1:-1]
        
    config_data["memory_base_dir"] = cleaned_memory_path_str if cleaned_memory_path_str else str(default_memory_path)
    Path(config_data["memory_base_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"'memory' directory (for insights, schema, history, vector stores) will be at: {config_data['memory_base_dir']}")

    # 3. Approved Queries (NLQ-SQL JSON pairs) Directory Path
    default_approved_queries_path = get_default_data_dir() / "Approved_NL2SQL_Pairs"
    raw_approved_queries_path_str = input(f"Enter path for 'Approved Queries' directory (stores NLQ-SQL JSON pairs, press Enter for default: {default_approved_queries_path}): ").strip()

    cleaned_approved_queries_path_str = raw_approved_queries_path_str
    if raw_approved_queries_path_str and raw_approved_queries_path_str.startswith('"') and raw_approved_queries_path_str.endswith('"'):
        cleaned_approved_queries_path_str = raw_approved_queries_path_str[1:-1]
    elif raw_approved_queries_path_str and raw_approved_queries_path_str.startswith("'") and raw_approved_queries_path_str.endswith("'"):
        cleaned_approved_queries_path_str = raw_approved_queries_path_str[1:-1]
        
    config_data["approved_queries_dir"] = cleaned_approved_queries_path_str if cleaned_approved_queries_path_str else str(default_approved_queries_path)
    Path(config_data["approved_queries_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"'Approved Queries' (NLQ-SQL JSON pairs) directory will be at: {config_data['approved_queries_dir']}")

    # NL2SQL base directory for LanceDB vector stores will be inside the memory_base_dir
    # No separate user prompt for this; it's derived.
    config_data["nl2sql_vector_store_base_dir"] = str(Path(config_data["memory_base_dir"]) / "lancedb_stores")
    Path(config_data["nl2sql_vector_store_base_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"LanceDB vector stores will be located under: {config_data['nl2sql_vector_store_base_dir']}")
    
    save_config(config_data)
    return config_data

# --- Main Configuration Accessor ---

def get_app_config() -> dict:
    """
    Gets the application configuration.
    If config file doesn't exist or is empty, runs initial setup.
    """
    config = load_config()
    if not config:
        print("No configuration found or configuration is empty. Starting initial setup...")
        config = initial_setup()
    
    # Set environment variables for LiteLLM from loaded config
    # This ensures LiteLLM can pick them up even if not set globally
    # but only if they are present in the config
    provider = config.get("llm_provider")
    if provider:
        if provider == "openai" and config.get("api_key"):
            os.environ["OPENAI_API_KEY"] = config["api_key"]
        elif provider == "gemini" and config.get("api_key"):
            os.environ["GEMINI_API_KEY"] = config["api_key"]
        elif provider == "anthropic" and config.get("api_key"):
            os.environ["ANTHROPIC_API_KEY"] = config["api_key"]
        elif provider == "bedrock":
            if config.get("aws_access_key_id"):
                os.environ["AWS_ACCESS_KEY_ID"] = config["aws_access_key_id"]
            if config.get("aws_secret_access_key"):
                os.environ["AWS_SECRET_ACCESS_KEY"] = config["aws_secret_access_key"]
            if config.get("aws_region_name"):
                os.environ["AWS_REGION_NAME"] = config["aws_region_name"]
        elif provider == "deepseek" and config.get("api_key"):
            os.environ["DEEPSEEK_API_KEY"] = config["api_key"]
        elif provider == "openrouter" and config.get("api_key"):
            os.environ["OPENROUTER_API_KEY"] = config["api_key"]
            if config.get("openrouter_api_base_url"): # Check if it exists and is not None/empty
                os.environ["OPENROUTER_API_BASE"] = config["openrouter_api_base_url"]
                os.environ["OPENROUTER_API_BASE_URL"] = config["openrouter_api_base_url"]


    return config

if __name__ == '__main__':
    # Example usage:
    config = get_app_config()
    print("\nCurrent Application Configuration:")
    print(json.dumps(config, indent=4))

    # Test specific paths
    if config:
        print(f"\nMemory base directory (insights, schema, history, vector stores): {config.get('memory_base_dir')}")
        print(f"Approved Queries (NLQ-SQL JSON pairs) directory: {config.get('approved_queries_dir')}")
        print(f"LanceDB vector store base directory: {config.get('nl2sql_vector_store_base_dir')}")
        print(f"LLM Provider: {config.get('llm_provider')}")
        print(f"LLM Model: {config.get('model_id')}")
