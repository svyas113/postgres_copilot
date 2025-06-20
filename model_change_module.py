import os
from typing import Dict, Optional, Tuple, Any

# Relative import to access config_manager's save and environment setting logic
from . import config_manager

def prompt_for_llm_details() -> Optional[Dict[str, str]]:
    """
    Prompts the user for LLM provider, API key, and model ID.
    Returns a dictionary with the new LLM details or None if aborted.
    """
    new_llm_config: Dict[str, str] = {}

    print("\nChange LLM Configuration")
    print("-------------------------")
    print("Supported LLM Providers:")
    print("1. OpenAI")
    print("2. Google Gemini")
    print("3. Anthropic")
    print("4. AWS Bedrock")
    print("5. DeepSeek")
    print("6. OpenRouter")
    print("0. Cancel")

    llm_provider_choice = ""
    valid_choices = [str(i) for i in range(7)] # 0 to 6

    while llm_provider_choice not in valid_choices:
        llm_provider_choice = input(f"Choose your new LLM provider (0-{len(valid_choices)-1}): ").strip()

    if llm_provider_choice == "0":
        print("LLM change cancelled.")
        return None

    if llm_provider_choice == "1":
        new_llm_config["llm_provider"] = "openai"
        new_llm_config["api_key"] = input("Enter your OpenAI API Key: ").strip()
        new_llm_config["model_id"] = input("Enter the OpenAI Model ID (e.g., gpt-4, gpt-3.5-turbo): ").strip()
    elif llm_provider_choice == "2":
        new_llm_config["llm_provider"] = "gemini"
        new_llm_config["api_key"] = input("Enter your Google Gemini API Key: ").strip()
        new_llm_config["model_id"] = input("Enter the Gemini Model ID (e.g., gemini-pro): ").strip()
    elif llm_provider_choice == "3":
        new_llm_config["llm_provider"] = "anthropic"
        new_llm_config["api_key"] = input("Enter your Anthropic API Key: ").strip()
        new_llm_config["model_id"] = input("Enter the Anthropic Model ID (e.g., claude-2): ").strip()
    elif llm_provider_choice == "4":
        new_llm_config["llm_provider"] = "bedrock"
        print("For AWS Bedrock, ensure your AWS credentials are configured in your environment or AWS CLI if not providing them here.")
        new_llm_config["aws_access_key_id"] = input("Enter your AWS Access Key ID (or press Enter if configured globally): ").strip()
        new_llm_config["aws_secret_access_key"] = input("Enter your AWS Secret Access Key (or press Enter if configured globally): ").strip()
        new_llm_config["aws_region_name"] = input("Enter the AWS Region Name (e.g., us-east-1): ").strip()
        new_llm_config["model_id"] = input("Enter the Bedrock Model ID (e.g., anthropic.claude-v2): ").strip()
    elif llm_provider_choice == "5":
        new_llm_config["llm_provider"] = "deepseek"
        new_llm_config["api_key"] = input("Enter your DeepSeek API Key: ").strip()
        new_llm_config["model_id"] = input("Enter the DeepSeek Model ID (e.g., deepseek-coder): ").strip()
    elif llm_provider_choice == "6":
        new_llm_config["llm_provider"] = "openrouter"
        new_llm_config["api_key"] = input("Enter your OpenRouter API Key: ").strip()
        new_llm_config["model_id"] = input("Enter the OpenRouter Model ID (e.g., openai/gpt-3.5-turbo): ").strip()
        new_llm_config["openrouter_api_base_url"] = input("Enter OpenRouter API Base URL (optional, press Enter for default): ").strip() or None
    
    # Basic validation
    if not new_llm_config.get("model_id"):
        print("Model ID is required. LLM change aborted.")
        return None
    if new_llm_config["llm_provider"] not in ["bedrock"] and not new_llm_config.get("api_key") and not (new_llm_config.get("aws_access_key_id") and new_llm_config.get("aws_secret_access_key")):
         # Bedrock might rely on env vars if keys not entered
        print("API Key (or equivalent AWS credentials) is required for the selected provider. LLM change aborted.")
        return None

    return new_llm_config

def update_llm_config_and_env_vars(current_full_config: Dict[str, Any], new_llm_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the full configuration dictionary with new LLM details and sets environment variables.
    Returns the updated full configuration.
    """
    updated_config = current_full_config.copy()

    # Clear old LLM specific keys that might not be relevant for the new provider
    # For example, if switching from bedrock to openai, aws_ keys should be removed.
    # List of all possible LLM-specific keys used across providers:
    possible_llm_keys = ["api_key", "model_id", "llm_provider", 
                           "aws_access_key_id", "aws_secret_access_key", "aws_region_name",
                           "openrouter_api_base_url"]
    for key in possible_llm_keys:
        if key in updated_config:
            del updated_config[key]

    # Add new LLM details
    for key, value in new_llm_details.items():
        if value is not None: # Only add if value is provided (e.g. openrouter_api_base_url can be None)
            updated_config[key] = value
    
    # Set environment variables for LiteLLM based on the new configuration
    provider = updated_config.get("llm_provider")
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = updated_config.get("api_key", "")
    elif provider == "gemini":
        os.environ["GEMINI_API_KEY"] = updated_config.get("api_key", "")
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = updated_config.get("api_key", "")
    elif provider == "bedrock":
        # Clear other provider keys first
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        # Set Bedrock specific ones
        if updated_config.get("aws_access_key_id"):
            os.environ["AWS_ACCESS_KEY_ID"] = updated_config["aws_access_key_id"]
        if updated_config.get("aws_secret_access_key"):
            os.environ["AWS_SECRET_ACCESS_KEY"] = updated_config["aws_secret_access_key"]
        if updated_config.get("aws_region_name"):
            os.environ["AWS_REGION_NAME"] = updated_config["aws_region_name"]
    elif provider == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = updated_config.get("api_key", "")
    elif provider == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = updated_config.get("api_key", "")
        if updated_config.get("openrouter_api_base_url"):
            os.environ["OPENROUTER_API_BASE"] = updated_config["openrouter_api_base_url"]
            os.environ["OPENROUTER_API_BASE_URL"] = updated_config["openrouter_api_base_url"]
        else: # Ensure it's unset if not provided
            os.environ.pop("OPENROUTER_API_BASE", None)
            os.environ.pop("OPENROUTER_API_BASE_URL", None)
            
    return updated_config

async def handle_change_model_interactive(app_config_ref: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Handles the interactive process of changing the LLM model.
    Modifies app_config_ref in place if successful.
    Returns (success_status, message_to_user).
    """
    new_llm_details = prompt_for_llm_details()

    if not new_llm_details:
        return False, "Model change cancelled by user."

    # Update the application's full config dictionary
    # The app_config_ref is the live dictionary from postgres_copilot_chat.py
    
    # First, clear out old LLM provider-specific keys from app_config_ref
    # to avoid carrying over incompatible settings (e.g., aws_region_name for openai)
    llm_provider_specific_keys_to_clear = [
        "api_key", # Generic, but might be tied to old provider
        # "model_id", # This will be overwritten by new_llm_details
        # "llm_provider", # This will be overwritten
        "aws_access_key_id", 
        "aws_secret_access_key", 
        "aws_region_name",
        "openrouter_api_base_url"
    ]
    for key_to_clear in llm_provider_specific_keys_to_clear:
        app_config_ref.pop(key_to_clear, None)

    # Now, update app_config_ref with the new details
    app_config_ref.update(new_llm_details)

    # Set environment variables based on the newly updated app_config_ref
    # This logic is similar to what's in config_manager.get_app_config()
    provider = app_config_ref.get("llm_provider")
    # Clear all potentially conflicting API keys first
    for env_key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_API_BASE", "OPENROUTER_API_BASE_URL"]:
        os.environ.pop(env_key, None)

    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = app_config_ref.get("api_key", "")
    elif provider == "gemini":
        os.environ["GEMINI_API_KEY"] = app_config_ref.get("api_key", "")
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = app_config_ref.get("api_key", "")
    elif provider == "bedrock":
        if app_config_ref.get("aws_access_key_id"):
            os.environ["AWS_ACCESS_KEY_ID"] = app_config_ref["aws_access_key_id"]
        if app_config_ref.get("aws_secret_access_key"):
            os.environ["AWS_SECRET_ACCESS_KEY"] = app_config_ref["aws_secret_access_key"]
        if app_config_ref.get("aws_region_name"):
            os.environ["AWS_REGION_NAME"] = app_config_ref["aws_region_name"]
    elif provider == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = app_config_ref.get("api_key", "")
    elif provider == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = app_config_ref.get("api_key", "")
        if app_config_ref.get("openrouter_api_base_url"):
            os.environ["OPENROUTER_API_BASE"] = app_config_ref["openrouter_api_base_url"]
            os.environ["OPENROUTER_API_BASE_URL"] = app_config_ref["openrouter_api_base_url"]
    
    # Save the modified app_config_ref to disk
    try:
        config_manager.save_config(app_config_ref)
        return True, f"LLM configuration updated to use {app_config_ref.get('llm_provider')} with model {app_config_ref.get('model_id')}."
    except Exception as e:
        return False, f"Error saving updated LLM configuration: {e}"

if __name__ == "__main__":
    import asyncio # Add import for asyncio.run
    # This is a basic test for the module, assuming config_manager works.
    # In a real scenario, this would be called from postgres_copilot_chat.py
    print("Testing model_change_module.py interactively...")
    
    # Simulate loading an existing config
    mock_current_config = config_manager.get_app_config() # Gets or runs initial_setup
    print("\nCurrent (or initial) LLM Config:")
    print(f"  Provider: {mock_current_config.get('llm_provider')}")
    print(f"  Model ID: {mock_current_config.get('model_id')}")

    # Run the interactive change process
    # The handle_change_model_interactive function now modifies the passed dict in-place
    # and also saves it.
    success, message = asyncio.run(handle_change_model_interactive(mock_current_config))
    
    print(f"\n{message}")
    if success:
        print("\nUpdated LLM Config (from potentially modified mock_current_config):")
        print(f"  Provider: {mock_current_config.get('llm_provider')}")
        print(f"  Model ID: {mock_current_config.get('model_id')}")
        print(f"  API Key (example): {mock_current_config.get('api_key', 'N/A (Bedrock might use env vars)')}")
        # Check an env var
        if mock_current_config.get("llm_provider") == "openai":
            print(f"  OPENAI_API_KEY env var: {os.getenv('OPENAI_API_KEY')}")

        print("\nConfiguration file should also be updated.")
        print(f"Config file path: {config_manager.get_config_file_path()}")
