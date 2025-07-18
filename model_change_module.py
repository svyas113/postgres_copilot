import os
from typing import Dict, Optional, Tuple, Any
from error_handler_module import handle_exception
import config_manager
import litellm_error_handler

def prompt_for_new_profile() -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Prompts the user for all details required for a new LLM profile.
    Returns a tuple of (profile_alias, profile_details) or None if aborted.
    """
    print("\nCreate a New LLM Profile")
    print("-------------------------")
    
    profile_alias = input("Enter a memorable name (alias) for this new profile (e.g., 'my_openai_4'): ").strip()
    if not profile_alias:
        print("Profile alias cannot be empty. Aborting.")
        return None

    profile_details: Dict[str, Any] = {"credentials": {}}

    print("\nSupported LLM Providers:")
    print("1. OpenAI")
    print("2. Google Gemini")
    print("3. Anthropic")
    print("4. AWS Bedrock")
    print("5. DeepSeek")
    print("6. OpenRouter")
    print("0. Cancel")

    llm_provider_choice = ""
    valid_choices = [str(i) for i in range(7)]

    while llm_provider_choice not in valid_choices:
        llm_provider_choice = input(f"Choose your LLM provider (0-{len(valid_choices)-1}): ").strip()

    if llm_provider_choice == "0":
        print("Profile creation cancelled.")
        return None

    provider_map = {
        "1": "openai", "2": "gemini", "3": "anthropic", 
        "4": "bedrock", "5": "deepseek", "6": "openrouter"
    }
    provider = provider_map[llm_provider_choice]
    profile_details["provider"] = provider

    if provider == "bedrock":
        print("For AWS Bedrock, ensure your AWS credentials are configured in your environment or AWS CLI if not providing them here.")
        profile_details["credentials"]["aws_access_key_id"] = input("Enter your AWS Access Key ID (or press Enter if configured globally): ").strip()
        profile_details["credentials"]["aws_secret_access_key"] = input("Enter your AWS Secret Access Key (or press Enter if configured globally): ").strip()
        profile_details["credentials"]["aws_region_name"] = input("Enter the AWS Region Name (e.g., us-east-1): ").strip()
        profile_details["model_id"] = input("Enter the Bedrock Model ID (e.g., anthropic.claude-v2): ").strip()
    elif provider == "openrouter":
        profile_details["credentials"]["api_key"] = input(f"Enter your {provider.title()} API Key: ").strip()
        profile_details["model_id"] = input(f"Enter the {provider.title()} Model ID (e.g., openai/gpt-3.5-turbo): ").strip()
        profile_details["credentials"]["openrouter_api_base_url"] = input("Enter OpenRouter API Base URL (optional, press Enter for default): ").strip() or None
    else: # For openai, gemini, anthropic, deepseek
        profile_details["credentials"]["api_key"] = input(f"Enter your {provider.title()} API Key: ").strip()
        profile_details["model_id"] = input(f"Enter the {provider.title()} Model ID: ").strip()

    # Validation
    if not profile_details.get("model_id"):
        print("Model ID is required. Profile creation aborted.")
        return None
    if provider != "bedrock" and not profile_details.get("credentials", {}).get("api_key"):
        print("API Key is required for the selected provider. Profile creation aborted.")
        return None

    return profile_alias, profile_details

def switch_active_profile(config: Dict[str, Any]) -> Optional[str]:
    """
    Lists available profiles and prompts the user to switch the active one.
    Returns the alias of the newly selected profile or None if aborted.
    """
    profiles = config.get("llm_profiles", {})
    if not profiles:
        print("No profiles found in configuration.")
        return None

    print("\nSelect a profile to make active:")
    profile_aliases = list(profiles.keys())
    for i, alias in enumerate(profile_aliases, 1):
        print(f"{i}. {alias} (Provider: {profiles[alias].get('provider')}, Model: {profiles[alias].get('model_id')})")
    print("0. Cancel")

    choice = -1
    while choice < 0 or choice > len(profile_aliases):
        try:
            choice_str = input(f"Enter your choice (0-{len(profile_aliases)}): ").strip()
            choice = int(choice_str)
        except ValueError:
            print("Invalid input. Please enter a number.")

    if choice == 0:
        print("Switch profile cancelled.")
        return None
    
    return profile_aliases[choice - 1]

async def handle_change_model_interactive(app_config_ref: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Handles the interactive process of managing LLM profiles.
    This can involve adding a new profile or switching the active one.
    The app_config_ref is NOT modified in place. Instead, the config file is updated directly.
    Returns (success_status, message_to_user).
    """
    full_config = config_manager.load_config()
    if not full_config:
        return False, "Could not load configuration file. Please check for errors."

    print("\nLLM Profile Management")
    print("----------------------")
    print("1. Add a new LLM profile")
    print("2. Switch active LLM profile")
    print("0. Cancel")

    choice = ""
    while choice not in ["1", "2", "0"]:
        choice = input("What would you like to do? (0-2): ").strip()

    if choice == "0":
        return False, "Operation cancelled."

    elif choice == "1": # Add a new profile
        new_profile_data = prompt_for_new_profile()
        if not new_profile_data:
            return False, "Profile creation cancelled."
        
        alias, details = new_profile_data

        # Test connection before saving
        print(f"Testing connection for new profile '{alias}'...")
        test_passed, message = await litellm_error_handler.test_llm_connection(details)
        if not test_passed:
            print(f"Connection Test Failed: {message}")
            return False, "Could not verify the new profile. Please check the credentials and model ID."
        print("Connection test successful!")

        full_config["llm_profiles"][alias] = details
        full_config["active_llm_profile_alias"] = alias
        
        try:
            config_manager.save_config(full_config)
            # The app_config_ref in the main loop will be stale.
            # The main loop should reload it after this function returns True.
            return True, f"Successfully added and activated new profile '{alias}'."
        except Exception as e:
            return False, handle_exception(e, "save_new_profile")

    elif choice == "2": # Switch active profile
        new_active_alias = switch_active_profile(full_config)
        if not new_active_alias:
            return False, "Switch profile cancelled."
            
        full_config["active_llm_profile_alias"] = new_active_alias
        
        try:
            config_manager.save_config(full_config)
            # The app_config_ref in the main loop will be stale.
            # The main loop should reload it after this function returns True.
            return True, f"Successfully switched active profile to '{new_active_alias}'."
        except Exception as e:
            return False, handle_exception(e, "switch_active_profile")
            
    return False, "Invalid selection."
