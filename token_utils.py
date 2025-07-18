import litellm
import tiktoken
from typing import Optional
import json

def count_tokens(text: str, model_name: str, provider: str) -> int:
    """
    Counts the number of tokens in a string.
    Uses tiktoken for OpenAI, Anthropic, and other models that use similar tokenization.
    Uses a fallback for Gemini since it has a specific tokenizer.
    """
    if provider == "gemini":
        # Placeholder for Gemini-specific token counting if needed.
        # For now, we can use litellm's token counter which is more robust.
        return litellm.token_counter(model=model_name, text=text)
    try:
        # For most models, tiktoken is a reliable and fast way to count tokens.
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        # If the model is not found in tiktoken, fall back to litellm's counter.
        return litellm.token_counter(model=model_name, text=text)

def calculate_token_size(model_id: str, json_file_path: str, provider: str) -> int:
    """
    Calculates the token size of a JSON file for a given model.
    """
    try:
        with open(json_file_path, 'r') as f:
            json_content = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Error: The file '{json_file_path}' is not a valid JSON file.")

    json_string = json.dumps(json_content)
    return count_tokens(json_string, model_id, provider)

def calculate_available_tokens(prompt_template: str, model_name: str, provider: str, reserved_for_response: int = 1000) -> int:
    """
    Calculates the available token budget for schema data.
    """
    context_window = get_context_window_size(model_name)
    prompt_boilerplate_tokens = count_tokens(prompt_template, model_name, provider)
    available_tokens = context_window - prompt_boilerplate_tokens - reserved_for_response
    return available_tokens

def get_context_window_size(model_name: str) -> int:
    """
    Retrieves the context window size for a given model using litellm.
    """
    try:
        model_info = litellm.get_model_info(model_name)
        return model_info.get('max_input_tokens', 8192)
    except Exception:
        return 8192
