import litellm
from token_logging_module import log_token_usage
import inspect
import token_utils

def handle_litellm_exception(e: Exception) -> str:
    """
    Translates a LiteLLM exception into a user-friendly error message.
    """
    if isinstance(e, litellm.AuthenticationError):
        return "Authentication Error: The provided API key is invalid or has expired. Please check your credentials."
    elif isinstance(e, litellm.RateLimitError):
        return "Rate Limit Error: You have exceeded the number of requests allowed by the API. Please wait and try again later."
    elif isinstance(e, litellm.NotFoundError):
        return f"Model Not Found Error: The model '{e.model_name}' is not available or does not exist. Please check the model ID."
    elif isinstance(e, litellm.APIConnectionError):
        return f"API Connection Error: Could not connect to the LLM service. Please check your network connection and the service status."
    else:
        # For any other LiteLLM exception or a generic exception
        return f"An unexpected error occurred while communicating with the LLM: {str(e)}"

async def test_llm_connection(profile: dict) -> tuple[bool, str]:
    """
    Tests the connection to an LLM using a given profile.
    Returns a tuple of (success: bool, message: str).
    """
    try:
        # Use a simple, low-token prompt for testing
        messages = [{"role": "user", "content": "hello"}]
        
        # Create a temporary dictionary for the call to avoid modifying the original profile
        call_params = {
            "model": profile["model_id"],
            "messages": messages,
            "max_tokens": 10, # Keep it short
        }
        
        # Add credentials directly for the test call
        if profile.get("credentials"):
            for key, value in profile["credentials"].items():
                call_params[key] = value

        response = await litellm.acompletion(**call_params)
        
        # Log token usage
        frame = inspect.currentframe()
        origin_script = inspect.getframeinfo(frame).filename
        origin_line = inspect.getframeinfo(frame).lineno
        prompt_text = messages[-1]['content']
        prompt_tokens = token_utils.count_tokens(prompt_text, profile["model_id"], profile["credentials"]["api_key"])
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        llm_response = response.choices[0].message.content

        log_token_usage(
            origin_script=origin_script,
            origin_line=origin_line,
            user_query="test_llm_connection",
            prompt=prompt_text,
            prompt_tokens=prompt_tokens,
            schema_tokens=0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_response=llm_response,
            model_id=profile["model_id"]
        )

        return True, "Connection successful!"
    except Exception as e:
        return False, handle_litellm_exception(e)
