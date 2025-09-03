import os
import json
import litellm
import asyncio
from typing import Dict, Any, List, Optional, Union
import re

class LLMClient:
    """
    A simple LLM client that uses environment variables for configuration.
    """
    
    def __init__(self, session=None):
        """
        Initialize the LLM client using environment variables.
        """
        # Get the model name from environment variables or file
        self.model_name = os.getenv("POSTGRES_LLM")
        
        # If not found, try to read from the env file directly
        if not self.model_name:
            self.model_name = self._read_model_name_from_file()
            
        # If still not found, use a default
        if not self.model_name:
            self.model_name = "mistral/mistral-small-latest"
        
        # Get the API key from environment variables
        # First try the specific API key for POSTGRES_LLM
        self.api_key = os.getenv("LITELLM_API_KEY_POSTGRES_LLM")
        
        # If not found, fall back to the general API key
        if not self.api_key:
            self.api_key = os.getenv("LITELLM_API_KEY_RESPONSE_GEN")
            
        # If still not found, try to read from the passwords file directly
        if not self.api_key:
            self.api_key = self._read_api_key_from_file()
        
        # Store the session object if provided
        self.session = session
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Print debug information
        print(f"Initializing LLM client with model: {self.model_name}")
        print(f"API key available: {bool(self.api_key)}")
        print(f"Database session available: {bool(self.session)}")
        
    def _read_model_name_from_file(self) -> Optional[str]:
        """
        Read the model name from the fastworkflow.env file.
        """
        try:
            # Try to find the env file in the current directory or parent directories
            env_file_paths = [
                "fastworkflow.env",
                "../fastworkflow.env",
                "../../fastworkflow.env",
                "../../../fastworkflow.env",
                "/home/shivam/fastworkflow/postgres_copilot (new)/fastworkflow.env",
                "/home/shivam/fastworkflow/postgres_copilot2/fastworkflow.env"
            ]
            
            for path in env_file_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        content = f.read()
                        # Look for POSTGRES_LLM
                        model_match = re.search(r'POSTGRES_LLM=([^\s]+)', content)
                        if model_match:
                            return model_match.group(1)
            
            return None
        except Exception as e:
            print(f"Error reading model name from file: {e}")
            return None
    
    def _read_api_key_from_file(self) -> Optional[str]:
        """
        Read the API key from the fastworkflow.passwords.env file.
        """
        try:
            # Try to find the passwords file in the current directory or parent directories
            passwords_file_paths = [
                "fastworkflow.passwords.env",
                "../fastworkflow.passwords.env",
                "../../fastworkflow.passwords.env",
                "../../../fastworkflow.passwords.env",
                "/home/shivam/fastworkflow/postgres_copilot (new)/fastworkflow.passwords.env",
                "/home/shivam/fastworkflow/postgres_copilot2/fastworkflow.passwords.env"
            ]
            
            for path in passwords_file_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        content = f.read()
                        # Look for LITELLM_API_KEY_POSTGRES_LLM or LITELLM_API_KEY_RESPONSE_GEN
                        postgres_match = re.search(r'LITELLM_API_KEY_POSTGRES_LLM=([^\s]+)', content)
                        if postgres_match:
                            return postgres_match.group(1)
                        
                        response_match = re.search(r'LITELLM_API_KEY_RESPONSE_GEN=([^\s]+)', content)
                        if response_match:
                            return response_match.group(1)
            
            return None
        except Exception as e:
            print(f"Error reading API key from file: {e}")
            return None
    
    def _extract_mcp_tool_call_output(self, result: Any, tool_name: str) -> Any:
        """
        Extract the output from an MCP tool call result.
        
        Args:
            result: The result from the MCP tool call
            tool_name: The name of the tool that was called
            
        Returns:
            The extracted output
        """
        # Handle different result formats
        if isinstance(result, dict):
            # If result is already a dict, return it directly
            return result
        elif isinstance(result, str):
            # If result is a string, try to parse it as JSON
            try:
                return json.loads(result)
            except:
                # If parsing fails, return the string as is
                return result
        else:
            # For any other type, convert to string
            return str(result)
    
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Send a chat message to the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to litellm.acompletion
            
        Returns:
            The response from the LLM
        """
        try:
            # Prepare the parameters for the LLM call
            params = {
                "model": self.model_name,
                "messages": messages,
                "api_key": self.api_key,
            }
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Make the LLM call
            response = await litellm.acompletion(**params)
            
            # Update conversation history with the user message and assistant response
            if messages and len(messages) > 0:
                last_message = messages[-1]
                if last_message not in self.conversation_history:
                    self.conversation_history.append(last_message)
                
                if response and response.choices and len(response.choices) > 0:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    }
                    self.conversation_history.append(assistant_message)
            
            return response
        except Exception as e:
            print(f"Error in LLM chat: {e}")
            raise

def create_llm_client(session=None) -> LLMClient:
    """
    Create and return an LLM client instance.
    
    Args:
        session: Optional database session to attach to the client
        
    Returns:
        An initialized LLM client
    """
    return LLMClient(session=session)

async def _send_message_to_llm(llm_client: LLMClient, messages: list, user_query: str, schema_tokens: int = 0, 
                              tools: Optional[list] = None, tool_choice: str = "auto",
                              response_format: Optional[dict] = None, source: str = None) -> Any:
    """
    Helper function to send messages to LLM and handle response.
    This mimics the functionality from postgres_copilot_chat.py for compatibility.
    
    Args:
        llm_client: The LLM client instance
        messages: List of message dictionaries with 'role' and 'content' keys
        user_query: The user's query for logging
        schema_tokens: Number of tokens in schema context
        tools: Optional list of tools for the LLM
        tool_choice: Tool choice strategy
        response_format: Optional response format specification
        source: Source of the request for logging
        
    Returns:
        The response from the LLM
    """
    # Prepare kwargs for chat
    kwargs = {}
    
    # Add tools if provided
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    
    # Add response_format if provided
    if response_format:
        kwargs["response_format"] = response_format
    
    # Make the LLM call
    response = await llm_client.chat(messages, **kwargs)
    
    # Add user message to conversation history if not already there
    if messages and len(messages) > 0 and messages[-1]["role"] == "user":
        if messages[-1] not in llm_client.conversation_history:
            llm_client.conversation_history.append(messages[-1])
    
    return response

async def _process_llm_response(llm_client: LLMClient, llm_response: Any) -> tuple[str, bool]:
    """
    Helper function to process LLM response.
    This mimics the functionality from postgres_copilot_chat.py for compatibility.
    
    Args:
        llm_client: The LLM client instance
        llm_response: The response from the LLM
        
    Returns:
        Tuple of (response_text, tool_calls_made)
    """
    if not llm_response or not llm_response.choices or not llm_response.choices[0].message:
        return "Error: Empty or invalid response from LLM.", False
    
    message = llm_response.choices[0].message
    response_text = message.content or ""
    
    # Add assistant response to conversation history if not already there
    assistant_message = {"role": "assistant", "content": response_text}
    if assistant_message not in llm_client.conversation_history:
        llm_client.conversation_history.append(assistant_message)
    
    return response_text, False  # No tool calls in this simplified version
