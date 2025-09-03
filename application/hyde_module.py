from typing import List, Tuple, Dict, Any, Optional
import re
import json
import os
import sys
from datetime import datetime

# Try to import error_handler_module with proper fallback
try:
    from .error_handler_module import handle_exception
except ImportError:
    try:
        from error_handler_module import handle_exception
    except ImportError:
        # Define a simple fallback error handler if the module can't be imported
        def handle_exception(e, user_query=None, context=None):
            error_msg = f"Error: {e}"
            if context:
                context_str = ", ".join(f"{k}={v}" for k, v in context.items())
                error_msg += f" [Context: {context_str}]"
            if user_query:
                error_msg += f" [Query: {user_query}]"
            print(error_msg, file=sys.stderr)
            return str(e)

# Try to import other modules with proper fallback
try:
    # First try relative import
    from . import vector_store_module
    from . import schema_vectorization_module
    from . import memory_module
except ImportError:
    try:
        # Then try absolute import
        import vector_store_module
        import schema_vectorization_module
        import memory_module
    except ImportError:
        try:
            # Try importing from parent directory
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application import vector_store_module
            from application import schema_vectorization_module
            from application import memory_module
        except ImportError as e:
            print(f"Error importing modules: {e}", file=sys.stderr)
            # Define fallback modules if imports fail
            vector_store_module = None
            schema_vectorization_module = None
            memory_module = None

# Define fallback functions for schema_vectorization_module if it's not available
if schema_vectorization_module is None:
    class SchemaVectorizationModuleFallback:
        @staticmethod
        def search_schema_vectors(db_name_identifier, query_embedding, limit=10):
            print(f"Warning: Using fallback schema_vectorization_module.search_schema_vectors", file=sys.stderr)
            return []  # Return empty list as fallback
    
    schema_vectorization_module = SchemaVectorizationModuleFallback()

# Define fallback functions for vector_store_module if it's not available
if vector_store_module is None:
    class VectorStoreModuleFallback:
        @staticmethod
        def _initialize_embedding_model():
            print(f"Warning: Using fallback vector_store_module._initialize_embedding_model", file=sys.stderr)
            # Return a minimal embedding model that just returns the input
            class DummyEmbeddingModel:
                def encode(self, texts, show_progress_bar=False):
                    print(f"Warning: Using dummy embedding model", file=sys.stderr)
                    # Return a dummy embedding for each text
                    return [[0.0] * 384 for _ in texts]
                
                def get_sentence_embedding_dimension(self):
                    return 384
            
            return DummyEmbeddingModel()
        
        @staticmethod
        def search_similar_nlqs(db_name_identifier, query_nlq, k=5, threshold=None):
            print(f"Warning: Using fallback vector_store_module.search_similar_nlqs", file=sys.stderr)
            return []  # Return empty list as fallback
    
    vector_store_module = VectorStoreModuleFallback()

# Define fallback functions for memory_module if it's not available
if memory_module is None:
    class MemoryModuleFallback:
        @staticmethod
        def get_memory_base_path():
            print(f"Warning: Using fallback memory_module.get_memory_base_path", file=sys.stderr)
            return "/home/shivam/fastworkflow/data/memory"
    
    memory_module = MemoryModuleFallback()

# Try to import token_utils
try:
    from .token_utils import count_tokens
except ImportError:
    try:
        from token_utils import count_tokens
    except ImportError:
        # Define a simple fallback token counter
        def count_tokens(text):
            # Simple approximation: ~4 chars per token
            return len(text) // 4

# Import the actual LLMClient instead of the expected LiteLLMMcpClient
# Try relative import first (for when running as part of a package)
# Fall back to absolute import if that fails (for when running directly)
try:
    from .llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient

# Initialize logs directory
def initialize_logs():
    """
    Initialize the logs directory for Hyde module logging.
    This should be called during application startup.
    """
    try:
        memory_base_path = memory_module.get_memory_base_path()
        logs_dir = os.path.join(memory_base_path, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        print(f"Hyde logs directory initialized at: {logs_dir}")
        return True
    except Exception as e:
        print(f"Error initializing logs directory: {e}", file=sys.stderr)
        return False

async def _generate_hypothetical_document(nlq: str, llm_client: LLMClient) -> str:
    """
    Generates a hypothetical document (a descriptive answer) based on the user's NLQ.
    """
    prompt = (
        "You are an expert data analyst. A user wants to query a database with the following question. "
        "Based on this question, generate a detailed, hypothetical description of the data that would be needed to answer it. "
        "Describe the ideal table and columns. Do not write SQL. Focus on describing the structure and content of the expected data.\n\n"
        f"User Question: \"{nlq}\"\n\n"
        "Hypothetical Data Description:"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Use the simpler chat interface of LLMClient
        response = await llm_client.chat(messages)
        
        # Extract the response text from the LLM response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_text = response.choices[0].message.content
        else:
            # Fallback in case the response structure is different
            response_text = str(response)
        
        return response_text
    except Exception as e:
        handle_exception(e, user_query=nlq, context={"step": "generate_hypothetical_document"})
        # Fallback to using the original NLQ if generation fails
        return nlq

def _log_hyde_retrieval_debug(db_name_identifier: str, event: str, data: Dict[str, Any]) -> None:
    """
    Logs HyDE debug information to a JSON file.
    
    Args:
        db_name_identifier: The database identifier
        event: The event name (e.g., 'hyde_started', 'hyde_completed', 'hyde_error')
        data: Dictionary containing debug data
    """
    try:
        # Get the logs directory path
        logs_dir = os.path.join('/home/shivam/fastworkflow/data/memory/logs')
        
        # Ensure logs directory exists
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare log file path
        log_file_path = os.path.join(logs_dir, 'sql_generation_debug.json')
        
        # Create log entry with timestamp
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "db_name_identifier": db_name_identifier,
            "module": "hyde",
            "event": event,
            **data  # Include all data fields
        }
        
        # Append to log file
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    except Exception as e:
        print(f"Error logging HyDE debug: {e}", file=sys.stderr)

def _log_hyde_retrieval(nlq: str, db_name_identifier: str, hypothetical_doc: str, 
                       relevant_schema_parts: List[str], unique_parts: List[str], 
                       table_names: List[str], similarity_scores: Optional[List[float]] = None) -> None:
    """
    Logs HyDE retrieval information to a JSON file.
    
    Args:
        nlq: The natural language question
        db_name_identifier: The database identifier
        hypothetical_doc: The generated hypothetical document
        relevant_schema_parts: The retrieved schema parts before deduplication
        unique_parts: The unique schema parts after deduplication
        table_names: The extracted table names
        similarity_scores: Optional similarity scores for the retrieved parts
    """
    try:
        # Get the logs directory path - use hardcoded path to avoid dependency issues
        logs_dir = os.path.join('/home/shivam/fastworkflow/data/memory/logs')
        
        # Ensure logs directory exists
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare log file path
        log_file_path = os.path.join(logs_dir, 'hyde_log.json')
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "db_name_identifier": db_name_identifier,
            "natural_language_question": nlq,
            "hypothetical_document": hypothetical_doc,
            "retrieved_chunks": {
                "total_count": len(relevant_schema_parts),
                "chunks": relevant_schema_parts,
                "unique_count": len(unique_parts)
            },
            "extracted_table_names": table_names
        }
        
        # Add similarity scores if available
        if similarity_scores:
            log_entry["retrieved_chunks"]["similarity_scores"] = similarity_scores
        
        # Append to log file
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    except Exception as e:
        handle_exception(e, user_query=nlq, context={"step": "log_hyde_retrieval"})

async def retrieve_hyde_context(nlq: str, db_name_identifier: str, llm_client: LLMClient) -> Tuple[str, List[str]]:
    """
    Orchestrates the HyDE process and extracts table names from the context.
    1. Generates a hypothetical document.
    2. Creates an embedding for it.
    3. Searches for relevant schema parts.
    4. Extracts table names from the retrieved context.
    5. Returns the concatenated context and the list of table names.
    
    This function also tracks token usage for the HyDE process.
    """
    # Initialize logs directory if needed
    initialize_logs()
    
    # Log the start of HyDE retrieval
    _log_hyde_retrieval_debug(db_name_identifier, "hyde_retrieval_started", {
        "nlq": nlq
    })
    
    try:
        # 1. Generate hypothetical document
        hypothetical_doc = await _generate_hypothetical_document(nlq, llm_client)
        
        # Log the hypothetical document generation
        _log_hyde_retrieval_debug(db_name_identifier, "hypothetical_document_generated", {
            "nlq": nlq,
            "hypothetical_doc": hypothetical_doc,
            "doc_length": len(hypothetical_doc)
        })

        # 2. Create embedding
        try:
            model = vector_store_module._initialize_embedding_model()
            encoded_result = model.encode([hypothetical_doc])[0]
            
            # Check the type of the encoded result and handle accordingly
            if hasattr(encoded_result, 'tolist'):
                # It's a numpy array, convert to list
                embedding = encoded_result.tolist()
            else:
                # It's already a list or similar structure
                embedding = list(encoded_result)
            
            # Log embedding creation with type information for debugging
            _log_hyde_retrieval_debug(db_name_identifier, "embedding_created", {
                "nlq": nlq,
                "embedding_dimension": len(embedding),
                "embedding_type": type(encoded_result).__name__  # Log the type for debugging
            })
        except Exception as e_embed:
            handle_exception(e_embed, user_query=nlq, context={"step": "create_embedding"})
            _log_hyde_retrieval_debug(db_name_identifier, "embedding_error", {
                "error": str(e_embed),
                "error_type": type(e_embed).__name__
            })
            return "Error creating embedding for HyDE. Using fallback.", []

        # 3. Search for relevant schema parts
        try:
            # Check if schema_vectorization_module is available and has search_schema_vectors method
            if schema_vectorization_module and hasattr(schema_vectorization_module, 'search_schema_vectors'):
                relevant_schema_parts = schema_vectorization_module.search_schema_vectors(
                    db_name_identifier=db_name_identifier,
                    query_embedding=embedding,
                    limit=45 # Fetch a decent number of relevant parts
                )
                
                # Ensure relevant_schema_parts is a list
                if not isinstance(relevant_schema_parts, list):
                    relevant_schema_parts = []
                
                # Log schema search results
                _log_hyde_retrieval_debug(db_name_identifier, "schema_search_completed", {
                    "nlq": nlq,
                    "parts_found": len(relevant_schema_parts)
                })
            else:
                # Log that schema_vectorization_module is not available
                _log_hyde_retrieval_debug(db_name_identifier, "schema_search_error", {
                    "error": "schema_vectorization_module not available or missing search_schema_vectors method",
                    "error_type": "ModuleNotFoundError"
                })
                relevant_schema_parts = []
        except Exception as e_search:
            handle_exception(e_search, user_query=nlq, context={"step": "search_schema_vectors"})
            _log_hyde_retrieval_debug(db_name_identifier, "schema_search_error", {
                "error": str(e_search),
                "error_type": type(e_search).__name__
            })
            return "Error searching schema vectors for HyDE. Using fallback.", []

        if not relevant_schema_parts:
            _log_hyde_retrieval_debug(db_name_identifier, "hyde_retrieval_failed", {
                "nlq": nlq,
                "reason": "No relevant schema parts found"
            })
            return "No relevant schema information found via HyDE search.", []
        
        # Extract similarity scores if available
        similarity_scores = None
        if hasattr(relevant_schema_parts, 'scores'):
            similarity_scores = relevant_schema_parts.scores

        # 4. Concatenate context and extract table names
        unique_parts = sorted(list(set(relevant_schema_parts)))
        hyde_context = "\n".join(unique_parts)
        
        # 5. Extract table names from the context
        table_names = []
        for part in unique_parts:
            # Regex to find "Table: table_name"
            match = re.search(r"Table: (\w+)", part)
            if match:
                table_name = match.group(1)
                if table_name not in table_names:
                    table_names.append(table_name)
        
        # Log the HyDE retrieval information
        _log_hyde_retrieval(
            nlq=nlq,
            db_name_identifier=db_name_identifier,
            hypothetical_doc=hypothetical_doc,
            relevant_schema_parts=relevant_schema_parts,
            unique_parts=unique_parts,
            table_names=table_names,
            similarity_scores=similarity_scores
        )
        
        # Log successful HyDE retrieval
        _log_hyde_retrieval_debug(db_name_identifier, "hyde_retrieval_completed", {
            "nlq": nlq,
            "context_length": len(hyde_context),
            "table_names_found": table_names,
            "unique_parts_count": len(unique_parts)
        })
            
        return hyde_context, table_names

    except Exception as e:
        error_message = str(e)
        handle_exception(e, user_query=nlq, context={"step": "retrieve_hyde_context"})
        
        # Log HyDE retrieval error
        _log_hyde_retrieval_debug(db_name_identifier, "hyde_retrieval_error", {
            "nlq": nlq,
            "error": error_message,
            "error_type": type(e).__name__
        })
        
        return f"Error retrieving HyDE context: {error_message}. Using fallback.", []
