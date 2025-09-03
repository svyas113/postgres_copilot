import asyncio
import sys
import os
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules we need to test
try:
    import hyde_module
    import vector_store_module
    from llm_client import create_llm_client
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

async def test_hyde_embedding_fix():
    """
    Test the fix for the embedding error in hyde_module.py
    """
    print("\n=== Testing HyDE Embedding Fix ===")
    
    # Create an LLM client
    llm_client = create_llm_client()
    
    # Test natural language question
    test_nlq = "List 25 employees"
    
    # Test database identifier
    db_name_identifier = "demo_db"
    
    try:
        # Call retrieve_hyde_context
        print(f"Calling retrieve_hyde_context with NLQ: '{test_nlq}'")
        hyde_context, table_names = await hyde_module.retrieve_hyde_context(test_nlq, db_name_identifier, llm_client)
        
        # Print results
        print("\nResults:")
        print(f"Retrieved {len(table_names)} table names: {table_names}")
        print(f"Context length: {len(hyde_context)} characters")
        print("\nFirst 200 characters of context:")
        print(hyde_context[:200] + "..." if len(hyde_context) > 200 else hyde_context)
        
        print("\nHyDE embedding fix test completed successfully!")
        return True
    except Exception as e:
        print(f"\nError during HyDE test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_retrieval_fix():
    """
    Test the fix for the RAG retrieval error in sql_generation.py
    """
    print("\n=== Testing RAG Retrieval Fix ===")
    
    # Test database identifier
    db_name_identifier = "demo_db"
    
    # Test natural language question
    test_nlq = "List 25 employees"
    
    try:
        # Call search_similar_nlqs
        print(f"Calling search_similar_nlqs with NLQ: '{test_nlq}'")
        
        # Using the hardcoded threshold from vector_store_module
        current_rag_threshold = getattr(vector_store_module, 'LITELLM_RAG_THRESHOLD', 0.75)
        
        similar_pairs = vector_store_module.search_similar_nlqs(
            db_name_identifier=db_name_identifier,
            query_nlq=test_nlq,
            k=3,
            threshold=current_rag_threshold
        )
        
        # Ensure similar_pairs is a list
        if not isinstance(similar_pairs, list):
            print(f"Warning: search_similar_nlqs returned a non-list: {type(similar_pairs).__name__}")
            similar_pairs = []
        
        # Print results
        print("\nResults:")
        print(f"Found {len(similar_pairs)} similar pairs")
        for i, pair in enumerate(similar_pairs):
            print(f"\nPair {i+1}:")
            print(f"  NLQ: {pair.get('nlq', 'N/A')}")
            print(f"  SQL: {pair.get('sql', 'N/A')}")
            print(f"  Similarity: {pair.get('similarity_score', 'N/A')}")
        
        print("\nRAG retrieval fix test completed successfully!")
        return True
    except Exception as e:
        print(f"\nError during RAG retrieval test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """
    Run all tests
    """
    print(f"Starting tests at {datetime.now().isoformat()}")
    
    # Test the HyDE embedding fix
    hyde_success = await test_hyde_embedding_fix()
    
    # Test the RAG retrieval fix
    rag_success = await test_rag_retrieval_fix()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"HyDE Embedding Fix: {'PASSED' if hyde_success else 'FAILED'}")
    print(f"RAG Retrieval Fix: {'PASSED' if rag_success else 'FAILED'}")
    
    # Exit with appropriate status code
    if hyde_success and rag_success:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
