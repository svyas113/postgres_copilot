import asyncio
from hyde_module import retrieve_hyde_context
from llm_client import create_llm_client

async def test_hyde_module():
    """
    Test the hyde_module to ensure it works with the LLMClient.
    """
    print("Testing hyde_module with LLMClient...")
    
    # Create an LLM client
    llm_client = create_llm_client()
    
    # Test natural language question
    test_nlq = "List 25 employees"
    
    # Test database identifier
    db_name_identifier = "demo_db"
    
    try:
        # Call retrieve_hyde_context
        print(f"Calling retrieve_hyde_context with NLQ: '{test_nlq}'")
        hyde_context, table_names = await retrieve_hyde_context(test_nlq, db_name_identifier, llm_client)
        
        # Print results
        print("\nResults:")
        print(f"Retrieved {len(table_names)} table names: {table_names}")
        print(f"Context length: {len(hyde_context)} characters")
        print("\nFirst 500 characters of context:")
        print(hyde_context[:500] + "..." if len(hyde_context) > 500 else hyde_context)
        
        print("\nTest completed successfully!")
        return True
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_hyde_module())
    
    # Exit with appropriate status code
    import sys
    sys.exit(0 if success else 1)
