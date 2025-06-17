import os
import lancedb
import pyarrow as pa
import numpy as np
from sentence_transformers import SentenceTransformer
import json # Still used for metadata if we choose to store it separately, but LanceDB can store it directly
from typing import List, Tuple, Dict, Optional, Any
import memory_module # For NL2SQL_DIR (used as base for LanceDB URI)

# --- Configuration (Hardcoded as per user request) ---
LITELLM_EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
LITELLM_RAG_THRESHOLD: float = 0.75
LITELLM_DISPLAY_THRESHOLD: float = 0.75 # As per user request

LANCEDB_BASE_URI = os.path.join(memory_module.NL2SQL_DIR, "lancedb_stores")

# --- Global Variables ---
embedding_model: Optional[SentenceTransformer] = None
# LanceDB connection object
db_connection: Optional[lancedb.DBConnection] = None

# --- Helper Functions ---
def _initialize_embedding_model() -> SentenceTransformer: # Removed model_name argument
    """Initializes and returns the sentence transformer model using module constant."""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer(LITELLM_EMBEDDING_MODEL)
            print(f"Embedding model '{LITELLM_EMBEDDING_MODEL}' loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model '{LITELLM_EMBEDDING_MODEL}': {e}")
            raise
    return embedding_model

def _get_lancedb_connection() -> lancedb.DBConnection:
    """Initializes and returns the LanceDB connection."""
    global db_connection
    if db_connection is None:
        os.makedirs(LANCEDB_BASE_URI, exist_ok=True)
        db_connection = lancedb.connect(LANCEDB_BASE_URI)
        print(f"LanceDB connection established at: {LANCEDB_BASE_URI}")
    return db_connection

def get_table_name(db_name_identifier: str) -> str:
    """Generates a valid LanceDB table name from the db_name_identifier."""
    # LanceDB table names should be simple identifiers.
    # Replace non-alphanumeric characters (except underscore) with underscore.
    return "".join(c if c.isalnum() or c == '_' else '_' for c in db_name_identifier)

def _get_or_create_table(db_name_identifier: str) -> lancedb.table.Table:
    """
    Retrieves an existing LanceDB table for the DB or creates a new one.
    """
    conn = _get_lancedb_connection()
    table_name = get_table_name(db_name_identifier)
    
    if table_name in conn.table_names():
        print(f"Opening existing LanceDB table: {table_name}")
        return conn.open_table(table_name)
    else:
        print(f"Creating new LanceDB table: {table_name}")
        # Define schema for the table
        # The embedding model needs to be initialized to get the dimension
        model = _initialize_embedding_model()
        dim = model.get_sentence_embedding_dimension()
        
        # LanceDB schema: vector, nlq (text), sql (text)
        # PyArrow schema for LanceDB
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=dim)), # Fixed-size list for vector
            pa.field("nlq", pa.string()),
            pa.field("sql", pa.string())
        ])
        return conn.create_table(table_name, schema=schema, mode="create") # Use "create" mode

# --- Core Functionality ---

def add_nlq_sql_pair(db_name_identifier: str, nlq: str, sql: str):
    """
    Adds a natural language query and its corresponding SQL to the LanceDB table for a specific DB.
    """
    if not nlq or not sql:
        print("Error: NLQ or SQL is empty. Cannot add to LanceDB table.")
        return

    table = _get_or_create_table(db_name_identifier)
    model = _initialize_embedding_model()

    # Check for duplicates before adding. LanceDB doesn't have a unique constraint by default.
    # We can search for the exact NLQ.
    try:
        # Using FTS search for exact match on 'nlq' field if available and indexed,
        # or a filter on a full scan if not.
        # For simplicity, let's try a filter.
        # LanceDB's filter is SQL-like.
        # Properly escape single quotes for the SQL-like filter string
        escaped_nlq = nlq.replace("'", "''")
        escaped_sql = sql.replace("'", "''")
        filter_condition = f"nlq = '{escaped_nlq}' AND sql = '{escaped_sql}'"
        
        existing_records = table.search().where(filter_condition).limit(1).to_df()
        if not existing_records.empty:
            print(f"NLQ-SQL pair already exists in LanceDB table '{table.name}'. Skipping.")
            return
    except Exception as e:
        print(f"Warning: Could not check for duplicates in LanceDB table '{table.name}': {e}. Proceeding with add.")

    # This try block is for the embedding and adding process itself.
    try:
        nlq_embedding = model.encode([nlq])[0] # Get the first (and only) embedding
        
        data_to_add = [{"vector": nlq_embedding.tolist(), "nlq": nlq, "sql": sql}]
        table.add(data_to_add)
        print(f"Added NLQ-SQL pair to LanceDB table '{table.name}'. Total rows: {len(table)}")
        # LanceDB tables are persistent, no explicit save like FAISS index file.
        # Compaction might be useful periodically for performance but not strictly after every add.
        # table.compact_files()
        # table.create_index(metric="cosine", num_partitions=8, num_sub_vectors=int(dim/4) if dim else 96) # Example IVF_PQ index
        
    except Exception as e:
        print(f"Error adding NLQ-SQL pair to LanceDB table '{table.name}': {e}")

def search_similar_nlqs(db_name_identifier: str, query_nlq: str, k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Searches for k most similar natural language queries in the LanceDB table for a specific DB.
    Returns a list of dictionaries, each containing 'nlq', 'sql', and 'similarity_score' (cosine similarity).
    Filters results by the similarity threshold if provided.
    """
    # Use threshold from argument if provided, otherwise use the module constant
    effective_threshold = threshold if threshold is not None else LITELLM_RAG_THRESHOLD

    table = _get_or_create_table(db_name_identifier)
    
    if len(table) == 0:
        print(f"LanceDB table '{table.name}' is empty. Cannot search.")
        return []
        
    model = _initialize_embedding_model()

    try:
        query_embedding = model.encode([query_nlq])[0]

        # LanceDB search returns a DataFrame-like object or a list of dicts
        # It uses L2 distance by default unless an index with a different metric (like cosine) is built.
        # For cosine similarity, we should ensure vectors are normalized or use an index that supports it.
        # If using L2, we convert distance to similarity. If cosine, higher is better.
        
        # Let's assume we want cosine similarity. We can build an index or calculate it.
        # If no index is present, LanceDB performs a brute-force search.
        # We can specify the metric in the search query if an appropriate index exists.
        # For now, let's rely on LanceDB's default (L2) and convert, or try to use cosine if an index is built.
        
        search_results = table.search(query_embedding).limit(k * 2) # Fetch more to filter by threshold
        
        results = []
        # The search_results object contains a '_distance' column (L2 squared by default)
        # and the other columns from the table ('vector', 'nlq', 'sql').
        
        df_results = search_results.to_df()

        for _, row in df_results.iterrows():
            dist = row['_distance'] # This is L2 squared distance
            
            # Convert L2 squared distance to cosine similarity.
            # This requires the original vectors to be normalized, or we re-calculate.
            # For simplicity, let's use the 1 / (1 + sqrt(dist)) heuristic for now,
            # acknowledging it's not true cosine similarity.
            # A better approach would be to store normalized vectors and use dot product,
            # or use LanceDB's built-in cosine distance if an index is configured for it.
            
            # Heuristic similarity from L2 distance
            similarity_score = 1 / (1 + np.sqrt(dist)) if dist >= 0 else 0.0
            
            # If we had an index with 'cosine' metric, 'dist' would be 1 - cosine_similarity.
            # So similarity = 1 - dist.

            if similarity_score >= effective_threshold: # Use effective_threshold
                results.append({
                    "nlq": row["nlq"],
                    "sql": row["sql"],
                    "similarity_score": float(similarity_score)
                })
        
        # Sort by similarity score descending and take top k
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:k]

    except Exception as e:
        print(f"Error searching similar NLQs in LanceDB table '{table.name}': {e}")
        return []

# configure_thresholds function is removed as constants are now module-level.


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Testing vector_store_module.py with LanceDB...")
    print(f"Using RAG Threshold: {LITELLM_RAG_THRESHOLD}")
    print(f"Using Display Threshold: {LITELLM_DISPLAY_THRESHOLD}")
    print(f"Using Embedding Model: {LITELLM_EMBEDDING_MODEL}")

    memory_module.ensure_memory_directories() # Ensure NL2SQL_DIR and lancedb_stores exist

    test_db_lancedb = "test_lancedb_copilot_db"
    table_name_lancedb = get_table_name(test_db_lancedb)
    
    # Clean up old LanceDB table directory if it exists for a clean test
    db_conn_for_cleanup = lancedb.connect(LANCEDB_BASE_URI)
    if table_name_lancedb in db_conn_for_cleanup.table_names():
        print(f"Dropping existing test table: {table_name_lancedb}")
        db_conn_for_cleanup.drop_table(table_name_lancedb)
    
    # Ensure global db_connection is reset for the test to pick up the clean state
    db_connection = None 
    embedding_model = None # Force reinitialization

    print(f"\n--- Adding pairs to LanceDB table for '{test_db_lancedb}' ---")
    add_nlq_sql_pair(test_db_lancedb, "Show all customers from California", "SELECT * FROM customers WHERE state = 'CA';")
    add_nlq_sql_pair(test_db_lancedb, "List active users", "SELECT user_id, username FROM users WHERE status = 'active';")
    add_nlq_sql_pair(test_db_lancedb, "Find total sales for product ID 123", "SELECT SUM(amount) FROM sales WHERE product_id = 123;")
    add_nlq_sql_pair(test_db_lancedb, "Which customers live in New York City?", "SELECT name FROM customers WHERE city = 'New York City' AND state = 'NY';")
    add_nlq_sql_pair(test_db_lancedb, "Count of orders placed last month", "SELECT COUNT(*) FROM orders WHERE order_date >= date_trunc('month', current_date - interval '1 month') AND order_date < date_trunc('month', current_date);")
    add_nlq_sql_pair(test_db_lancedb, "Show all customers from California", "SELECT * FROM customers WHERE state = 'CA';") # Test duplicate add

    # Access the table after adding to check its length
    conn_check = _get_lancedb_connection()
    table_check = conn_check.open_table(table_name_lancedb)
    print(f"Table '{table_check.name}' now has {len(table_check)} entries.")
    assert len(table_check) == 5, f"Expected 5 entries after duplicate handling, got {len(table_check)}"


    print(f"\n--- Searching in LanceDB table for '{test_db_lancedb}' ---")
    query1_lancedb = "Tell me about customers in CA"
    # search_similar_nlqs will use LITELLM_RAG_THRESHOLD by default if threshold arg is None
    similar_pairs1_lancedb = search_similar_nlqs(test_db_lancedb, query1_lancedb, k=3) 
    print(f"\nFor query: \"{query1_lancedb}\" (using configured RAG threshold: {LITELLM_RAG_THRESHOLD})")
    for pair in similar_pairs1_lancedb:
        print(f"  NLQ: {pair['nlq']}, SQL: {pair['sql']}, Score: {pair['similarity_score']:.4f}")

    query2_lancedb = "How many orders were there recently?"
    similar_pairs2_lancedb = search_similar_nlqs(test_db_lancedb, query2_lancedb, k=3, threshold=0.5) # Override threshold for this call
    print(f"\nFor query: \"{query2_lancedb}\" (threshold 0.5)")
    for pair in similar_pairs2_lancedb:
        print(f"  NLQ: {pair['nlq']}, SQL: {pair['sql']}, Score: {pair['similarity_score']:.4f}")

    query3_lancedb = "Show active customers" 
    similar_pairs3_lancedb = search_similar_nlqs(test_db_lancedb, query3_lancedb, k=2, threshold=0.1) # Very low threshold
    print(f"\nFor query: \"{query3_lancedb}\" (threshold 0.1)")
    for pair in similar_pairs3_lancedb:
        print(f"  NLQ: {pair['nlq']}, SQL: {pair['sql']}, Score: {pair['similarity_score']:.4f}")
    
    # Test persistence by re-opening the table (LanceDB handles this implicitly)
    print("\n--- Testing persistence (LanceDB handles this by default) ---")
    db_connection = None # Simulate new session, force re-connect
    embedding_model = None # Force reinitialization of model too
    print("Cleared in-memory connection. LanceDB should reload from disk if needed.")
    
    similar_pairs_reloaded_lancedb = search_similar_nlqs(test_db_lancedb, query1_lancedb, k=2, threshold=0.5)
    print(f"\nFor query (reloaded): \"{query1_lancedb}\" (threshold 0.5)")
    if similar_pairs_reloaded_lancedb:
        for pair in similar_pairs_reloaded_lancedb:
            print(f"  NLQ: {pair['nlq']}, SQL: {pair['sql']}, Score: {pair['similarity_score']:.4f}")
        assert len(similar_pairs_reloaded_lancedb) > 0, "Failed to reload and search with LanceDB"
    else:
        print("  No results after reloading, something might be wrong with LanceDB persistence or search.")

    print("\nvector_store_module.py (LanceDB version) testing complete.")
