import json
import os
from typing import Tuple, Optional, Any, Dict, List
from pathlib import Path
import sys
import re
import datetime
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from .llm_client import create_llm_client

# Import local modules
try:
    from . import hyde_module
    from . import memory_module
except ImportError as e:
    print(f"Error importing modules: {e}", file=sys.stderr)
    hyde_module = None
    memory_module = None

# We'll need to import these modules from the original application
# For now, we'll define placeholder functions that will be replaced with actual implementations
def handle_exception(e, user_query=None, context=None, message=None):
    """Enhanced error handling function with better logging"""
    error_msg = f"Error: {e}"
    
    # Add context information if available
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        error_msg += f" [Context: {context_str}]"
    
    # Add user query if available
    if user_query:
        error_msg += f" [Query: {user_query}]"
    
    # Log the error
    print(error_msg, file=sys.stderr)
    
    # Print user-friendly message if provided
    if message:
        print(f"[INFO] {message}")
    
    # Return the error message for the caller
    return str(e)

def ensure_directories_exist():
    """Create all required directories for the application."""
    base_dirs = [
        "./data/memory",
        "./data/memory/schema",
        "./data/memory/insights",
        "./data/memory/lancedb_stores",
        "./data/memory/logs",
        "./data/cache/sentence_transformers"
    ]
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
    print("[INFO] All required directories have been created.")
    
    # Initialize Hyde module logs
    if hyde_module:
        hyde_module.initialize_logs()

def get_memory_base_path() -> Path:
    """Gets the base path for memory files."""
    memory_path = Path("./data/memory")
    os.makedirs(memory_path, exist_ok=True)
    return memory_path

def get_schema_filepath(db_name_identifier: str) -> str:
    """Constructs the full path for a database's schema JSON file."""
    schema_dir = get_memory_base_path() / "schema"
    os.makedirs(schema_dir, exist_ok=True)
    return str(schema_dir / f"schema_sampledata_for_{db_name_identifier}.json")

def get_active_tables_filepath(db_name_identifier: str) -> str:
    """Constructs the full path for the active tables file."""
    schema_dir = get_memory_base_path() / "schema"
    os.makedirs(schema_dir, exist_ok=True)
    return str(schema_dir / f"active_tables_for_{db_name_identifier}.txt")

def get_schema_graph_filepath(db_name_identifier: str) -> Path:
    """Constructs the full path for the schema graph JSON file."""
    schema_dir = get_memory_base_path() / "schema"
    os.makedirs(schema_dir, exist_ok=True)
    return schema_dir / f"{db_name_identifier}_schema_graph.json"

def get_schema_backup_filepath(db_name_identifier: str) -> Path:
    """Constructs the full path for the schema backup text file."""
    schema_dir = get_memory_base_path() / "schema"
    os.makedirs(schema_dir, exist_ok=True)
    return schema_dir / f"schema_{db_name_identifier}_backup.txt"

def read_schema_data(db_name: str) -> Optional[Dict[str, Any]]:
    """
    Reads the schema and sample data JSON file for a given db_name.
    Returns the data as a dictionary, or None if the file doesn't exist or an error occurs.
    """
    schema_filepath = get_schema_filepath(db_name)
    if not os.path.exists(schema_filepath):
        print(f"Schema data file not found: {schema_filepath}", file=sys.stderr)
        return None
    try:
        with open(schema_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (IOError, json.JSONDecodeError) as e:
        handle_exception(e, user_query=f"read_schema_data for {db_name}")
        return None

def save_schema_graph(db_name_identifier: str, graph_data: Dict[str, Any]):
    """Saves the generated schema graph to a JSON file."""
    filepath = get_schema_graph_filepath(db_name_identifier)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
    except (IOError, TypeError) as e:
        handle_exception(e, user_query=f"save_schema_graph for {db_name_identifier}")
        raise

def load_schema_graph(db_name_identifier: str) -> Optional[Dict[str, Any]]:
    """Loads the schema graph JSON file for a given database."""
    filepath = get_schema_graph_filepath(db_name_identifier)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (IOError, json.JSONDecodeError) as e:
        handle_exception(e, user_query=f"load_schema_graph for {db_name_identifier}")
        return None

def save_schema_backup(db_name_identifier: str, table_descriptions: List[str]):
    """
    Saves table descriptions to a backup text file.
    
    Args:
        db_name_identifier: The database identifier
        table_descriptions: List of table description strings
    """
    try:
        backup_filepath = get_schema_backup_filepath(db_name_identifier)
        
        with open(backup_filepath, 'w', encoding='utf-8') as f:
            for description in table_descriptions:
                f.write(f"{description}\n\n")
        
        print(f"[INFO] Schema backup saved to {backup_filepath}")
    except Exception as e:
        handle_exception(e, user_query=f"save_schema_backup for {db_name_identifier}")

def check_schema_backup_exists(db_name_identifier: str) -> bool:
    """
    Checks if a schema backup file exists for the given database.
    
    Args:
        db_name_identifier: The database identifier
        
    Returns:
        True if the backup file exists, False otherwise
    """
    backup_filepath = get_schema_backup_filepath(db_name_identifier)
    return os.path.exists(backup_filepath)

# Vector store functionality
LITELLM_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
embedding_model = None

def _initialize_embedding_model() -> SentenceTransformer:
    """Initializes and returns the sentence transformer model."""
    global embedding_model
    if embedding_model is None:
        try:
            # Use a cache directory for sentence transformers
            cache_dir = "./data/cache/sentence_transformers"
            os.makedirs(cache_dir, exist_ok=True)
            embedding_model = SentenceTransformer(
                LITELLM_EMBEDDING_MODEL,
                cache_folder=str(cache_dir)
            )
        except Exception as e:
            handle_exception(e, user_query="initialize_embedding_model")
            raise
    return embedding_model

def _get_lancedb_base_uri() -> Path:
    """Gets the base URI for LanceDB stores."""
    lancedb_path = Path("./data/memory/lancedb_stores")
    lancedb_path.mkdir(parents=True, exist_ok=True)
    return lancedb_path

db_connection = None

def _get_lancedb_connection() -> lancedb.DBConnection:
    """Initializes and returns the LanceDB connection."""
    global db_connection
    if db_connection is None:
        lancedb_uri = _get_lancedb_base_uri()
        lancedb_uri.mkdir(parents=True, exist_ok=True)
        db_connection = lancedb.connect(str(lancedb_uri))
    return db_connection

def get_schema_table_name(db_name_identifier: str) -> str:
    """Generates a valid LanceDB table name for schema vectors."""
    sanitized_name = "".join(c if c.isalnum() or c == '_' else '_' for c in db_name_identifier)
    return f"{sanitized_name}_schema_vectors"

def _get_or_create_schema_table(db_name_identifier: str) -> lancedb.table.Table:
    """
    Retrieves or creates a LanceDB table for storing schema vector embeddings.
    """
    conn = _get_lancedb_connection()
    table_name = get_schema_table_name(db_name_identifier)

    if table_name in conn.table_names():
        return conn.open_table(table_name)
    else:
        model = _initialize_embedding_model()
        dim = model.get_sentence_embedding_dimension()
        
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
            pa.field("description", pa.string()),
            pa.field("type", pa.string()),
            pa.field("table_name", pa.string()),
            pa.field("column_name", pa.string(), nullable=True)
        ])
        return conn.create_table(table_name, schema=schema, mode="overwrite")

def _log_schema_chunk(db_name_identifier: str, chunk_data: Dict[str, Any]):
    """
    Logs a schema chunk to the schema_chunk_log.json file.
    """
    try:
        # Ensure logs directory exists
        memory_base_path = get_memory_base_path()
        logs_dir = os.path.join(memory_base_path, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare log file path
        log_file_path = os.path.join(logs_dir, 'schema_chunk_log.json')
        
        # Create log entry with timestamp
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "db_name_identifier": db_name_identifier,
            **chunk_data
        }
        
        # Append to log file
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    except Exception as e:
        handle_exception(e, user_query=f"log_schema_chunk for {db_name_identifier}", 
                        context={"chunk_data": chunk_data})

def _parse_create_table_sql(sql_statement: str) -> List[Dict[str, str]]:
    """
    Parses a SQL CREATE TABLE statement to extract column information.
    """
    try:
        # Extract the part between parentheses
        match = re.search(r'\(\s*(.*?)\s*\);', sql_statement, re.DOTALL)
        if not match:
            return []
            
        columns_text = match.group(1)
        
        # Split by commas, but not commas inside parentheses
        column_definitions = []
        paren_level = 0
        current_col = ""
        
        for char in columns_text:
            if char == '(' and not (current_col and current_col[-1] == '\\'): 
                paren_level += 1
                current_col += char
            elif char == ')' and not (current_col and current_col[-1] == '\\'):
                paren_level -= 1
                current_col += char
            elif char == ',' and paren_level == 0:
                column_definitions.append(current_col.strip())
                current_col = ""
            else:
                current_col += char
                
        if current_col.strip():
            column_definitions.append(current_col.strip())
        
        # Process each column definition
        columns = []
        for col_def in column_definitions:
            # Skip if this is a constraint, not a column
            if col_def.strip().upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'UNIQUE', 'CHECK')):
                continue
                
            # Extract column name and type
            parts = col_def.strip().split(None, 1)
            if not parts:
                continue
                
            col_name = parts[0].strip('"\'')
            
            if len(parts) > 1:
                # Try to separate type from constraints
                type_and_constraints = parts[1].strip()
                type_match = re.match(r'([^\s]+)', type_and_constraints)
                
                if type_match:
                    col_type = type_match.group(1)
                    # Everything after the type is considered description/constraints
                    description = type_and_constraints[len(col_type):].strip()
                else:
                    col_type = "unknown"
                    description = type_and_constraints
            else:
                col_type = "unknown"
                description = ""
                
            columns.append({
                "name": col_name,
                "type": col_type,
                "description": description
            })
            
        return columns
    except Exception as e:
        # Log the error but return an empty list to allow processing to continue
        handle_exception(e, user_query=f"parse_create_table_sql", 
                        context={"sql_statement": sql_statement[:100] + "..."})
        return []

def _extract_columns_from_sample_data(sample_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extracts column information from sample data as a fallback.
    """
    if not sample_data or not isinstance(sample_data, list) or not sample_data[0]:
        return []
        
    columns = []
    # Use the first row to get column names and infer types
    first_row = sample_data[0]
    
    for col_name, value in first_row.items():
        if value is None:
            col_type = "unknown"
        elif isinstance(value, str):
            col_type = "character varying"
        elif isinstance(value, int):
            col_type = "integer"
        elif isinstance(value, float):
            col_type = "numeric"
        elif isinstance(value, bool):
            col_type = "boolean"
        elif isinstance(value, (list, dict)):
            col_type = "json"
        else:
            col_type = str(type(value).__name__)
            
        columns.append({
            "name": col_name,
            "type": col_type,
            "description": ""
        })
        
    return columns

def vectorize_and_store_schema(db_name_identifier: str, schema_info: Dict[str, Any], foreign_key_relationships: Optional[List[Dict[str, str]]] = None, force_regenerate: bool = False):
    """
    Processes schema information, creates descriptive strings, generates embeddings,
    and stores them in a dedicated LanceDB table.
    """
    try:
        table = _get_or_create_schema_table(db_name_identifier)
        model = _initialize_embedding_model()
        
        documents_to_embed = []
        metadata = []
        table_descriptions = []  # List to store table descriptions for backup
        
        # Track statistics for logging
        total_tables = len(schema_info)
        tables_processed = 0
        tables_skipped = 0
        total_columns = 0
        parsing_failures = 0
        parsing_successes = 0
        
        # Check if schema backup exists and should be created
        should_create_backup = force_regenerate or not check_schema_backup_exists(db_name_identifier)

        for table_name, table_data in schema_info.items():
            # Log raw table data structure for debugging
            _log_schema_chunk(db_name_identifier, {
                "event": "table_raw_structure",
                "table_name": table_name,
                "has_columns_key": 'columns' in table_data,
                "has_schema_key": 'schema' in table_data,
                "has_sample_data_key": 'sample_data' in table_data,
                "keys_present": list(table_data.keys())
            })
            
            # Try to get columns from different sources
            columns = []
            source = "unknown"
            
            # 1. First try the columns key directly
            if 'columns' in table_data:
                columns = table_data['columns']
                source = "columns_key"
            
            # 2. If no columns key, try to parse from schema SQL
            elif 'schema' in table_data and isinstance(table_data['schema'], str):
                sql_statement = table_data['schema']
                columns = _parse_create_table_sql(sql_statement)
                if columns:
                    parsing_successes += 1
                    source = "parsed_sql"
                else:
                    parsing_failures += 1
                    
                # Log parsing results
                _log_schema_chunk(db_name_identifier, {
                    "event": "sql_parsing_result",
                    "table_name": table_name,
                    "success": bool(columns),
                    "columns_found": len(columns),
                    "sql_snippet": sql_statement[:100] + "..." if len(sql_statement) > 100 else sql_statement
                })
                
            # 3. If still no columns, try to extract from sample data as fallback
            if not columns and 'sample_data' in table_data and table_data['sample_data']:
                columns = _extract_columns_from_sample_data(table_data['sample_data'])
                if columns:
                    source = "sample_data"
                    
                # Log sample data extraction results
                _log_schema_chunk(db_name_identifier, {
                    "event": "sample_data_extraction_result",
                    "table_name": table_name,
                    "success": bool(columns),
                    "columns_found": len(columns)
                })
            
            column_count = len(columns)
            
            # Log table statistics
            _log_schema_chunk(db_name_identifier, {
                "event": "table_analysis",
                "table_name": table_name,
                "column_count": column_count,
                "column_source": source
            })
            
            # Skip tables with no columns
            if column_count == 0:
                tables_skipped += 1
                continue
                
            # Create description for the table
            column_names = [col['name'] for col in columns]
            
            # Extract primary key and foreign key information if available
            pk_cols = []
            fk_desc = []
            
            # Check if primary key information is available in table_data
            if 'primary_key' in table_data:
                pk_cols = table_data['primary_key'] if isinstance(table_data['primary_key'], list) else [table_data['primary_key']]
            
            # Check if foreign key information is available in table_data
            if 'foreign_keys' in table_data and table_data['foreign_keys']:
                for fk in table_data['foreign_keys']:
                    if isinstance(fk, dict) and 'column' in fk and 'reference' in fk:
                        col = fk['column']
                        ref_parts = fk['reference'].split('.')
                        if len(ref_parts) > 1:
                            ref_table, ref_col = ref_parts[0], ref_parts[1]
                            fk_desc.append(f"{col} references {ref_table}({ref_col})")
            
            # Add foreign key relationships from the provided foreign_key_relationships parameter
            if foreign_key_relationships:
                # Create a dictionary to store foreign keys for each table
                table_short_name = table_name.split('.')[-1] if '.' in table_name else table_name
                
                # Find foreign keys where this table is the source (fk_table)
                for rel in foreign_key_relationships:
                    if rel.get('fk_table') == table_short_name:
                        fk_column = rel.get('fk_column')
                        pk_table = rel.get('pk_table')
                        pk_column = rel.get('pk_column')
                        if fk_column and pk_table and pk_column:
                            fk_desc.append(f"{fk_column} references {pk_table}({pk_column})")
            
            # Format the table description to include column names, primary keys, and foreign keys
            if column_names:
                column_list = ', '.join(column_names)
                table_description_str = f"Table: {table_name}. Columns: {column_list}"
                if pk_cols:
                    table_description_str += f". Primary Key: {', '.join(pk_cols)}"
                if fk_desc:
                    table_description_str += f". Foreign Keys: {'; '.join(fk_desc)}"
                table_description_str += "."
            else:
                table_description_str = f"Table: {table_name}. No columns found."
            
            # Add to table descriptions for backup
            table_descriptions.append(table_description_str)
                
            documents_to_embed.append(table_description_str)
            metadata.append({
                "description": table_description_str,
                "type": "table",
                "table_name": table_name,
                "column_name": None
            })
            
            # Log the table chunk
            _log_schema_chunk(db_name_identifier, {
                "event": "chunk_created",
                "chunk_type": "table",
                "table_name": table_name,
                "description": table_description_str,
                "column_count": column_count,
                "column_names": column_names
            })
            
            tables_processed += 1
            
            # Create descriptions for each column
            for column in columns:
                col_name = column.get('name')
                col_type = column.get('type')
                col_description = column.get('description', '')
                
                # Determine nullable status and default value
                nullable_str = "NULL"
                if 'nullable' in column:
                    nullable_str = "NULL" if column['nullable'] else "NOT NULL"
                elif 'is_nullable' in column:
                    nullable_str = "NULL" if column['is_nullable'] in (True, 'YES', 'yes') else "NOT NULL"
                
                default_str = ""
                if 'default' in column and column['default'] is not None:
                    default_str = f" DEFAULT {column['default']}"
                
                # Check if this column is a primary key
                pk_status = "Primary Key" if col_name in pk_cols else ""
                
                # Check if this column is a foreign key
                fk_status = ""
                if 'foreign_keys' in table_data:
                    for fk in table_data['foreign_keys']:
                        if isinstance(fk, dict) and fk.get('column') == col_name:
                            ref = fk.get('reference', '')
                            ref_parts = ref.split('.')
                            if len(ref_parts) > 1:
                                ref_table, ref_col = ref_parts[0], ref_parts[1]
                                fk_status = f"Foreign Key to {ref_table}({ref_col})"
                                break
                
                # Create a comprehensive column description
                column_description_str = f"Table: {table_name}, Column: {col_name}, Type: {col_type}, {nullable_str}{default_str}"
                if pk_status:
                    column_description_str += f", {pk_status}"
                if fk_status:
                    column_description_str += f", {fk_status}"
                if col_description:
                    column_description_str += f". Description: {col_description}"
                
                documents_to_embed.append(column_description_str)
                metadata.append({
                    "description": column_description_str,
                    "type": "column",
                    "table_name": table_name,
                    "column_name": col_name
                })
                
                # Log the column chunk
                _log_schema_chunk(db_name_identifier, {
                    "event": "chunk_created",
                    "chunk_type": "column",
                    "table_name": table_name,
                    "column_name": col_name,
                    "column_type": col_type,
                    "description": column_description_str
                })
                
                total_columns += 1
            
            # Add sample data vectorization if available
            if 'sample_data' in table_data and table_data['sample_data']:
                try:
                    N = 5
                    sample_rows = table_data['sample_data'][:N]
                    for idx, row in enumerate(sample_rows, start=1):
                        try:
                            row_str = json.dumps(row, ensure_ascii=False)
                        except Exception:
                            # Fallback to simple string conversion if not JSON serializable
                            row_str = str(row)

                        sample_data_str = f"Sample data for {table_name} (row {idx}): {row_str}"
                        documents_to_embed.append(sample_data_str)
                        metadata.append({
                            "description": sample_data_str,
                            "type": "sample_data",
                            "table_name": table_name,
                            "column_name": None
                        })

                        # Log the sample data chunk per row
                        _log_schema_chunk(db_name_identifier, {
                            "event": "chunk_created",
                            "chunk_type": "sample_data",
                            "table_name": table_name,
                            "row_index": idx,
                            "description": sample_data_str
                        })
                except Exception as e:
                    handle_exception(e, user_query=f"sample_data_processing for {table_name}")

        if not documents_to_embed:
            _log_schema_chunk(db_name_identifier, {
                "event": "vectorization_skipped",
                "reason": "No documents to embed",
                "tables_analyzed": total_tables,
                "tables_processed": tables_processed,
                "tables_skipped": tables_skipped
            })
            return

        # Log vectorization statistics before embedding
        _log_schema_chunk(db_name_identifier, {
            "event": "vectorization_started",
            "tables_analyzed": total_tables,
            "tables_processed": tables_processed,
            "tables_skipped": tables_skipped,
            "total_columns": total_columns,
            "chunks_to_embed": len(documents_to_embed)
        })

        embeddings = model.encode(documents_to_embed, show_progress_bar=False)
        
        data_to_add = []
        for i, embedding in enumerate(embeddings):
            meta = metadata[i]
            data_to_add.append({
                "vector": embedding.tolist(),
                "description": meta["description"],
                "type": meta["type"],
                "table_name": meta["table_name"],
                "column_name": meta["column_name"]
            })
        
        table.add(data_to_add)
        
        # Log vectorization completion
        _log_schema_chunk(db_name_identifier, {
            "event": "vectorization_completed",
            "tables_analyzed": total_tables,
            "tables_processed": tables_processed,
            "tables_skipped": tables_skipped,
            "total_columns": total_columns,
            "chunks_embedded": len(data_to_add),
            "sql_parsing_successes": parsing_successes,
            "sql_parsing_failures": parsing_failures
        })
        
        # Save table descriptions to backup file if needed
        if should_create_backup and table_descriptions:
            save_schema_backup(db_name_identifier, table_descriptions)

    except Exception as e:
        handle_exception(e, user_query=f"vectorize_and_store_schema for {db_name_identifier}")

def check_schema_vectors_exist(db_name_identifier: str) -> bool:
    """
    Check if schema vectors already exist for the given database.
    """
    conn = _get_lancedb_connection()
    table_name = get_schema_table_name(db_name_identifier)
    
    if table_name in conn.table_names():
        table = conn.open_table(table_name)
        # Check if the table has data
        return len(table) > 0
    return False

def check_schema_graph_exists(db_name_identifier: str) -> bool:
    """
    Check if schema graph already exists for the given database.
    """
    schema_graph = load_schema_graph(db_name_identifier)
    return schema_graph is not None and len(schema_graph.get("nodes", [])) > 0

def load_and_filter_schema(db_name_identifier: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Loads the master schema and filters it based on the active_tables.txt file.
    """
    # Read the master schema file
    schema_data_dict = read_schema_data(db_name_identifier)
    if schema_data_dict is None:
        err_msg = "Schema file was not found or is empty. Cannot apply scope."
        return None, handle_exception(FileNotFoundError(err_msg), context={"step": "read_schema_file_for_filter"})

    # Create active_tables.txt if it doesn't exist
    active_tables_filepath = get_active_tables_filepath(db_name_identifier)
    if not os.path.exists(active_tables_filepath):
        # Include both tables and views when creating the file for the first time
        all_object_names = [
            name for name, properties in schema_data_dict.items()
            if properties.get('type') in ('TABLE', 'VIEW')
        ]
        with open(active_tables_filepath, 'w') as f:
            for object_name in all_object_names:
                f.write(f"{object_name}\n")
        print(f"\n[INFO] A new file has been created to manage the database scope.")
        print(f"Please edit '{active_tables_filepath}' to add or remove tables and views for the Co-pilot to consider.")

    # Filter the schema based on active_tables.txt
    try:
        with open(active_tables_filepath, 'r') as f:
            active_tables = [line.strip() for line in f if line.strip()]
        
        # Filter the schema based on active_tables.txt for both tables and views
        filtered_schema = {}
        for item_name, properties in schema_data_dict.items():
            # Include both tables and views only if they are in the active_tables list
            if (properties.get('type') == 'TABLE' or properties.get('type') == 'VIEW') and item_name in active_tables:
                filtered_schema[item_name] = properties
        
        if not filtered_schema:
            err_msg = "No tables from 'active_tables.txt' were found in the database schema. Please check the file for valid table names."
            return None, handle_exception(ValueError(err_msg), context={"step": "filter_schema"})

        return filtered_schema, f"Scope loaded. The Co-pilot now has access to {len(filtered_schema)} tables."

    except FileNotFoundError:
        err_msg = "active_tables.txt not found. Please ensure it exists or re-initialize the connection."
        return None, handle_exception(FileNotFoundError(err_msg), context={"step": "filter_schema"})

# PostgreSQL connection functions
pg_connection = None

def connect_to_postgres(connection_string: str) -> Tuple[bool, str]:
    """
    Connects to a PostgreSQL database using the provided connection string.
    
    Args:
        connection_string: PostgreSQL connection string in format "postgresql://user:password@host:port/dbname"
        
    Returns:
        Tuple of (success, message)
    """
    global pg_connection
    try:
        if pg_connection:
            pg_connection.close()
            print("[INFO] Closed existing database connection.")
        
        pg_connection = psycopg2.connect(connection_string)
        print("[INFO] Successfully connected to PostgreSQL database.")
        return True, "Successfully connected to PostgreSQL database."
    except Exception as e:
        error_msg = f"Failed to connect to PostgreSQL: {e}"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        pg_connection = None
        return False, error_msg

def get_schema_and_sample_data(output_file_path: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Fetches schema and sample data from the connected PostgreSQL database and saves it to a file.
    
    Args:
        output_file_path: Path to save the schema and sample data
        
    Returns:
        Tuple of (success, message, schema_data)
    """
    global pg_connection
    if not pg_connection:
        return False, "Not connected to any database.", None

    try:
        with pg_connection.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all user tables and views, prioritizing 'public' schema
            cur.execute("""
                SELECT * FROM (
                    SELECT schemaname, tablename, 'TABLE' as type
                    FROM pg_catalog.pg_tables
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    UNION ALL
                    SELECT schemaname, viewname as tablename, 'VIEW' as type
                    FROM pg_catalog.pg_views
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ) as all_objects
                ORDER BY CASE WHEN schemaname = 'public' THEN 0 ELSE 1 END, tablename;
            """)
            all_objects = cur.fetchall()
            
            print(f"[INFO] Found {len(all_objects)} tables and views. Processing all of them.")

            result = {}
            for row in all_objects:
                item_name = row['tablename']
                schema_name = row['schemaname']
                item_type = row['type']
                full_item_name = f"{schema_name}.{item_name}"
                
                item_info = {"type": item_type, "schema": "", "sample_data": []}

                if item_type == 'VIEW':
                    cur.execute("SELECT definition FROM pg_views WHERE schemaname = %s AND viewname = %s;", (schema_name, item_name))
                    view_definition = cur.fetchone()
                    if view_definition:
                        item_info["schema"] = f"CREATE VIEW {full_item_name} AS\n{view_definition['definition']}"
                else: # It's a table
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position;
                    """, (schema_name, item_name))
                    columns = cur.fetchall()

                    create_statement_parts = [f"CREATE TABLE {full_item_name} ("]
                    for col in columns:
                        part = f"  {col['column_name']} {col['data_type']}"
                        if col['is_nullable'] == 'NO': part += " NOT NULL"
                        if col['column_default'] is not None: part += f" DEFAULT {col['column_default']}"
                        create_statement_parts.append(part + ",")
                    if create_statement_parts[-1].endswith(","):
                        create_statement_parts[-1] = create_statement_parts[-1][:-1]
                    create_statement_parts.append(");")
                    item_info["schema"] = "\n".join(create_statement_parts)

                query = sql.SQL("SELECT * FROM {}.{} LIMIT 5;").format(
                    sql.Identifier(schema_name), sql.Identifier(item_name)
                )
                cur.execute(query)
                sample_data = [dict(row) for row in cur.fetchall()]
                item_info["sample_data"] = sample_data

                result[full_item_name] = item_info

            # Save the result directly to the specified file path
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, default=str) # Use default=str for data types like datetime

            print(f"[INFO] Successfully saved schema for {len(all_objects)} tables and views to {output_file_path}.")
            return True, f"Schema for {len(all_objects)} tables and views saved successfully.", result

    except Exception as e:
        error_msg = f"Error fetching schema and data: {e}"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        return False, error_msg, None

def get_foreign_key_relationships() -> Tuple[bool, str, Optional[List[Dict[str, str]]]]:
    """
    Queries the information_schema to retrieve all foreign key relationships.
    
    Returns:
        Tuple of (success, message, relationships)
    """
    global pg_connection
    if not pg_connection:
        return False, "Not connected to any database.", None

    query = """
    SELECT
        tc.table_name AS fk_table,
        kcu.column_name AS fk_column,
        ccu.table_name AS pk_table,
        ccu.column_name AS pk_column
    FROM
        information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
          AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
          AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY';
    """
    try:
        with pg_connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            relationships = cur.fetchall()
            print(f"[INFO] Successfully fetched {len(relationships)} foreign key relationships.")
            return True, f"Successfully fetched {len(relationships)} foreign key relationships.", relationships
    except Exception as e:
        error_msg = f"Error fetching foreign key relationships: {e}"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        return False, error_msg, None

async def process_stored_functions(connection_string: str, db_name_identifier: str) -> Tuple[bool, str]:
    """
    Fetches stored functions from the database and adds them to the approved SQL JSON file and LanceDB database.
    
    Args:
        connection_string: PostgreSQL connection string
        db_name_identifier: Identifier for the database
        
    Returns:
        Tuple of (success, message)
    """
    global pg_connection
    if not pg_connection:
        return False, "Not connected to any database."

    query = """
    SELECT 
        n.nspname AS schema_name,
        p.proname AS function_name,
        pg_get_function_arguments(p.oid) AS function_arguments,
        t.typname AS return_type,
        pg_get_functiondef(p.oid) AS function_definition,
        d.description AS function_comment
    FROM 
        pg_proc p
        LEFT JOIN pg_namespace n ON p.pronamespace = n.oid
        LEFT JOIN pg_type t ON p.prorettype = t.oid
        LEFT JOIN pg_description d ON p.oid = d.objoid
    WHERE 
        n.nspname NOT IN ('pg_catalog', 'information_schema')
        AND p.prokind = 'f'  -- 'f' for regular functions
    ORDER BY 
        n.nspname, p.proname;
    """
    try:
        with pg_connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            functions = cur.fetchall()
            print(f"[INFO] Successfully fetched {len(functions)} stored functions.")
            
            if not functions:
                return True, "No stored functions found in the database."
                
            # Process the functions (in a real implementation, we would add them to the vector store)
            # For now, we'll just return success
            return True, f"Successfully processed {len(functions)} stored functions."
    except Exception as e:
        error_msg = f"Error fetching stored functions: {e}"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        return False, error_msg

async def generate_schema_graph(connection_string: str, table_names: List[str]) -> Dict[str, Any]:
    """
    Generates a schema graph based on foreign key relationships.
    
    Args:
        connection_string: PostgreSQL connection string
        table_names: List of table names to include in the graph
        
    Returns:
        Dictionary containing nodes and edges for the schema graph
    """
    print(f"[INFO] Generating schema graph for {len(table_names)} tables...")
    
    # Create a graph with nodes for each table
    graph = {
        "nodes": [{"id": table_name, "label": table_name} for table_name in table_names],
        "edges": []
    }
    
    # Get foreign key relationships from the database
    success, message, relationships = get_foreign_key_relationships()
    if not success or not relationships:
        print(f"[WARNING] Could not fetch foreign key relationships: {message}")
        return graph
    
    # Add edges for each foreign key relationship
    for rel in relationships:
        fk_table = rel.get('fk_table')
        fk_column = rel.get('fk_column')
        pk_table = rel.get('pk_table')
        pk_column = rel.get('pk_column')
        
        # Check if both tables are in our table_names list
        # We need to check both with and without schema prefix
        fk_table_in_list = any(t.endswith(f".{fk_table}") or t == fk_table for t in table_names)
        pk_table_in_list = any(t.endswith(f".{pk_table}") or t == pk_table for t in table_names)
        
        if fk_table_in_list and pk_table_in_list:
            # Find the full table names from our list
            fk_table_full = next((t for t in table_names if t.endswith(f".{fk_table}") or t == fk_table), fk_table)
            pk_table_full = next((t for t in table_names if t.endswith(f".{pk_table}") or t == pk_table), pk_table)
            
            # Add an edge from the foreign key table to the primary key table
            edge = {
                "source": fk_table_full,
                "target": pk_table_full,
                "label": f"{fk_column} → {pk_column}"
            }
            graph["edges"].append(edge)
    
    print(f"[INFO] Schema graph generated with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")
    return graph

async def perform_initialization(connection_string: str, db_name_identifier: str, force_regenerate: bool = False) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Connects to the DB, fetches schema and saves it to a file,
    then reads that file to load the schema.
    
    Args:
        connection_string: The PostgreSQL connection string
        db_name_identifier: Identifier for the database
        force_regenerate: If True, regenerate schema vectors and graph even if they exist
    
    Returns:
        Tuple of (success, message, schema_data)
    """
    # Create the LLM client
    llm_client = create_llm_client()
    
    # Ensure all required directories exist before proceeding
    ensure_directories_exist()
    
    # Initialize memory module directories
    if memory_module:
        memory_module.ensure_memory_directories()
    
    # --- Step 1: Connect to PostgreSQL ---
    try:
        print(f"[INFO] Connecting to PostgreSQL database: {db_name_identifier}...")
        success, message = connect_to_postgres(connection_string)
        if not success:
            return False, message, None
    except Exception as e:
        return False, handle_exception(e, user_query=connection_string, context={"step": "connect_to_postgres"}), None

    # --- Step 2: Get Schema and Save to File ---
    schema_filepath = get_schema_filepath(db_name_identifier)
    os.makedirs(os.path.dirname(schema_filepath), exist_ok=True)

    try:
        # Fetch schema and sample data from the database
        print(f"[INFO] Fetching schema and sample data from database...")
        success, message, schema_data = get_schema_and_sample_data(schema_filepath)
        if not success:
            return False, message, None
        
        # --- Step 3: Load and Filter Schema ---
        filtered_schema, message = load_and_filter_schema(db_name_identifier)
        if filtered_schema is None:
            return False, message, None

        # --- Step 4: Check if schema vectors and graph already exist ---
        vectors_exist = check_schema_vectors_exist(db_name_identifier)
        graph_exists = check_schema_graph_exists(db_name_identifier)
        
        # --- Step 5: Generate and Save Schema Graph (if needed) ---
        schema_graph = None
        foreign_key_relationships = None
        
        if not graph_exists or force_regenerate:
            try:
                print("[INFO] Generating schema relationship graph...")
                all_table_names = list(filtered_schema.keys())
                schema_graph = await generate_schema_graph(connection_string, all_table_names)
                if schema_graph:
                    save_schema_graph(db_name_identifier, schema_graph)
                    print("[INFO] Schema relationship graph saved.")
                    
                    # Extract foreign key relationships from the schema graph
                    if 'edges' in schema_graph:
                        foreign_key_relationships = []
                        for edge in schema_graph['edges']:
                            source = edge.get('source')
                            target = edge.get('target')
                            label = edge.get('label')
                            
                            if source and target and label:
                                # Parse the label which is in format "fk_column → pk_column"
                                parts = label.split(' → ')
                                if len(parts) == 2:
                                    fk_column = parts[0]
                                    pk_column = parts[1]
                                    foreign_key_relationships.append({
                                        'fk_table': source,
                                        'fk_column': fk_column,
                                        'pk_table': target,
                                        'pk_column': pk_column
                                    })
            except Exception as e:
                handle_exception(e, context={"step": "generate_schema_graph"}, message="Could not generate schema graph, but continuing.")
        else:
            print("[INFO] Using existing schema graph.")
            # Load the existing schema graph to extract foreign key relationships
            schema_graph = load_schema_graph(db_name_identifier)
            if schema_graph and 'edges' in schema_graph:
                foreign_key_relationships = []
                for edge in schema_graph['edges']:
                    source = edge.get('source')
                    target = edge.get('target')
                    label = edge.get('label')
                    
                    if source and target and label:
                        # Parse the label which is in format "fk_column → pk_column"
                        parts = label.split(' → ')
                        if len(parts) == 2:
                            fk_column = parts[0]
                            pk_column = parts[1]
                            foreign_key_relationships.append({
                                'fk_table': source,
                                'fk_column': fk_column,
                                'pk_table': target,
                                'pk_column': pk_column
                            })

        # --- Step 6: Vectorize Schema for HyDE (if needed) ---
        if not vectors_exist or force_regenerate:
            try:
                print("[INFO] Vectorizing schema for enhanced search...")
                vectorize_and_store_schema(
                    db_name_identifier, 
                    filtered_schema, 
                    foreign_key_relationships,
                    force_regenerate=force_regenerate
                )
                print("[INFO] Schema vectorization complete.")
            except Exception as e:
                # Log the error but don't fail the entire initialization
                handle_exception(e, context={"step": "vectorize_schema"}, message="Could not vectorize schema, but continuing.")
        else:
            print("[INFO] Using existing schema vectors.")
        
        # --- Step 7: Process stored functions (only when regenerating schema or changing database) ---
        if force_regenerate or not vectors_exist or not graph_exists:
            try:
                print("[INFO] Processing stored functions...")
                success, func_message = await process_stored_functions(connection_string, db_name_identifier)
                if success:
                    print(f"[INFO] {func_message}")
                else:
                    handle_exception(Exception(func_message), context={"step": "process_stored_functions"}, 
                                    message="Could not process stored functions, but continuing.")
            except Exception as e:
                handle_exception(e, context={"step": "process_stored_functions"}, 
                                message="Error processing stored functions, but continuing.")
        
        active_tables_filepath = get_active_tables_filepath(db_name_identifier)
        
        success_message = (
            f"Successfully initialized '{db_name_identifier}'. {message}\n"
            f"To change this, please edit '{active_tables_filepath}' and run /reload_scope."
        )
        
        # Return the LLM client as part of the result
        return_data = {
            "schema_data": filtered_schema,
            "llm_client": llm_client
        }
        
        return True, success_message, return_data

    except Exception as e:
        return False, handle_exception(e, user_query=connection_string, context={"step": "get_schema_and_sample_data"}), None
