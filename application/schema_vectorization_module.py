import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import os
import re
import sys
from datetime import datetime

# Try to import modules with proper fallback
try:
    # First try relative import
    from . import vector_store_module
    from . import memory_module
    from .error_handler_module import handle_exception
except ImportError:
    try:
        # Then try absolute import
        import vector_store_module
        import memory_module
        from error_handler_module import handle_exception
    except ImportError:
        try:
            # Try importing from parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application import vector_store_module
            from application import memory_module
            from application.error_handler_module import handle_exception
        except ImportError as e:
            print(f"Error importing modules: {e}", file=sys.stderr)
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
            
            # Define fallback modules if imports fail
            vector_store_module = None
            memory_module = None

def save_schema_backup(db_name_identifier: str, table_descriptions: List[str]):
    """
    Saves table descriptions to a backup text file.
    
    Args:
        db_name_identifier: The database identifier
        table_descriptions: List of table description strings
    """
    try:
        backup_filepath = memory_module.get_schema_backup_filepath(db_name_identifier)
        
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
    backup_filepath = memory_module.get_schema_backup_filepath(db_name_identifier)
    return os.path.exists(backup_filepath)

def get_schema_table_name(db_name_identifier) -> str:
    """Generates a valid LanceDB table name for schema vectors."""
    # Convert to string first to handle integer identifiers
    db_name_str = str(db_name_identifier)
    sanitized_name = "".join(c if c.isalnum() or c == '_' else '_' for c in db_name_str)
    return f"{sanitized_name}_schema_vectors"

def _get_or_create_schema_table(db_name_identifier: str) -> lancedb.table.Table:
    """
    Retrieves or creates a LanceDB table for storing schema vector embeddings.
    """
    conn = vector_store_module._get_lancedb_connection()
    table_name = get_schema_table_name(db_name_identifier)

    if table_name in conn.table_names():
        return conn.open_table(table_name)
    else:
        model = vector_store_module._initialize_embedding_model()
        dim = model.get_sentence_embedding_dimension()
        
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
            pa.field("description", pa.string()),
            pa.field("type", pa.string()), # e.g., 'table' or 'column'
            pa.field("table_name", pa.string()),
            pa.field("column_name", pa.string(), nullable=True) # Null for table descriptions
        ])
        return conn.create_table(table_name, schema=schema, mode="overwrite")

def _log_schema_chunk(db_name_identifier: str, chunk_data: Dict[str, Any]):
    """
    Logs a schema chunk to the schema_chunk_log.json file.
    
    Args:
        db_name_identifier: The database identifier
        chunk_data: Data about the chunk to log
    """
    try:
        # Ensure logs directory exists
        memory_base_path = memory_module.get_memory_base_path()
        logs_dir = os.path.join(memory_base_path, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare log file path
        log_file_path = os.path.join(logs_dir, 'schema_chunk_log.json')
        
        # Create log entry with timestamp
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "db_name_identifier": db_name_identifier,
            **chunk_data
        }
        
        # Append to log file
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    except Exception as e:
        handle_exception(e, user_query=f"log_schema_chunk for {db_name_identifier}", 
                        context={"chunk_data": chunk_data})

def _log_skipped_table(db_name_identifier: str, table_name: str, reason: str):
    """
    Logs information about a table that was skipped during vectorization.
    
    Args:
        db_name_identifier: The database identifier
        table_name: The name of the skipped table
        reason: The reason why the table was skipped
    """
    _log_schema_chunk(db_name_identifier, {
        "event": "table_skipped",
        "table_name": table_name,
        "reason": reason
    })

def _parse_create_table_sql(sql_statement: str) -> List[Dict[str, str]]:
    """
    Parses a SQL CREATE TABLE statement to extract column information.
    
    Args:
        sql_statement: The SQL CREATE TABLE statement
        
    Returns:
        A list of dictionaries, each containing column name, type, and description
    """
    try:
        # Extract the part between parentheses
        match = re.search(r'\(\s*(.*?)\s*\);', sql_statement, re.DOTALL)
        if not match:
            return []
            
        columns_text = match.group(1)
        
        # Split by commas, but not commas inside parentheses (for array types, etc.)
        # This regex handles nested parentheses
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
    
    Args:
        sample_data: A list of dictionaries containing sample data
        
    Returns:
        A list of dictionaries, each containing column name and type
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
    
    Args:
        db_name_identifier: The database identifier
        schema_info: Dictionary containing schema information
        foreign_key_relationships: Optional list of foreign key relationships from get_foreign_key_relationships()
        force_regenerate: If True, regenerate schema backup even if it exists
    """
    try:
        table = _get_or_create_schema_table(db_name_identifier)
        model = vector_store_module._initialize_embedding_model()
        
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
            
            # 2. If no columns key, try to parse from schema SQL (prioritized for data types)
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
                _log_skipped_table(db_name_identifier, table_name, f"No columns found from any source")
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
                col_description = column.get('description', '') # Assuming there might be a description
                
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
            
            # Add sample data vectorization if available: embed up to N sample rows (one document per row)
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

def search_schema_vectors(db_name_identifier: str, query_embedding: List[float], limit: int = 10) -> List[str]:
    """
    Searches the schema vector table for the most relevant schema descriptions.
    """
    try:
        conn = vector_store_module._get_lancedb_connection()
        table_name = get_schema_table_name(db_name_identifier)

        if table_name not in conn.table_names():
            return []

        table = conn.open_table(table_name)
        if len(table) == 0:
            return []

        search_results = table.search(query_embedding).limit(limit).to_df()
        
        # Return the 'description' column of the results
        return search_results['description'].tolist()

    except Exception as e:
        handle_exception(e, user_query=f"search_schema_vectors for {db_name_identifier}")
        return []
