import networkx as nx
from itertools import combinations

def find_join_path(required_tables, schema_graph):
    """
    Finds the shortest path of joins to connect a list of required tables.

    Args:
        required_tables (list): A list of table names that need to be included in the query.
        schema_graph (dict): A dictionary representing the database schema with nodes (tables) and edges (relationships).

    Returns:
        list: An ordered list of SQL JOIN clauses, or an empty list if no path is found.
    """
    if not required_tables or len(required_tables) < 2:
        return []

    G = nx.Graph()
    for table in schema_graph['nodes']:
        G.add_node(table['name'])

    for edge in schema_graph['edges']:
        G.add_edge(edge['source'], edge['target'], relationship=edge['relationship'])

    # Find all pairs of tables that need to be connected
    table_pairs = list(combinations(required_tables, 2))

    all_paths = []
    for source, target in table_pairs:
        try:
            path = nx.shortest_path(G, source=source, target=target)
            all_paths.append(path)
        except nx.NetworkXNoPath:
            # If there's no path between any two required tables, we can't form a valid query
            return []

    # Combine all paths into a single set of nodes to visit
    nodes_in_path = set()
    for path in all_paths:
        for node in path:
            nodes_in_path.add(node)

    # Create a subgraph with only the nodes needed for the final path
    subgraph = G.subgraph(nodes_in_path)

    # Use a minimum spanning tree to find the most efficient path connecting all nodes
    mst = nx.minimum_spanning_tree(subgraph)

    # Convert the MST edges to JOIN clauses
    join_clauses = []
    for u, v, data in mst.edges(data=True):
        relationship = data['relationship']
        join_clauses.append(f"{u} JOIN {v} ON {relationship}")

    return join_clauses
