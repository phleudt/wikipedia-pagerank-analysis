"""

"""

# Standard Library Imports
from typing import Union, Optional

# Third-Party Imports
from scipy.sparse import csr_matrix

# Local Imports
from core.graph import Graph
from core.pagerank_graph import PageRankGraph, PageRankConfig
from core.pagerank_matrix import PageRank


def create_pagerank(graph_or_matrix: Union[Graph, csr_matrix], config: Optional[PageRankConfig] = None) -> Union[PageRankGraph, PageRank]:
    if isinstance(graph_or_matrix, Graph):
        return PageRankGraph(graph_or_matrix, config)
    elif isinstance(graph_or_matrix, csr_matrix):
        return PageRank(graph_or_matrix, config)
    else:
        raise ValueError("Input must be either a Graph object or a csr_matrix.")