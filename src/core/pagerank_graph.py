"""
Core PageRank algorithm implementation for graph-based data structures.

This module implements both the standard PageRank algorithm and a temporal variant using a graph-based approach.
The temporal variant was developed as part of an Abiturarbeit research project (2022/2023),
which extends PageRank to incorporate temporal node importance.
"""

# Standard Library Imports
from dataclasses import dataclass
from typing import Optional

# Third-Party Imports
import numpy as np
from scipy.sparse import csr_matrix, spmatrix

# Local Imports
from core.graph import Graph


@dataclass
class PageRankConfig:
    """
    Configuration parameters for PageRank calculation.

    Attributes:
        damping_factor: Probability of following links vs. random jump (default: 0.85)
        max_iterations: Maximum number of iterations for convergence (default: 100)
        convergence_threshold: Minimum change between iterations to continue (default: 1e-6)
        use_temporal: Whether to use temporal weighting (default: False)
        omega: Weight factor for temporal component (default: 0.25)

    Raises:
        ValueError: If parameters are outside their valid ranges
    """
    damping_factor: float = 0.85
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    use_temporal: bool = False
    omega: float = 0.25

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if not 0 <= self.damping_factor <= 1:
            raise ValueError("Damping factor must be between 0 and 1")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.convergence_threshold <= 0:
            raise ValueError("Convergence threshold must be positive")
        if not 0 <= self.omega <= 1:
            raise ValueError("Temporal weight (omega) must be between 0 and 1")


class PageRankBase:
    """
    Base class for PageRank calculations providing common utilities.

    This class handles the common functionality between standard and temporal
    PageRank variants, including transition matrix creation and initialization.
    """

    def __init__(self, graph: Graph, config: Optional[PageRankConfig] = None):
        """
        Initialize PageRank calculator.

        Args:
            graph: Input graph structure
            config: Optional configuration parameters

        Raises:
            ValueError: If graph is empty or invalid
        """
        if not graph or graph.node_count == 0:
            raise ValueError("Graph cannot be null or empty")

        self.graph = graph
        self.config = config or PageRankConfig()
        self.n = graph.node_count
        self._transition_matrix: Optional[spmatrix] = None
        self._rank_vector: Optional[np.ndarray] = None

    def _create_transition_matrix(self) -> spmatrix:
        """Creates the transition probability matrix from the graph."""
        rows, cols, data = [], [], []

        for node in self.graph.iter_nodes():
            out_edges = self.graph.get_outgoing_edges(node)
            if out_edges:
                prob = 1.0 / len(out_edges)
                for target in out_edges:
                    rows.append(node)
                    cols.append(target)
                    data.append(prob)

        return csr_matrix((data, (rows, cols)), shape=(self.n, self.n))

    def _get_dangling_nodes(self) -> np.ndarray:
        """Returns a vector indicating nodes with no outgoing edges."""
        if self._transition_matrix is None:
            raise ValueError("Transition matrix not initialized")
        row_sums = self._transition_matrix.sum(axis=1).A1
        return (row_sums == 0).astype(float)

    def _initialize_rank_vector(self) -> np.ndarray:
        """Initializes the rank vector with uniform probabilities."""
        return np.ones(self.n) / self.n


class StandardPageRank(PageRankBase):
    """Implementation of the standard PageRank algorithm."""

    def calculate(self) -> np.ndarray:
        """
        Calculates PageRank values using the standard iterative formula.

        Returns:
            numpy array of final PageRank scores

        Raises:
            ValueError: If calculation fails to converge
        """
        self._transition_matrix = self._create_transition_matrix()
        self._rank_vector = self._initialize_rank_vector()
        dangling_nodes = self._get_dangling_nodes()

        d = self.config.damping_factor
        uniform_vector = np.ones(self.n) / self.n

        for _ in range(self.config.max_iterations):
            prev_vector = self._rank_vector.copy()

            # Handle dangling nodes
            dangling_factor = (1 - d) + d * np.sum(self._rank_vector * dangling_nodes)

            # Main PageRank calculation
            self._rank_vector = (
                    d * self._transition_matrix.T.dot(self._rank_vector) +
                    dangling_factor * uniform_vector
            )

            # Ensure normalization
            self._rank_vector = self._rank_vector / np.sum(self._rank_vector)

            # Check convergence
            if np.abs(self._rank_vector - prev_vector).sum() < self.config.convergence_threshold:
                return self._rank_vector

        raise ValueError(
            f"PageRank failed to converge after {self.config.max_iterations} iterations. "
            f"Consider increasing max_iterations or adjusting convergence_threshold."
        )



class TemporalPageRank(PageRankBase):
    """
    Implementation of a temporal PageRank variant.
    
    This implementation is based on original research from an Abiturarbeit (2022/2023),
    extending the standard PageRank algorithm with temporal weighting factors to account
    for time-based importance of nodes in the network.
    """

    def calculate(self, temporal_vector: np.ndarray) -> np.ndarray:
        """
        Calculates PageRank values using my temporal formula.

        Args:
            temporal_vector: Vector of temporal importance scores

        Returns:
            numpy array of final PageRank scores
            
        Raises:
            ValueError: If temporal vector is invalid or calculation fails to converge
        """
        # Validate temporal vector
        if temporal_vector.shape != (self.n,):
            raise ValueError(f"Temporal vector must have shape ({self.n},)")
        if not np.isclose(np.sum(temporal_vector), 1.0, rtol=1e-5):
            raise ValueError("Temporal vector must sum to 1")
        if np.any(temporal_vector < 0):
            raise ValueError("Temporal vector cannot contain negative values")

        self._transition_matrix = self._create_transition_matrix()
        self._rank_vector = self._initialize_rank_vector()
        dangling_nodes = self._get_dangling_nodes()

        d = self.config.damping_factor
        omega = self.config.omega
        uniform_vector = np.ones(self.n) / self.n

        for _ in range(self.config.max_iterations):
            prev_vector = self._rank_vector.copy()

            # Handle dangling nodes
            dangling_factor = d * np.sum(self._rank_vector * dangling_nodes)

            # Standard component
            standard_component = (1 - omega) * (
                    d * self._transition_matrix.T.dot(self._rank_vector) +
                    (dangling_factor + (1 - d)) * uniform_vector
            )

            # Temporal component
            temporal_component = omega * temporal_vector

            # Combined calculation
            self._rank_vector = standard_component + temporal_component

            # Ensure normalization
            self._rank_vector = self._rank_vector / np.sum(self._rank_vector)

            # Check convergence
            if np.abs(self._rank_vector - prev_vector).sum() < self.config.convergence_threshold:
                return self._rank_vector

        raise ValueError(
            f"Temporal PageRank failed to converge after {self.config.max_iterations} iterations. "
            f"Consider increasing max_iterations or adjusting convergence_threshold."
        )


class PageRankGraph:
    """Main interface for PageRank calculations."""

    def __init__(self, graph: Graph, config: Optional[PageRankConfig] = None):
        """
        Initialize PageRank calculator.

        Args:
            graph: Input graph structure
            config: Optional configuration parameters
        """
        self.graph = graph
        self.config = config or PageRankConfig()

    def calculate(self, temporal_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate PageRank scores using either standard or temporal variant.

        Args:
            temporal_vector: Optional temporal importance scores for temporal variant

        Returns:
            numpy array of PageRank scores

        Raises:
            ValueError: If parameters are invalid or calculation fails
        """
        if self.config.use_temporal:
            if temporal_vector is None:
                raise ValueError("Temporal vector required when use_temporal=True")
            calculator = TemporalPageRank(self.graph, self.config)
            return calculator.calculate(temporal_vector)
        else:
            calculator = StandardPageRank(self.graph, self.config)
            return calculator.calculate()


if __name__ == "__main__":
    # Example usage
    # Standard PageRank
    graph_test = Graph.from_edge_list([[2], [0, 2, 3], [0, 3], [2]])  # BLL Graph
    config_test = PageRankConfig(use_temporal=False)
    pr_test = PageRankGraph(graph_test, config_test)
    scores = pr_test.calculate()
    print("Standard PageRank scores:", scores)
    
    # Temporal PageRank
    config_test = PageRankConfig(use_temporal=True, omega=0.3)
    pr_test = PageRankGraph(graph_test, config_test)
    temporal_vector_test = np.array([0.4, 0.3, 0.2, 0.1])  # Example temporal importance
    scores = pr_test.calculate(temporal_vector_test)
    print("Temporal PageRank scores:", scores)