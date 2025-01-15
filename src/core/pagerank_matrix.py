"""
Core PageRank algorithm implementation optimized for sparse matrices.

This module provides an efficient implementation of PageRank algorithm variants using
sparse matrix operations. It includes both the standard PageRank algorithm and a temporal variant.
The temporal variant was developed as part of an Abiturarbeit research project (2022/2023),
which extends PageRank to incorporate temporal node importance.

The implementation is particularly suitable for:
- Large-scale web graphs
- Any directed graph represented as a sparse adjacency matrix
"""

# Standard Library Imports
from dataclasses import dataclass
from typing import Optional

# Third-Party Imports
import numpy as np
from scipy.sparse import csr_matrix, spmatrix


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
    max_iterations: int = 200
    convergence_threshold: float = 1e-10
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
    """Base class for PageRank calculations providing common utilities."""

    def __init__(self, matrix: csr_matrix, config: Optional[PageRankConfig] = None):
        """
        Initialize PageRank calculator with transition matrix.

        Args:
            matrix: Transition probability matrix as CSR sparse matrix
            config: Optional configuration parameters

        Raises:
            TypeError: If matrix is not a csr_matrix
            ValueError: If matrix is invalid
        """
        if not isinstance(matrix, csr_matrix):
            raise TypeError("Matrix must be a scipy.sparse.csr_matrix")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")

        # Additional validation
        if np.any(np.isnan(matrix.data)) or np.any(np.isinf(matrix.data)):
            raise ValueError("Matrix contains NaN or infinite values")
        if np.any(matrix.data < 0):
            raise ValueError("Matrix contains negative values")

        self.matrix = matrix
        self.config = config or PageRankConfig()
        self.n = matrix.shape[0]
        self._rank_vector: Optional[np.ndarray] = None

    def _get_dangling_nodes(self) -> np.ndarray:
        """
        Returns a vector indicating nodes with no outgoing edges.

        Returns:
            numpy array where 1.0 indicates a dangling node
        """
        row_sums = np.array(self.matrix.sum(axis=1)).flatten()
        return (row_sums == 0).astype(float)

    def _initialize_rank_vector(self) -> np.ndarray:
        """Initializes the rank vector with uniform probabilities."""
        return np.ones(self.n) / self.n

    def _detect_oscillation(self, vectors: list) -> bool:
        """
        Detect if PageRank vectors are oscillating.

        Args:
            vectors: List of recent PageRank vectors

        Returns:
            bool: True if oscillation is detected
        """
        if len(vectors) < 4:
            return False

        # Check if the pattern repeats
        delta1 = np.abs(vectors[-1] - vectors[-3]).sum()
        delta2 = np.abs(vectors[-2] - vectors[-4]).sum()

        # If both deltas are small, we have an oscillation
        return (delta1 < self.config.convergence_threshold * 10 and
                delta2 < self.config.convergence_threshold * 10)


class StandardPageRank(PageRankBase):
    """Implementation of the standard PageRank algorithm"""

    def calculate(self) -> np.ndarray:
        """
        Calculates PageRank values using the standard iterative formula.

        Returns:
            numpy array of final PageRank scores

        Raises:
            ValueError: If calculation fails to converge
        """
        self._rank_vector = self._initialize_rank_vector()
        dangling_nodes = self._get_dangling_nodes()

        d = self.config.damping_factor
        uniform_vector = np.ones(self.n) / self.n

        # Store previous vectors for oscillation detection
        previous_vectors = []
        oscillation_window = 4  # Check for oscillation patterns
        
        for iteration in range(self.config.max_iterations):
            prev_vector = self._rank_vector.copy()

            # Handle dangling nodes
            dangling_sum = np.sum(self._rank_vector * dangling_nodes)
            dangling_contribution = dangling_sum * uniform_vector

            # Main PageRank calculation with proper normalization
            self._rank_vector = (
                    d * (self.matrix.T.dot(self._rank_vector) + dangling_contribution) +
                    (1 - d) * uniform_vector
            )

            # Ensure normalization
            self._rank_vector = self._rank_vector / np.sum(self._rank_vector)

            # Check convergence
            delta = np.abs(self._rank_vector - prev_vector).sum()

            # Store vector for oscillation detection
            previous_vectors.append(self._rank_vector.copy())
            if len(previous_vectors) > oscillation_window:
                previous_vectors.pop(0)

            # Detect oscillation
            if len(previous_vectors) == oscillation_window:
                if self._detect_oscillation(previous_vectors):
                    # Take the average of oscillating vectors as final result
                    self._rank_vector = np.mean(previous_vectors, axis=0)
                    return self._rank_vector / np.sum(self._rank_vector)

            if delta < self.config.convergence_threshold:
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
    for time-based importance of nodes in the network. The matrix-based implementation
    provides efficient computation for large-scale temporal graphs.
    """

    def calculate(self, temporal_vector: np.ndarray) -> np.ndarray:
        """
        Calculates PageRank values using the temporal formula.

        Args:
            temporal_vector: Vector of temporal importance scores

        Returns:
            numpy array of final PageRank scores
        
        Raises:
            ValueError: If temporal vector is invalid or calculation fails to converge
        """
        if temporal_vector.shape != (self.n,):
            raise ValueError(f"Temporal vector must have shape ({self.n},)")

        self._rank_vector = self._initialize_rank_vector()
        dangling_nodes = self._get_dangling_nodes()

        d = self.config.damping_factor
        omega = self.config.omega
        uniform_vector = np.ones(self.n) / self.n

        # Store previous vectors for oscillation detection
        previous_vectors = []
        oscillation_window = 4  # Check for oscillation patterns

        for iteration in range(self.config.max_iterations):
            prev_vector = self._rank_vector.copy()

            # Handle dangling nodes
            dangling_sum = np.sum(self._rank_vector * dangling_nodes)
            dangling_contribution = dangling_sum * uniform_vector

            # Standard component
            standard_component = (
                    d * (self.matrix.T.dot(self._rank_vector) + dangling_contribution) +
                    (1 - d) * uniform_vector
            )

            # Temporal component
            temporal_component = omega * temporal_vector

            # Combined calculation
            self._rank_vector = (1 - omega) * standard_component + temporal_component

            # Ensure normalization
            self._rank_vector = self._rank_vector / np.sum(self._rank_vector)

            # Check convergence
            delta = np.abs(self._rank_vector - prev_vector).sum()

            # Store vector for oscillation detection
            previous_vectors.append(self._rank_vector.copy())
            if len(previous_vectors) > oscillation_window:
                previous_vectors.pop(0)

            if len(previous_vectors) == oscillation_window:
                if self._detect_oscillation(previous_vectors):
                    self._rank_vector = np.mean(previous_vectors, axis=0)
                    return self._rank_vector / np.sum(self._rank_vector)

            if delta < self.config.convergence_threshold:
                return self._rank_vector

        raise ValueError(
            f"Temporal PageRank failed to converge after {self.config.max_iterations} iterations."
        )


class PageRank:
    """Main interface for PageRank calculations on sparse matrices."""

    def __init__(self, matrix: csr_matrix, config: Optional[PageRankConfig] = None):
        """
        Initialize PageRank calculator.

        Args:
            matrix: Transition probability matrix as CSR sparse matrix
            config: Optional configuration parameters
        """
        # Verify row sum normalization (allowing for small numerical errors)
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        non_zero_rows = row_sums != 0
        if not np.allclose(row_sums[non_zero_rows], 1.0, rtol=1e-05, atol=1e-08):
            raise ValueError("Matrix rows are not properly normalized")

        self.matrix = matrix
        self.config = config or PageRankConfig()

    def calculate(self, temporal_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate PageRank scores using either standard or temporal variant.

        Args:
            temporal_vector: Optional temporal importance scores for temporal variant

        Returns:
            numpy array of PageRank scores

        Raises:
            ValueError: If temporal_vector is required but not provided
        """
        if self.config.use_temporal:
            if temporal_vector is None:
                raise ValueError("Temporal vector required when use_temporal=True")
            calculator = TemporalPageRank(self.matrix, self.config)
            return calculator.calculate(temporal_vector)
        else:
            calculator = StandardPageRank(self.matrix, self.config)
            return calculator.calculate()


if __name__ == "__main__":
    # Example usage
    # Create a simple test matrix

    row_indices = [0, 1, 1, 1, 2, 2, 3]
    col_indices = [2, 0, 2, 3, 0, 3, 2]
    probabilities = [1.0, 1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 1.0]

    # Create 4x4 sparse matrix
    matrix = csr_matrix((probabilities, (row_indices, col_indices)), shape=(4, 4))

    print("CSR Matrix representation:")
    print(matrix.toarray())

    config = PageRankConfig(use_temporal=False)
    pr = PageRank(matrix, config)
    scores = pr.calculate()
    print("Standard PageRank scores:", scores)

    data = np.array([0.5, 0.5, 1.0, 0.5, 0.5])
    rows = np.array([0, 0, 1, 2, 2])
    cols = np.array([1, 2, 0, 0, 1])
    test_matrix = csr_matrix((data, (rows, cols)), shape=(3, 3))

    # Standard PageRank
    config = PageRankConfig(use_temporal=False)
    pr = PageRank(test_matrix, config)
    scores = pr.calculate()
    print("Standard PageRank scores:", scores)

    # Temporal PageRank
    config = PageRankConfig(use_temporal=True, omega=0.5)
    pr = PageRank(test_matrix, config)
    temporal_vector = np.array([0.1, 0.3, 0.3])  # Example temporal importance
    scores = pr.calculate(temporal_vector)
    print("Temporal PageRank scores:", scores)
