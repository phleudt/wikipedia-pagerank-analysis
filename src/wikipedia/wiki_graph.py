"""
A specialized graph implementation for Wikipedia PageRank calculations.

This script implements an efficient graph processing system for computing PageRank scores of Wikipedia pages,
with support for both standard and temporal PageRank algorithms.
The implementation uses sparse matrices for memory efficiency and supports incremental processing of large graphs.

Notes:
    - Large graphs may require significant memory and processing time
    - Matrix saving is optional but recommended for repeated calculations
    - Temporal PageRank requires valid timestamp file in the data directory
"""

# Standard Library Imports
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Third-Party Imports
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

# Local Imports
from core.pagerank_matrix import PageRank, PageRankConfig


@dataclass
class WikiGraphConfig:
    """Configuration parameters for WikiGraph processing."""
    data_dir: Path # Base data directory
    input_file: str = "wiki_sorted_page_reference_graph.txt" # Raw input file
    num_nodes: Optional[int] = None # Calculated during graph initialization
    save_matrix: bool = False
    use_temporal: bool = False
    load_existing_pagerank: bool = True
    damping_factor: float = 0.85
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    omega: float = 0.25 # Temporal weight factor
    timestamp_file: Optional[str] = None # File containing page timestamps

    def __post_init__(self):
        """Set up directory structure and file paths after initialization."""
        # Ensure data directories exist
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Set up file paths
        self.input_path = self.processed_dir / self.input_file
        self.matrix_file = self.processed_dir / "wiki_matrix.npz"
        self.pagerank_file = self.processed_dir / "wiki_pagerank.npz"
        self.temporal_pagerank_file = self.processed_dir / "wiki_temporal_pagerank.npz"

        if self.timestamp_file:
            self.timestamp_path = self.processed_dir / self.timestamp_file
        else:
            self.timestamp_path = None

    @property
    def paths_exist(self) -> bool:
        """Check if required input files exist."""
        return self.input_path.exists()


class WikiGraph:
    """
    Enhanced Wikipedia graph processor with support for both temporal and non-temporal PageRank.
    Handles large-scale graph operations efficiently with sparse matrix support.
    """

    def __init__(self, config: WikiGraphConfig):
        """
        Initialize WikiGraph with configuration parameters.

        Args:
            config: WikiGraphConfig instance containing processing parameters
        """
        self.config = config
        self.matrix: Optional[csr_matrix] = None
        self.pagerank_vector: Optional[np.ndarray] = None
        self.temporal_vector: Optional[np.ndarray] = None

        # Warn about memory usage when saving matrix
        if self.config.save_matrix:
            print(
                "WARNING: `save_matrix=True` can result in high memory usage, especially "
                "for large graphs. Ensure sufficient system memory is available."
            )

        # Verify input files exist
        if not self.config.paths_exist:
            raise FileNotFoundError(
                f"Input file not found at {self.config.input_path}. "
            )

        self._initialize_matrix()

        # Load temporal data if configured
        if self.config.use_temporal and self.config.timestamp_path:
            self._initialize_temporal_vector()

        # Load existing PageRank if configured
        if self.config.load_existing_pagerank:
            try:
                self.load_pagerank()
            except FileNotFoundError:
                print(f"Warning: PageRank file {self.config.pagerank_file} not found. Will need to copmute PageRank.")

    def _initialize_matrix(self) -> None:
        """Initialize the graph structure based on configuration."""
        if self.config.save_matrix:
            self._build_matrix_incremental()
            self._save_matrix()
        else:
            self._load_matrix()

    def _build_matrix_incremental(self) -> None:
        """Build the transition matrix incrementally"""
        chunk_size = 100_000

        self.config.num_nodes = self._count_nodes()
        out_degrees = self._calculate_out_degrees()
        self.matrix = self._build_normalized_matrix(out_degrees, chunk_size)

        print(f"Final matrix shape: {self.matrix.shape}")

    def _count_nodes(self) -> int:
        """Count the total number of nodes from the input file."""
        print("Counting nodes...")
        with open(self.config.input_path, 'r', encoding='utf-8') as file:
            num_nodes = sum(1 for _ in file)
        print(f"Number of unique nodes: {num_nodes}\n")
        return num_nodes

    def _calculate_out_degrees(self) -> np.ndarray:
        """Calculate the out-degrees for all nodes in a single pass."""
        print("Calculating out-degrees...\n")
        out_degrees = np.zeros(self.config.num_nodes, dtype=np.float64)
        with open(self.config.input_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = [int(part) for part in line.strip().split("ÿ~") if part.isdigit()]
                if len(parts) >= 2:
                    source = parts[0]
                    targets = set(parts[1:])  # Remove duplicates
                    out_degrees[source] = len(targets)
        return out_degrees

    def _build_normalized_matrix(self, out_degrees: np.ndarray, chunk_size: int) -> csr_matrix:
        """Construct the normalized transition matrix using precomputed out-degrees."""
        print("Building normalized transition matrix...")
        matrix = None

        with open(self.config.input_path, 'r', encoding='utf-8') as file:
            for chunk_start in range(0, self.config.num_nodes, chunk_size):
                row_indices, col_indices, data = [], [], []

                for _ in range(chunk_size):
                    line = file.readline()
                    if not line:
                        break

                    parts = [int(part) for part in line.strip().split("ÿ~") if part.isdigit()]
                    if len(parts) < 2:
                        continue

                    source = parts[0]
                    targets = set(parts[1:])

                    if out_degrees[source] > 0:
                        prob = 1.0 / out_degrees[source]
                        for target in targets:
                            row_indices.append(source)
                            col_indices.append(target)
                            data.append(prob)

                chunk_matrix = csr_matrix(
                    (data, (row_indices, col_indices)),
                    shape=(self.config.num_nodes, self.config.num_nodes)
                )

                matrix = chunk_matrix if matrix is None else matrix + chunk_matrix
                print(f"Processed nodes {chunk_start} to {min(chunk_start + chunk_size, self.config.num_nodes)}")

        return matrix

    def _save_matrix(self) -> None:
        """Save the transition matrix to file."""
        print(f"Saving matrix to {self.config.matrix_file}...")
        save_npz(self.config.matrix_file, self.matrix)
        print("Matrix saved successfully.\n")

    def _load_matrix(self) -> None:
        """Load the transition matrix from file."""
        print(f"Loading matrix from {self.config.matrix_file}...")
        try:
            self.matrix = load_npz(self.config.matrix_file)
            self.config.num_nodes = self.matrix.shape[0]
            print(f"Matrix loaded successfully with {self.config.num_nodes} nodes.\n")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Matrix file not found at {self.config.matrix_file}. "
                f"Please run with save_matrix=True first."
            )

    def _initialize_temporal_vector(self) -> None:
        """Initialize temporal importance vector from timestamps."""
        if not self.config.timestamp_path.exists():
            raise FileNotFoundError(f"Timestamp file not found: {self.config.timestamp_path}")

        timestamps = []
        with open(self.config.timestamp_path, 'r', encoding='utf-8') as f:
            for line in f:
                time = datetime.fromisoformat(line.strip()).astimezone(timezone.utc)
                timestamps.append(time)

        vector = np.zeros((self.config.num_nodes,))

        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)

            if min_time == max_time:
                # All timestamps are the same, use uniform distribution
                vector.fill(1.0 / self.config.num_nodes)
            else:
                # Calculate temporal weights
                for i, time in enumerate(timestamps):
                    vector[i] = (time - min_time).total_seconds()

                # Normalize to [0, 1] range
                vector = (vector - vector.min()) / (vector.max() - vector.min())

                # Ensure no zero weights by adding small epsilon
                epsilon = 1e-10
                vector = vector + epsilon

                # Final normalization
                vector = vector / vector.sum()
        else:
            # No valid timestamps, use uniform distribution
            vector.fill(1.0 / self.config.num_nodes)

        self.temporal_vector = vector

    def compute_pagerank(self,
                         force_compute: bool = False,
                         save_results: bool = True) -> np.ndarray:
        """
        Compute PageRank scores for the Wikipedia graph.

        Args:
            force_compute: Whether to force recomputation even if already loaded
            save_results: Whether to save the computed PageRank vector

        Returns:
            Array of PageRank scores
        """
        if not force_compute and self.pagerank_vector is not None:
            return self.pagerank_vector

        print("Starting PageRank computation...")

        # Configure PageRank

        pr_config = PageRankConfig(
            damping_factor=self.config.damping_factor,
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold,
            use_temporal=self.config.use_temporal,
            omega=self.config.omega
        )

        # Initialize PageRank calculator
        pr = PageRank(self.matrix, pr_config)

        # Calculate PageRank with or without temporal vector
        self.pagerank_vector = pr.calculate(self.temporal_vector if self.config.use_temporal else None)

        if save_results:
            self._save_pagerank()

        print("PageRank computation completed.")
        return self.pagerank_vector

    def _save_pagerank(self) -> None:
        """Save PageRank vector to appropriate file based on type."""
        file_path = (
            self.config.temporal_pagerank_file
            if self.config.use_temporal
            else self.config.pagerank_file
        )

        print(f"Saving PageRank vector to {file_path}...")
        np.savez(file_path, self.pagerank_vector)
        print("PageRank vector saved successfully.\n")

    def load_pagerank(self) -> np.ndarray:
        """Load previously computed PageRank vector."""
        file_path = (
            self.config.temporal_pagerank_file
            if self.config.use_temporal
            else self.config.pagerank_file
        )

        print(f"Loading PageRank vector from {file_path}...")
        try:
            data = np.load(file_path)
            self.pagerank_vector = data['arr_0']
            print("PageRank vector loaded successfully.")
            return self.pagerank_vector
        except FileNotFoundError:
            raise FileNotFoundError(
                f"PageRank file not found at {file_path}. "
                f"Please compute PageRank first using compute_pagerank()."
            )

    def get_pagerank_vector(self, force_compute: bool = False) -> np.ndarray:
        """
        Get the PageRank vector, loading from file or computing if necessary.

        Args:
            force_compute: If True, recompute PageRank even if already loaded

        Returns:
            Array of PageRank scores
        """
        if force_compute or self.pagerank_vector is None:
            return self.compute_pagerank(force_compute=force_compute)
        return self.pagerank_vector

    def get_top_pages(self, n: int = 10) -> list[tuple[int, float]]:
        """
        Get the top N pages by PageRank score.

        Args:
            n: Number of top pages to return

        Returns:
            List of (page_id, score) tuples
        """
        if self.pagerank_vector is None:
            raise ValueError("PageRank vector not computed or loaded yet")

        top_indices = np.argsort(self.pagerank_vector)[-n:][::-1]
        return [(idx, self.pagerank_vector[idx]) for idx in top_indices]


if __name__ == "__main__":
    # Example usage

    # Setup configuration with proper paths
    config = WikiGraphConfig(
        data_dir=Path("../../data"),  # Base data directory
        save_matrix=True  # Will build and save the matrix
    )

    # First time processing
    wiki = WikiGraph(config)
    pr_scores = wiki.compute_pagerank(save_results=True)

    # Later usage - load existing data
    config = WikiGraphConfig(
        data_dir=Path("../../data"),
        save_matrix=False,
        load_existing_pagerank=True
    )
    wiki = WikiGraph(config)

    # Get top pages using loaded PageRank
    wiki.load_pagerank()
    top_pages = wiki.get_top_pages(10)
    print(top_pages)
    print("\nTop 10 pages by PageRank:")
    for idx, score in top_pages:
        print(f"Page ID: {idx}, Score: {score:.6f}")

    config = WikiGraphConfig(
        data_dir=Path("../../data"),
        save_matrix=False,
        load_existing_pagerank=True,
        use_temporal=True,
        timestamp_file="wiki_page_timestamps.txt"
    )
    wiki = WikiGraph(config)
    print(wiki.matrix.shape[0], wiki.matrix.shape[1])

    print("\nMatrix Structure Debug:")
    print(f"Shape: {wiki.matrix.shape}")
    print(f"Total non-zero elements: {wiki.matrix.nnz}")
    print("\nRow by row analysis:")

    # For each row
    for i in range(wiki.matrix.shape[0]):
        # Get the indices and data for this row
        row_start = wiki.matrix.indptr[i]
        row_end = wiki.matrix.indptr[i + 1]
        col_indices = wiki.matrix.indices[row_start:row_end]
        col_data = wiki.matrix.data[row_start:row_end]

        if len(col_indices) > 0:  # Only print rows with non-zero entries
            print(f"Row {i} -> {sorted(col_indices.tolist())}")

        # Print progress every 10000 rows
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} rows...")

    wiki.compute_pagerank()
    wiki.load_pagerank()
    print(wiki.get_top_pages())