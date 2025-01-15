"""
Wikipedia search implementation using WikiGraph-based PageRank scores.
Combines efficient graph operations with flexible search capabilities.
"""

# Standard Library Imports
from dataclasses import dataclass
from typing import List, NamedTuple, Dict

# Local Imports
from wikipedia.wiki_graph import *


@dataclass
class SearchConfig:
    """Configuration for Wikipedia search."""
    max_results: int = 20
    include_categories: bool = False
    exact_match_first: bool = True
    partial_match: bool = True
    case_sensitive: bool = False
    starting_with_only: bool = False # Match the beginning of titles only
    use_temporal: bool = False
    temporal_weight: float = 0.25

    def merge_with(self, other: Optional['SearchConfig']) -> 'SearchConfig':
        """Create a new config by merging with another, preferring other's values."""
        if other is None:
            return self
        return SearchConfig(
            max_results=other.max_results if other else self.max_results,
            include_categories=other.include_categories if other else self.include_categories,
            exact_match_first=other.exact_match_first if other else self.exact_match_first,
            partial_match=other.partial_match if other else self.partial_match,
            case_sensitive=other.case_sensitive if other else self.case_sensitive,
            starting_with_only=other.starting_with_only if other else self.starting_with_only,
            use_temporal=other.use_temporal if other else self.use_temporal,
            temporal_weight=other.temporal_weight if other else self.temporal_weight
        )


class SearchResult(NamedTuple):
    """Immutable container for search results."""
    title: str
    score: float
    page_id: int
    is_exact_match: bool = False


class TitleMapper:
    """Handles mapping between page IDs and titles."""

    def __init__(self, title_mapping_file: Path):
        self._id_to_title: Dict[int, str] = {}
        self._title_to_id: Dict[str, int] = {}
        self._load_titles(title_mapping_file)

    def _load_titles(self, file_path: Path) -> None:
        """Load page title mappings from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    page_id, title = line.strip().split('ÿ~')
                    page_id = int(page_id)
                    self._id_to_title[page_id] = title
                    self._title_to_id[title] = page_id
        except (IOError, ValueError) as e:
            raise RuntimeError(f"Failed to load title mappings: {e}")

    def get_title(self, page_id: int) -> Optional[str]:
        """Get title for a page ID."""
        return self._id_to_title.get(page_id)

    def get_id(self, title: str) -> Optional[int]:
        """Get ID for a page title."""
        return self._title_to_id.get(title)

    def iterate_mappings(self):
        """Iterate through all ID-title pairs."""
        return self._id_to_title.items()



class WikipediaSearch:
    """
    Wikipedia search implementation using WikiGraph-based PageRank scores.
    """

    def __init__(
            self,
            data_dir: Path,
            title_mapping_file: Path,
            config: Optional[SearchConfig] = None
    ):
        """
        Initialize Wikipedia search.

        Args:
            data_dir: Base directory for WikiGraph data
            title_mapping_file: File mapping page IDs to titles
            config: Search configuration
        """
        self.base_config = config or SearchConfig()
        self.title_mapper = TitleMapper(title_mapping_file)

        # Initialize WikiGraph
        graph_config = WikiGraphConfig(
            data_dir=data_dir,
            save_matrix=False,
            load_existing_pagerank=True,
            use_temporal=config.use_temporal,
            omega=self.base_config.temporal_weight,
        )

        self.wiki_graph = WikiGraph(graph_config)
        self._scores = self.wiki_graph.get_pagerank_vector()

    def _check_title_match(self, title: str, query: str, config: SearchConfig) -> tuple[bool, bool]:
        """
        Check if a title matches the query.
        Returns tuple of (is_exact_match, is_partial_match)
        """
        if not config.case_sensitive:
            title = title.lower()
            query = query.lower()

        is_exact = title == query

        if self.base_config.starting_with_only:
            is_partial = title.startswith(query)
        else:
            is_partial = query in title

        return is_exact, is_partial
    
    def search(self, query: str, config: Optional[SearchConfig] = None) -> List[SearchResult]:
        """
        Search Wikipedia pages using the query string.

        Args:
            query: Search query
            config: Optional configuration to override base config

        Returns:
            List of search results sorted by relevance
        """
        if not query:
            return []
        
        # Merge configs, preferring the provided config
        effective_config = self.base_config.merge_with(config)
        results: List[SearchResult] = []

        # Find matching pages
        for page_id, title in self.title_mapper.iterate_mappings():
            # Skip categories if not included
            if not self.base_config.include_categories and title.startswith('Category:'):
                continue

            # Check for matches
            is_exact, is_partial = self._check_title_match(title, query, effective_config)

            if is_exact or (effective_config.partial_match and is_partial):
                results.append(SearchResult(
                    title=title,
                    score=float(self._scores[page_id]),
                    page_id=page_id,
                    is_exact_match=is_exact
                ))

        # Sort results using key function
        results.sort(
            key=self._get_sort_key(effective_config)
        )

        return results[:effective_config.max_results]

    @staticmethod
    def _get_sort_key(config: SearchConfig):
        """Get the appropriate sort key function based on config."""
        if config.exact_match_first:
            return lambda x: (not x.is_exact_match, -x.score)
        return lambda x: -x.score

    def get_top_pages(self, n: int = 10) -> List[SearchResult]:
        """
        Get the top N pages by PageRank score.

        Args:
            n: Number of top pages to return

        Returns:
            List of search results for top-ranked pages
        """
        top_pages = self.wiki_graph.get_top_pages(n)

        return [
            SearchResult(
                title=self.title_mapper.get_title(page_id),
                score=score,
                page_id=page_id,
                is_exact_match=False
            )
            for page_id, score in top_pages
            if self.title_mapper.get_title(page_id) is not None
        ]


if __name__ == "__main__":
    # Example usage
    data_dir = Path("../../data")
    title_file = data_dir / "processed" / "wiki_sequential_page_mapping.txt"

    # Example with temporal search
    temporal_config = SearchConfig(
        max_results=10,
        include_categories=True,
        starting_with_only=True,
        use_temporal=True,
    )

    temporal_search = WikipediaSearch(
        data_dir=data_dir,
        title_mapping_file=title_file,
        config=temporal_config
    )

    print("\nTemporal search for 'Python':")
    temporal_results = temporal_search.search("Python")
    for result in temporal_results:
        print(f"{result.title}: {result.score:.6f}")

    # Example with standard search
    standard_config = SearchConfig(
        max_results=10,
        include_categories=True,
        starting_with_only=True,
        use_temporal=False
    )

    standard_search = WikipediaSearch(
        data_dir=data_dir,
        title_mapping_file=title_file,
        config=standard_config
    )

    print("\nStandard search for 'Python':")
    standard_results = standard_search.search("Python")
    for result in standard_results:
        print(f"{result.title}: {result.score:.6f}")

    print("\nTop 5 pages by PageRank:")
    top_pages = standard_search.get_top_pages(5)
    for result in top_pages:
        print(f"{result.title}: {result.score:.6f}")