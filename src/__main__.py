"""
Main Entry Point for Wikipedia PageRank Search Tool

This script serves as the main entry point for the Wikipedia PageRank search tool.
It initializes the search engine, checks for required files, and runs the command-line interface (CLI) for user interaction.
"""
import sys
from pathlib import Path
from typing import Optional

from ui.cli import WikipediaSearchCLI
from wikipedia.search import WikipediaSearch, SearchConfig
from wikipedia.setup_wizard import run_setup


def check_required_files(data_dir: Path) -> bool:
    """
    Check if all required files exist.

    Args:
        data_dir: Data directory path

    Returns:
        bool: True if all required files exist, False otherwise
    """
    required_files = [
        "processed/wiki_matrix.npz",
        "processed/wiki_pagerank.npz",
        "processed/wiki_temporal_pagerank.npz",
        "processed/wiki_sequential_page_mapping.txt",
        "processed/wiki_page_timestamps.txt"
    ]

    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            missing_files.append(file)

    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        return False

    return True

def setup_search_engine(data_dir: Path) -> Optional[WikipediaSearch]:
    """
    Set up the search engine with required components.

    Args:
        data_dir: Data directory path

    Returns:
        Optional[WikipediaSearch]: Configured search engine or None if setup fails
    """
    try:
        title_mapping_file = data_dir / "processed" / "wiki_sequential_page_mapping.txt"

        # Prompt user for search configuration
        config = prompt_for_search_config()

        # Initialize search engine
        search = WikipediaSearch(
            data_dir=data_dir,
            title_mapping_file=title_mapping_file,
            config=config
        )

        return search

    except Exception as e:
        print(f"Error setting up search engine: {e}", file=sys.stderr)
        return None

def prompt_for_search_config() -> SearchConfig:
    """Prompt the user for search configuration parameters."""
    try:
        max_results: int = int(input("Enter the maximum number of search results (default is 20): ").strip() or 20)
        include_categories: bool = input("Include categories in search results? [Y/n]: ").strip().lower() != 'n'
        exact_match_first: bool = input("Prioritize exact matches first? [Y/n]: ").strip().lower() != 'n'
        partial_match: bool = input("Allow partial matches? [Y/n]: ").strip().lower() != 'n'
        case_sensitive: bool = input("Case sensitive search? [y/N]: ").strip().lower() == 'y'
        starting_with_only: bool = input("Match titles starting with the query only? [y/N]: ").strip().lower() == 'y'
        use_temporal: bool = input("Use temporal PageRank? [y/N]: ").strip().lower() == 'y'
        temporal_weight = float(input("Enter the temporal weight (0 to 1, default is 0.25): ").strip() or 0.25)

        return SearchConfig(
            max_results=max_results,
            include_categories=include_categories,
            exact_match_first=exact_match_first,
            partial_match=partial_match,
            case_sensitive=case_sensitive,
            starting_with_only=starting_with_only,
            use_temporal=use_temporal,
            temporal_weight=temporal_weight
        )
    except ValueError as e:
        print(f"Invalid input: {e}. Using default configuration.")
        return SearchConfig()

def main():
    """Main entry point for the Wikipedia PageRank search tool."""
    try:
        # Set up paths
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data"

        # Check if setup is needed
        if not check_required_files(data_dir):
            print("Required files not found. Running setup wizard...")
            if not run_setup(base_dir):
                print("Setup failed. Please try again.", file=sys.stderr)
                sys.exit(1)

        # Initialize search engine
        search_engine = setup_search_engine(data_dir)
        if not search_engine:
            sys.exit(1)

        # Create and run CLI
        cli = WikipediaSearchCLI(search_engine)
        cli.run()

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
