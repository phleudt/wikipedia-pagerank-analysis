"""
Wikipedia CLI Module

This module provides a command-line interface for searching Wikipedia pages using the PageRank algorithm.
"""

# Standard Library Imports
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

# Local Imports
from wikipedia.search import WikipediaSearch, SearchConfig


@dataclass
class CLIPrompts:
    """Container for CLI prompt messages"""
    welcome: str = "Mit dieser Datei können Sie nach Wikipedia-Seiten suchen"
    algorithm: str = "Der PageRank-Algorithmus wird zu diesem Zweck verwendet"
    search_prompt: str = "Möchten Sie nach einer Seite suchen? [J] [N]"
    search_term: str = "Nach welcher Seite suchen Sie?"
    advanced_search: str = "Möchten Sie die erweiterte Suche verwenden? [J] [N]"
    max_results: str = "Wie viele Seiten sollen maximal angezeigt werden?"
    start_with: str = "Soll die Seite mit Ihrem Suchbegriff beginnen? [J] [N]"
    include_categories: str = "Möchten Sie Wikipedia-Kategorien einbeziehen? [J] [N]"
    use_timestamps: str = "Soll der Zeitpunkt der letzten Aktualisierung berücksichtigt werden? [J] [N]"
    exact_match_first: str = "Sollen exakte Übereinstimmungen zuerst angezeigt werden? [J] [N]"
    partial_match: str = "Sollen auch Teilübereinstimmungen gefunden werden? [J] [N]"
    case_sensitive: str = "Soll die Groß-/Kleinschreibung berücksichtigt werden? [J] [N]"
    no_results: str = "Keine Suchergebnisse: {homepage}"
    error: str = "Fehler bei der Suche: {error}"
    exit: str = "Das Programm wird beendet"
    search_results: str = 'Suchergebnisse für den Suchbegriff "{term}":'

@dataclass
class CLIConfig:
    """Configuration for the CLI interface"""
    language: str = "de"
    base_url: str = "https://de.wikipedia.org/wiki/"
    prompts: CLIPrompts = field(default_factory=CLIPrompts)


class UserInteraction:
    """Handles all user input and output formatting"""

    def __init__(self, config: CLIConfig):
        self.config = config

    @staticmethod
    def get_yes_no(prompt: str) -> Optional[bool]:
        """Get yes/no input from user. Returns True for 'J', False for 'N', None for invalid input."""
        response = input(f"{prompt} ").strip().lower()
        if response == "j":
            return True
        elif response == "n":
            return False
        return None

    @staticmethod
    def get_number(prompt: str) -> Optional[int]:
        """Get numeric input from user. Returns None if invalid."""
        try:
            return int(input(f"{prompt} ").strip())
        except ValueError:
            return None

    @staticmethod
    def get_search_term(prompt: str) -> Optional[str]:
        """Get search term from user. Returns None if empty."""
        term = input(f"{prompt} ").strip()
        return term if term else None

    def build_search_config(self) -> Optional[SearchConfig]:
        """Build SearchConfig object based on user input"""
        max_results = self.get_number(self.config.prompts.max_results)
        if max_results is None:
            return None
        config = SearchConfig(max_results=max_results)

        # Get boolean configurations
        configs = [
            ('starting_with_only', self.config.prompts.start_with),
            ('include_categories', self.config.prompts.include_categories),
            ('use_temporal', self.config.prompts.use_timestamps),
            ('exact_match_first', self.config.prompts.exact_match_first),
            ('partial_match', self.config.prompts.partial_match),
            ('case_sensitive', self.config.prompts.case_sensitive)
        ]

        for attr, prompt in configs:
            value = self.get_yes_no(prompt)
            if value is None:
                return None
            setattr(config, attr, value)

        return config

    def format_wiki_link(self, title: str) -> str:
        """Format Wikipedia page title into a URL"""
        return f"{self.config.base_url}{title.replace(' ', '_')}"

    def display_results(self, results: List[str], query: str) -> None:
        """Display search results in the console"""
        print()
        print(self.config.prompts.search_results.format(term=query))

        if not results:
            homepage = self.format_wiki_link("Wikipedia:Hauptseite")
            print(self.config.prompts.no_results.format(homepage=homepage))
            return

        for title in results:
            link = self.format_wiki_link(title)
            print(f"{title}: {link}")

class WikipediaSearchCLI:
    """Command-line interface for searching Wikipedia using PageRank."""

    def __init__(self,search_engine: WikipediaSearch, config: Optional[CLIConfig] = None):
        self.search_engine = search_engine
        self.config = config or CLIConfig()
        self.ui = UserInteraction(self.config)

    def run_search(self, query: str, advanced: bool = False) -> bool:
        """Run a search query with optional advanced settings."""
        try:
            if advanced:
                advanced_config = self.ui.build_search_config()
                if advanced_config is None:
                    return False
                # Use the advanced config for this search only
                results = self.search_engine.search(query, config=advanced_config)
            else:
                # Use default config
                results = self.search_engine.search(query, config=self.config)

            self.ui.display_results([r.title for r in results], query)
            return True

        except Exception as e:
            print(self.config.prompts.error.format(error=str(e)))
            return False

    def run(self) -> None:
        """Run the interactive CLI for Wikipedia search"""
        print(self.config.prompts.welcome)
        print(self.config.prompts.algorithm)
        print()

        while True:
            should_search = self.ui.get_yes_no(self.config.prompts.search_prompt)
            if should_search is None:
                continue
            if not should_search:
                print(self.config.prompts.exit)
                break

            query = self.ui.get_search_term(self.config.prompts.search_term)
            if not query:
                continue

            advanced = self.ui.get_yes_no(self.config.prompts.advanced_search)
            if advanced is None:
                continue

            self.run_search(query, advanced)
            print()

if __name__ == "__main__":
    data_dir = Path("../../data")
    title_file = data_dir / "processed" / "wiki_sequential_page_mapping.txt"

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

    cli = WikipediaSearchCLI(standard_search)
    cli.run()