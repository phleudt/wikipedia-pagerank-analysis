"""
Setup Wizard for Wikipedia PageRank Analysis

This script provides a setup wizard for configuring and initializing the Wikipedia PageRank analysis.
It includes steps for downloading, extracting, parsing Wikipedia dumps, and calculating PageRank matrices.
"""

# Standard Library Imports
import bz2
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Third-Party Imports
import requests
from tqdm import tqdm

# Local Imports
from data_processing.wiki_xml_parser import WikiXMLParser
from wikipedia.wiki_graph import WikiGraph, WikiGraphConfig


class SetupWizard:
    """Setup wizard for initial Wikipedia PageRank analysis configuration."""

    WIKI_DUMP_URL = "https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles-multistream.xml.bz2"

    def __init__(self, base_dir: Path):
        """
        Initialize setup wizard.

        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.dump_file = self.raw_dir / "dewiki-latest-pages-articles-multistream.xml.bz2"
        self.extracted_file = self.raw_dir / "dewiki-latest-pages-articles-multistream.xml"

    def download_dump(self) -> bool:
        """
        Download Wikipedia dump file with progress bar.

        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\nStep 1: Downloading Wikipedia dump from {self.WIKI_DUMP_URL}")

        if self.dump_file.exists():
            response = input("Dump file already exists. Download again? [y/N]: ")
            if response.lower() != 'y':
                return True

        try:
            response = requests.get(self.WIKI_DUMP_URL, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
                # Use tempfile to handle partial downloads safely
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    try:
                        for data in response.iter_content(block_size):
                            if data:  # Filter out keep-alive chunks
                                progress_bar.update(len(data))
                                temp_file.write(data)

                        # Move temporary file to final location only after successful download
                        shutil.move(temp_file.name, self.dump_file)
                        return True
                    except Exception as e:
                        os.unlink(temp_file.name)  # Clean up temp file on error
                        raise e

        except requests.Timeout:
            print("Download timed out. Please check your internet connection.", file=sys.stderr)
            return False
        except requests.RequestException as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return False
        except IOError as e:
            print(f"File operation failed: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Unexpected error during download: {e}", file=sys.stderr)
            return False

    def extract_dump(self) -> bool:
        """
        Extract the bz2 compressed dump file with progress bar.

        Returns:
            bool: True if successful, False otherwise
        """
        print("\nStep 2: Extracting dump file")

        if self.extracted_file.exists():
            response = input("Extracted file already exists. Extract again? [y/N]: ")
            if response.lower() != 'y':
                return True

        try:
            total_size = os.path.getsize(self.dump_file)

            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                with bz2.open(self.dump_file, 'rb') as source, \
                        open(self.extracted_file, 'wb') as target:
                    while True:
                        block = source.read(8192)
                        if not block:
                            break
                        target.write(block)
                        progress_bar.update(len(block))

            return True

        except Exception as e:
            print(f"Error extracting dump file: {e}", file=sys.stderr)
            return False

    def parse_dump(self) -> bool:
        """
        Parse the Wikipedia XML dump using WikiXMLParser.

        Returns:
            bool: True if successful, False otherwise
        """
        print("\nStep 3: Parsing Wikipedia dump")

        try:
            parser = WikiXMLParser(
                xml_file_path=str(self.extracted_file),
                output_dir=str(self.processed_dir)
            )
            parser.process_wikipedia_dump()
            return True

        except Exception as e:
            print(f"Error parsing dump file: {e}", file=sys.stderr)
            return False

    def cleanup(self) -> bool:
        """
        Clean up temporary files to save disk space.

        Returns:
            bool: True if successful, False otherwise
        """
        print("\nStep 4: Cleaning up temporary files")

        temporary_file_names = [
            "wiki_page_id_to_title_mapping.txt",
            "wiki_raw_page_data.txt",
            self.dump_file.name,
            self.extracted_file.name
        ]

        temporary_file_paths = [self.processed_dir / temporary_file for temporary_file in temporary_file_names]
        print(temporary_file_paths)

        try:
            print("All temporary files: ")
            for temporary_file in temporary_file_names:
                print(f" - {temporary_file}")
            response = input("Delete all temporary files? [y/N]: ")

            if response.lower() == 'y':
                for temporary_file_path in temporary_file_paths:
                    if isinstance(temporary_file_path, Path) and temporary_file_path.exists():
                        temporary_file_path.unlink()
                    elif os.path.exists(temporary_file_path):
                        os.remove(temporary_file_path)

            else:
                for temporary_file_path in temporary_file_paths:
                    if isinstance(temporary_file_path, Path):
                        file_path_str = str(temporary_file_path)
                    else:
                        file_path_str = temporary_file_path

                    if os.path.exists(file_path_str):
                        response = input(f"Delete {file_path_str}? [y/N]: ")
                        if response.lower() == 'y':
                            if isinstance(temporary_file_path, Path):
                                temporary_file_path.unlink()
                            else:
                                os.remove(file_path_str)
            return True

        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr)
            return False

    def calculate_matrices(self) -> bool:
        """
        Calculate necessary matrices and PageRank vectors.

        Returns:
            bool: True if successful, False otherwise
        """
        print("\nStep 5: Calculating matrices and PageRank vectors")

        try:
            # Standard PageRank configuration
            standard_config = WikiGraphConfig(
                data_dir=self.data_dir,
                save_matrix=True,
                load_existing_pagerank=False,
                use_temporal=False
            )

            # Create and process standard WikiGraph
            print("Calculating standard PageRank...")
            wiki_graph = WikiGraph(standard_config)
            wiki_graph.compute_pagerank(save_results=True)

            # Prompt user for omega value
            omega = self.prompt_for_omega()

            # Temporal PageRank configuration
            temporal_config = WikiGraphConfig(
                data_dir=self.data_dir,
                save_matrix=False,  # Reuse existing matrix
                load_existing_pagerank=False,
                use_temporal=True,
                timestamp_file="wiki_page_timestamps.txt",
                omega=omega
            )

            # Create and process temporal WikiGraph
            print("Calculating temporal PageRank...")
            temporal_graph = WikiGraph(temporal_config)
            temporal_graph.compute_pagerank(save_results=True)

            return True

        except Exception as e:
            print(f"Error calculating matrices: {e}", file=sys.stderr)
            return False

    @staticmethod
    def prompt_for_omega() -> float:
        """Prompt the user for the omega value."""
        while True:
            try:
                omega = float(input(
                    "Enter the omega value (0 to 1) for temporal PageRank calculation (default is 0.25): ").strip() or 0.25)
                if 0 <= omega <= 1:
                    return omega
                else:
                    print("Please enter a value between 0 and 1.")
            except ValueError:
                print("Invalid input. Please enter a numeric value between 0 and 1.")

    def run(self) -> bool:
        """
        Run the complete setup process.

        Returns:
            bool: True if all steps completed successfully, False otherwise
        """
        steps = [
            (self.download_dump, "Downloading Wikipedia dump"),
            (self.extract_dump, "Extracting dump file"),
            (self.parse_dump, "Parsing Wikipedia dump"),
            (self.cleanup, "Cleaning up temporary files"),
            (self.calculate_matrices, "Calculating matrices")
        ]

        print("Wikipedia PageRank Analysis Setup Wizard")
        print("=======================================")

        for step_func, description in steps:
            try:
                if not step_func():
                    print(f"\nSetup failed during: {description}")
                    return False
            except KeyboardInterrupt:
                print("\nSetup interrupted by user")
                return False

        print("\nSetup completed successfully!")
        return True


def run_setup(base_dir: Optional[Path] = None) -> bool:
    """
    Run the setup wizard.

    Args:
        base_dir: Base directory for the project. If None, uses the parent of the parent of the current directory

    Returns:
        bool: True if setup completed successfully, False otherwise
    """
    if base_dir is None:
        base_dir = Path.cwd().parent.parent

    wizard = SetupWizard(base_dir)
    return wizard.run()


if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1)