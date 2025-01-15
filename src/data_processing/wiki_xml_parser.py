"""
WikiXMLParser Module

This module provides a set of classes for parsing Wikipedia XML dumps, extracting page data, 
and generating structured outputs for PageRank calculations.
"""

# Standard Library Imports
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator, Set

# Third-Party Imports
from lxml import etree


@dataclass
class WikiParserPaths:
    """Container for all file paths used by the parser"""
    xml_file: Path
    output_dir: Path
    raw_data: Path
    id_to_title: Path
    sequential_mapping: Path
    reference_graph: Path
    timestamps: Path

    @classmethod
    def create(cls, xml_path: str, output_dir: str) -> 'WikiParserPaths':
        """Factory method to create WikiParserPaths with proper path objects"""
        base_dir = Path(output_dir)
        return cls(
            xml_file=Path(xml_path),
            output_dir=base_dir,
            raw_data=base_dir / "wiki_raw_page_data.txt",
            id_to_title=base_dir / "wiki_page_id_to_title_mapping.txt",
            sequential_mapping=base_dir / "wiki_sequential_page_mapping.txt",
            reference_graph=base_dir / "wiki_sorted_page_reference_graph.txt",
            timestamps=base_dir / "wiki_page_timestamps.txt"
        )

class ProcessingStage(Enum):
    """Enum for different processing stages with their display messages"""
    XML_PARSING = "Parsing Wikipedia XML dump"
    ID_MAPPING = "Creating page ID to title mapping"
    SEQUENTIAL_MAPPING = "Generating sequential page mapping"
    GRAPH_CREATION = "Building page reference graph"
    TIMESTAMP_EXTRACTION = "Extracting page timestamps"


class ProcessingState:
    """Class to manage processing state and decisions"""

    def __init__(self, paths: 'WikiParserPaths'):
        self.paths = paths
        self.raw_data_exists: bool = paths.raw_data.exists()

    def should_parse_xml(self, force: bool = False) -> bool:
        """Determine if XML parsing is needed based on existing files and user preference."""
        if force:
            return True

        if not self.raw_data_exists:
            print("No existing processed data found. XML parsing is required.")
            return True

        while True:
            user_input = input(
                "\nProcessed data already exists. Do you want to:\n"
                "  [r] Reparse the XML file (may take significant time)\n"
                "  [u] Use existing processed data\n"
                "Choice [r/U]: "
            ).lower()

            if user_input in ('', 'u'):
                print("Using existing processed data...")
                return False
            if user_input == 'r':
                print("Reparsing XML file...")
                return True

            print("Invalid choice. Please try again.")


class WikiXMLParser:
    """Wikipedia XML dump parser for extracting page links and metadata."""
    DELIMITER: str = "ÿ~" # Unique delimiter unlikely to appear in Wikipedia content

    def __init__(self, xml_file_path: str, output_dir: str) -> None:
        """
        Initialize the WikiXMLParser with file paths.
        
        Args:
            xml_file_path: Path to the Wikipedia XML dump file
            output_dir: Directory where output files will be stored
            
        Raises:
            FileNotFoundError: If input XML file or output directory doesn't exist
        """

        self.paths = WikiParserPaths.create(xml_file_path, output_dir)
        self._validate_paths()
        self.state = ProcessingState(self.paths)

    def _validate_paths(self) -> None:
        """Validate existence of input file and output directory"""
        if not self.paths.xml_file.exists():
            raise FileNotFoundError(f"Input XML file not found: {self.paths.xml_file}")
        if not self.paths.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.paths.output_dir}")

    def process_wikipedia_dump(self, force_xml_parse: bool = False) -> None:
        """Execute the complete Wikipedia dump processing pipeline with progress tracking."""
        should_parse_xml = self.state.should_parse_xml(force_xml_parse)

        stages: List[Tuple[ProcessingStage, callable]]

        if should_parse_xml:
            stages = [
                (ProcessingStage.XML_PARSING, self._parse_xml_content),
                (ProcessingStage.ID_MAPPING, self._create_id_title_mapping),
                (ProcessingStage.SEQUENTIAL_MAPPING, self._create_sequential_mapping),
                (ProcessingStage.GRAPH_CREATION, self._create_graph_structure),
                (ProcessingStage.TIMESTAMP_EXTRACTION, self._extract_page_timestamps)
            ]
        else:
            # Skip XML parsing if data already exists
            stages = [
                (ProcessingStage.ID_MAPPING, self._create_id_title_mapping),
                (ProcessingStage.SEQUENTIAL_MAPPING, self._create_sequential_mapping),
                (ProcessingStage.GRAPH_CREATION, self._create_graph_structure),
                (ProcessingStage.TIMESTAMP_EXTRACTION, self._extract_page_timestamps)
            ]

        # Execute selected stages
        total_stages = len(stages)
        for i, (stage, processor) in enumerate(stages, 1):
            print(f"\n[Stage {i}/{total_stages}] {stage.value}")
            processor()
            print(f" Completed {stage.value}")

        print("\n Wikipedia dump processing completed successfully!")

    def _parse_xml_content(self) -> None:
        """
        Parse the XML file and extract relevant content with progress tracking.

        Source: https://stackoverflow.com/questions/12160418/why-is-lxml-etree-iterparse-eating-up-all-my-memory/12161078#12161078
        """
        context = etree.iterparse(self.paths.xml_file)
        total_size: int = os.path.getsize(self.paths.xml_file)
        lines_estimation: int = total_size // self._estimate_line_count(self.paths.xml_file)
        progress_interval: int = max(lines_estimation // 100, 1)  # Update progress every 1%

        with open(self.paths.raw_data, "w", encoding="utf-8") as output_file:
            for i, (_, element) in enumerate(context):
                self._process_xml_element(element, output_file)

                # Memory cleanup
                element.clear()
                for ancestor in element.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]

                if i % progress_interval == 0:
                    progress: float = (i / lines_estimation) * 100
                    print(f"\rProgress: {progress:.2f}%", end='')
            del context

        print("\rXML Parsing Progress: 100%")

    @staticmethod
    def _estimate_line_count(filename: Path, sample_size: int = 1 << 16) -> int:
        """
        Estimate the total number of lines in a file based on a sample.

        Args:
            filename: Path to the file
            sample_size: Size of the sample to read

        Returns:
            Estimated number of lines in the file
        """
        with open(filename, 'rb') as file:
            sample: bytes = file.read(sample_size)
            return len(sample) // max(sample.count(b'\n'), 1)

    def _process_xml_element(self, element, output_file) -> None:
        """
        Process a single XML element and extract relevant data.

        Args:
            element: XML element to process
            output_file: File handle to write processed data
        """

        if "title" in element.tag or "id" in element.tag or "timestamp" in element.tag:
            output_file.write(self.DELIMITER + element.text)

        elif "text" in element.tag:
            references: List[str] = self._extract_wiki_links(element.text)
            if references:
                reference_text: str = self.DELIMITER.join([''] + references)
                output_file.write(f"{reference_text}\n")

    @staticmethod
    def _extract_wiki_links(text: Optional[str]) -> List[str]:
        """
        Extract Wikipedia page links from text enclosed in double square brackets.

        Args:
            text: Raw text containing Wikipedia markup

        Returns:
            List of extracted page links
        """
        try:
            return [x for x in re.findall(r"\[\[(.*?)]]", text)] if text else []
        except TypeError:
            return []

    def _create_id_title_mapping(self) -> None:
        """CreateID to page title mapping with progress tracking"""
        total_lines = sum(1 for _ in open(self.paths.raw_data, 'r', encoding='utf-8'))

        with open(self.paths.id_to_title, "w", encoding="utf-8") as output_file:
            for i, line in enumerate(self._read_file_lines(self.paths.raw_data), 1):
                parts = line.split(self.DELIMITER)
                if len(parts) >= 3:
                    output_file.write(f"{parts[2]}{self.DELIMITER}{parts[1]}\n")

                if i % 10000 == 0:
                    print(f"\rProgress: {min(99, int((i / total_lines) * 100))}%", end='', flush=True)

    def _create_sequential_mapping(self) -> None:
        """Generate sequential page mapping with progress tracking"""
        total_lines = sum(1 for _ in open(self.paths.id_to_title, 'r', encoding='utf-8'))

        with open(self.paths.sequential_mapping, "w", encoding="utf-8") as output_file:
            for i, line in enumerate(self._read_file_lines(self.paths.id_to_title)):
                parts = line.split(self.DELIMITER)[1:]
                output_file.write(f"{i}{self.DELIMITER}{self.DELIMITER.join(map(str, parts))}")

                if i % 10000 == 0:
                    print(f"\rProgress: {min(99, int((i / total_lines) * 100))}%", end='', flush=True)
    
    @staticmethod
    def _read_file_lines(file_path: Path) -> Iterator[str]:
        """Generator for memory-efficient file reading"""
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                yield line

    def _create_graph_structure(self) -> None:
        """
        Convert page references to numerical IDs for graph representation.
        Ensures each reference appears only once per page and sorts them.
        """
        title_to_id: Dict[str, str] = self._build_title_to_id_mapping()
        total_lines = sum(1 for _ in open(self.paths.raw_data, 'r', encoding='utf-8'))

        with open(self.paths.reference_graph, "w", encoding="utf-8") as output_file:
            for i, line in enumerate(self._read_file_lines(self.paths.raw_data)):
                references = line.split(self.DELIMITER)[7:]
                unique_refs = self._process_references(references, title_to_id)
                output_file.write(f"{i}{self.DELIMITER}{self.DELIMITER.join(sorted(unique_refs, key=int))}\n")

                if i % 10000 == 0:
                    print(f"\rProgress: {min(99, int((i / total_lines) * 100))}%", end='', flush=True)

    def _build_title_to_id_mapping(self) -> Dict[str, str]:
        """Build efficient title to ID mapping dictionary"""
        title_to_id = {}
        for line in self._read_file_lines(self.paths.sequential_mapping):
            parts = line.split(self.DELIMITER)
            if len(parts) >= 2:
                title_to_id[parts[1].strip()] = parts[0]
        return title_to_id

    @staticmethod
    def _process_references(references: List[str], title_to_id: Dict[str, str]) -> Set[str]:
        """Process and deduplicate page references"""
        unique_refs = set()
        for ref in references:
            if "|" in ref and "Datei:" not in ref:
                # Handle pipe syntax (e.g., [[page|display text]])
                for subref in ref.split("|"):
                    if subref in title_to_id:
                        unique_refs.add(title_to_id[subref])
            elif ref in title_to_id:
                unique_refs.add(title_to_id[ref])
        return unique_refs

    def _extract_page_timestamps(self) -> None:
        """Extract page timestamps with progress tracking"""
        total_lines = sum(1 for _ in open(self.paths.raw_data, 'r', encoding='utf-8'))
        
        with open(self.paths.timestamps, "w", encoding="utf-8") as output_file:
            for i, line in enumerate(self._read_file_lines(self.paths.raw_data)):
                parts = line.split(self.DELIMITER)
                timestamp_index = 4 if len(parts) > 4 and "-" in parts[4] else 5
                if len(parts) > timestamp_index:
                    output_file.write(f"{parts[timestamp_index].strip()}\n")
                
                if i % 10000 == 0:
                    print(f"\rProgress: {min(99, int((i / total_lines) * 100))}%", end='', flush=True)
    
if __name__ == "__main__":
    relative_data_path = Path("../../data/")
    xml_file_path = relative_data_path / "raw/dewiki-latest-pages-articles-multistream.xml"
    output_dir = relative_data_path / "processed/"

    parser = WikiXMLParser(str(xml_file_path), str(output_dir))
    parser.process_wikipedia_dump()