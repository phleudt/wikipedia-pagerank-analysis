# Wikipedia PageRank Analysis
A Python implementation of Google's PageRank algorithm (The anatomy of a large-scale hypertextual Web search engine),
applied to Wikipedia data dumps, originally developed as part of a high school thesis (Abitur) in 2022-2023 with a major revision in 2024.

## Projekt Overview
This project implements Google's PageRank algorithm and a possible extension using Wikipedia data as a test case. 
The extension forms the basis of the accompanying thesis paper, which explores a temporal-based modification of 
the algorithm. This extension incorporates the last editing date for a page as an additional factor, enhancing the 
algorithm's adaptability to dynamic content. The project demonstrates the processing of large amounts of data while 
maintaining memory efficiency when working with multi-gigabyte Wikipedia dumps.

### Key Features
- Implementation of the PageRank algorithm optimized for large datasets
- Memory-efficient processing of Wikipedia data dumps
- Performance optimizations for handling gigabyte-scale data
- Clean, maintainable code structure following Python best practices

## Technical Highlights
- Memory Optimization: Implemented efficient data structures and processing methods to handle large Wikipedia dumps
- Performance Improvements: Significant optimizations in the 2024 update to reduce memory footprint and processing time
- Code Quality: Restructured codebase following software engineering best practices
- Documentation: Comprehensive inline documentation and clear project structure

## Getting Started
### Prerequisites
- Python 3.8 or higher
- Sufficient RAM to process Wikipedia dumps (recommended: 16GB+)
- Git

### Installation
1. Clone the repository:
    ```
    git clone https://github.com/phleudt/wikipedia-pagerank-analysis
    cd wikipedia-pagerank-analysis
    ```

2. Create and activate a virtual environment (recommended):
    ```
    # Create virtual environment
    python -m venv .venv

    # Activate virtual environment
    # macOs/Linux
    source venv/bin/activate
    # Windows (PowerShell)
    .\venv\Scripts\Activate
    ```

3. Install dependencies:
    ```
   pip install -r requirements.txt
   pip list  # To check the installation 
    ```
   
### Usage
Run the main analysis:
`python3 src/__main__.py`


## Project History and Evolution
- 2022-2023: Initial development as paart of Abitur thesis
- 2024 Update: Major refactoring and improvements including:
  - Code restructuring for better maintainability
  - Performance optimizations
  - Implementation of Python best practices
  - Enhanced documentation
  - Improved project setup process

## Future Development Plans
- Implementation of comprehensive test cases
- Further performance optimizations
- Additional algorithm variations and analysis
- Enhanced documentation

## Documentation
The original thesis paper (in German) can be found at:
`./paper/Wie_koennte_eine_geheime_Erweiterung_des_PageRank_Algorithmus_von_Google_aussehen.pdf`

## Language Note
While the codebase and documentation are primarily in English, the accompanying thesis paper is written in German as it 
was part of the German Abitur requirements. Non-German speakers may use translation tools to understand its contents.

## Contributing
While this project was primarily developed as a personal academic project, suggestions and feedback are welcome through issues and pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact
For questions or feedback about this project, please open an issue on GitHub.
