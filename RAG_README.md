# RAG Implementation for Gene Ontology Analysis

This document provides a comprehensive guide to the Retrieval-Augmented Generation (RAG) implementation for Gene Ontology (GO) term analysis.

## Directory Structure

```
.
├── data/
│   └── GO_term_analysis/
│       ├── rag_vs_traditional_comparison.tsv  # Results comparison file
│       └── toy_example.csv                    # Example input data
├── jsonFiles/
│   └── toyexample.json                        # Configuration file
├── utils/
│   ├── chroma_utils.py                        # ChromaDB utilities
│   ├── openai_query.py                        # OpenAI API interface
│   └── prompt_factory.py                      # Prompt generation
├── test_rag_vs_traditional.py                 # Main comparison script
├── test_new_go_terms.py                       # Script to test new GO terms
└── RAG_README.md                              # This documentation
```

## Key Components

### 1. ChromaDB Integration (`utils/chroma_utils.py`)

- `ChromaGO` class for managing GO term embeddings
- Functions for loading GO terms and retrieving relevant context
- Vector similarity search for finding related GO terms

### 2. OpenAI Interface (`utils/openai_query.py`)

- Handles API calls to OpenAI
- Manages rate limits and token usage
- Tracks costs and usage statistics

### 3. Prompt Generation (`utils/prompt_factory.py`)

- Creates structured prompts for both traditional and RAG analysis
- Formats gene lists and context appropriately
- Handles confidence score generation

### 4. Testing Scripts

#### `test_rag_vs_traditional.py`

- Main comparison script for batch processing
- Implements comprehensive comparison between traditional and RAG approaches
- Features:
  - Path validation
  - Rate limiting
  - Progress saving
  - Error handling
  - Results comparison
  - Detailed logging

#### `test_new_go_terms.py`

- Interactive script for testing individual or small batches of GO terms
- Provides detailed summary statistics
- Features:
  - Single GO term testing
  - CSV batch testing
  - Real-time statistics
  - Score extraction and comparison
  - Progress tracking

## Configuration

The `toyexample.json` configuration file contains:

```json
{
  "GPT_MODEL": "gpt-3.5-turbo-0125",
  "TEMP": 0.0,
  "MAX_TOKENS": 500,
  "RATE_PER_TOKEN": 0.000002,
  "LOG_NAME": "logs/rag_analysis.log",
  "DOLLAR_LIMIT": 5.0,
  "CONTEXT": "You are a helpful assistant..."
}
```

## Usage

### Main Comparison Script (`test_rag_vs_traditional.py`)

1. **Prepare Test Data**

   - Create a CSV file with columns: GO, Genes, Gene_Count, Term_Description
   - Place it in `data/GO_term_analysis/`

2. **Run the Comparison**

```bash
python test_rag_vs_traditional.py
```

3. **View Results**
   - Results are saved to `data/GO_term_analysis/rag_vs_traditional_comparison.tsv`
   - The file includes:
     - GO term identifiers
     - Gene lists
     - Traditional and RAG responses
     - Confidence scores
     - Score differences

### Testing New GO Terms (`test_new_go_terms.py`)

1. **Single GO Term Testing**:

```bash
python test_new_go_terms.py
# Choose option 1
# Enter GO term (e.g., GO:0000001)
# Enter genes separated by spaces
```

2. **Batch Testing from CSV**:

```bash
python test_new_go_terms.py
# Choose option 2
# Enter path to CSV file
```

### Output and Statistics

Both scripts generate comprehensive statistics including:

- Total number of GO terms analyzed
- Mean scores for both traditional and RAG methods
- Mean difference between RAG and traditional scores
- Count of terms where RAG scored higher, traditional scored higher, or scores were equal
- Maximum and minimum score differences
- Standard deviation of score differences

Results are saved in `data/GO_term_analysis/rag_vs_traditional_comparison.tsv` with columns:

- GO: GO term identifier
- Genes: Space-separated list of genes
- Traditional_Response: Analysis without RAG
- RAG_Response: Analysis with RAG context

### Score Extraction

The system automatically extracts confidence scores from responses using the following format:

```
Process: [analysis] (0.XX)
```

where 0.XX is the confidence score between 0 and 1.

## Error Handling

The system includes several error handling mechanisms:

1. File path validation
2. API rate limit management
3. Cost tracking and limits
4. Graceful handling of missing files
5. Progress saving to prevent data loss
6. Exponential backoff on rate limit errors

## Updating the Vector Database

To update the ChromaDB with new GO terms:

1. Add new terms to your data source
2. The system will automatically incorporate new terms during analysis
3. No manual database updates are required

## Best Practices

1. **API Usage**:

   - Monitor your OpenAI API usage
   - Stay within rate limits
   - Keep track of costs using the logging system

2. **Data Format**:

   - Ensure CSV files follow the correct format
   - Use valid GO term identifiers
   - Separate genes with spaces in the input

3. **Performance**:
   - The RAG system typically provides more detailed analysis
   - Compare results using the provided statistics
   - Monitor score differences to evaluate RAG effectiveness

## Troubleshooting

Common issues and solutions:

1. **API Rate Limits**:

   - Reduce batch sizes
   - Implement exponential backoff
   - Check your OpenAI account status

2. **File Not Found**:

   - Verify file paths
   - Check directory structure
   - Ensure required files exist

3. **Score Extraction Issues**:
   - Verify response format
   - Check for proper parentheses in scores
   - Ensure Process line is properly formatted

For additional support, please refer to the code documentation or contact the development team.
