# Multimodal Pretraining Curation Tutorial
This tutorial demonstrates how to extract and analyze multimodal content like text, table, charts and images from PDF documents using NeMo Retriever Parse and Nano VLM. The pipeline combines document layout analysis with vision-language models to extract structured information from PDFs containing text, tables, and images. 
The dataset used in this tutorial is small, making it ideal for developing and validating data curation pipelines on either a local machine or a computing cluster. 

## Overview

The multimodal extraction pipeline processes PDF documents through a two-stage analysis:

1. **Layout Analysis**: Uses `nvidia/nemoretriever-parse` to identify and extract document elements (text, tables, charts, images) with precise bounding boxes
2. **Content Analysis**: Leverages `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` for deep analysis of visual content, including:
   - Image classification (Infographics vs Other)
   - Detailed image descriptions
   - Table reconstruction from LaTeX to HTML
   - Text content extraction

The system outputs structured JSON results containing extracted content, metadata, and analysis for each page and element.

## Walkthrough

### Pipeline Stages

#### Stage 1: Document Layout Analysis
- Converts PDF pages to high-resolution images
- Uses `nemoretriever-parse` to identify document elements
- Extracts bounding boxes and content types (Text, Table, Picture)
- Provides structured layout information for each element

#### Stage 2: Content Analysis
- **Text Elements**: Direct extraction of textual content
- **Tables**: Parses LaTeX tabular content and reconstructs as HTML tables
- **Images**: 
  - Classifies as "Infographics" or "Other"
  - Performs detailed analysis based on classification
  - For infographics: Attempts to extract tabular data or provides summaries
  - For other images: Provides detailed descriptions

### Output Structure
```json
{
  "source_filename": "document.pdf",
  "pages": [
    {
      "page_number": 1,
      "status": "Layout extraction successful",
      "content": [
        {
          "extraction_id": 0,
          "metadata": {
            "source_page": 1,
            "type": "Picture",
            "bbox": {"xmin": 0.1, "ymin": 0.2, "xmax": 0.8, "ymax": 0.6}
          },
          "data": {
            "content_classification": "infographics",
            "analysis_result": "Detailed analysis..."
          }
        }
      ]
    }
  ]
}
```

## Setup and Hardware Requirements

### Prerequisites
- Python 3.7+
- NVIDIA API access with valid API key

### Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### Environment Setup
1. Set your NVIDIA API key as an environment variable:
```bash
export NVIDIA_API_KEY="your_api_key_here"
```

2. Ensure you have the required directory structure:
```
multimodal_pretraining_curation/
├── source/
│   └── pdf_urls.jsonl  # File containing PDF URLs to process
└── multimodal_extraction.py
```

### Hardware Requirements TODO
- **Minimum**: Standard CPU with 8GB RAM
- **Recommended**: 
  - 16GB+ RAM for processing large PDFs
  - Fast internet connection for API calls
  - SSD storage for faster file I/O

## Running the Tutorial

### 1. Prepare Input Data
Create a `pdf_urls.jsonl` file in the `source/` directory with one PDF URL per line:
```json
{"0": "https://example.com/document1.pdf"}
{"0": "https://example.com/document2.pdf"}
```
For this example, we have provided example documents in the `pdf_urls.jsonl` file in the `source/` directory.

### 2. Run the Extraction Pipeline
```bash
python multimodal_extraction.py
```

### 3. Monitor Progress
The script will:
- Download PDFs from the provided URLs
- Convert each page to high-resolution images
- Process through the two-stage analysis pipeline
- Save results as JSON files in `extraction_results/`

### 4. Review Results TODO
Check the `extraction_results/` directory for JSON files containing:
- Extracted text content
- Reconstructed HTML tables
- Image classifications and analyses
- Bounding box coordinates for all elements

### Example Output
```bash
Converting 'document.pdf' to images...
Converted 5 pages.

Analyzing Page 1/5 ...
[Stage 1] Calling nemoretriever-parse for layout analysis...
[Stage 1] Found 12 document objects.
[Stage 2] Found Picture (ID: 0). Triggering Specialist VLM...
- VLM Classification: 'infographics'
- Running deep analysis...

Results saved to 'extraction_results/document.json'
```

### Troubleshooting
- **API Key Error**: Ensure `NVIDIA_API_KEY` environment variable is set
- **PDF Download Issues**: Check internet connection and URL validity
- **Memory Issues**: Reduce PDF_DPI value for lower memory usage
- **API Rate Limits**: The script includes error handling for API failures
