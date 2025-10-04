# SDG Mapping Project Progress

## Overview

This project aims to map NGO annual reports to UN Sustainable Development Goals (SDGs) using LLM.

## Completed Tasks

### 1. SDG Data Collection and Processing 

- **What**: Extracted all 17 SDG goals with their indicators from the UN SDG API
- **How**: Used `fetch_and_save_sdg_data.py` to retrieve data from the UN Statistics API
- **Output**:
  - `sdg_indicators.json` - Complete SDG data with all goals and indicators
  - `sdg_keywords_output_*.json` files - AI-extracted keywords for each SDG (goals 1, 2, 5, 6 completed) // PENDING
- **Status**: Core data infrastructure ready

### 2. PDF Extraction Pipeline 

- **What**: Built a PDF text extraction system using Mistral Document AI
- **Components**:
  - `sdg_mapping_simple/pdf_extraction_ui.py` - Streamlit UI for interactive PDF extraction
  - `sdg_mapping_simple/clients/mistral_document_ai_client.py` - Client for Mistral Document AI API
  - Added `extract_text_to_file()` method for direct text file output
- **Tool**: `extract_pdf_simple.py` - Command-line tool to extract PDF text
- **Usage**:
  ```bash
  python extract_pdf_simple.py <pdf_url> [output_file.txt]
  ```
- **Status**: Fully functional PDF to text extraction

### 3. SDG Mapping Implementation 

- **What**: Created AI-powered mapping of text to specific SDG indicators
- **Components**:
  - `simple_sdg_mapping.py` - Maps text to SDG goals (high-level)
  - `simple_sdg_mapping_indicators.py` - Maps text to specific SDG indicators (granular)
- **Features**:
  - Uses OpenAI GPT-4 for intelligent mapping
  - Provides confidence scores and explanations
  - Groups results by SDG for easy analysis
- **Usage**:
  ```python
  # After extracting PDF text
  python simple_sdg_mapping_indicators.py
  ```
- **Status**: Basic implementation complete

## Current Workflow

1. **Extract SDG Data** (one-time setup)

   ```bash
   python fetch_and_save_sdg_data.py
   ```

2. **Extract PDF Text**

   ```bash
   python extract_pdf_simple.py https://example.com/annual_report.pdf report.txt
   ```

3. **Map to SDGs**
   ```bash
   # Modify simple_sdg_mapping_indicators.py to read from report.txt
   python simple_sdg_mapping_indicators.py
   ```

## Next Steps

### Short-term

1. **Complete SDG Keywords**: Generate keywords for remaining SDGs (3, 4, 7-17)
2. **Automate Pipeline**: Create end-to-end script that combines extraction and mapping
3. **Improve Text Input**: Update mapping scripts to accept file paths as arguments

### Medium-term

1. **Vector Search Implementation**: Build embedding-based SDG matching for better accuracy
2. **Evaluation Framework**: Create metrics to measure mapping quality
3. **Batch Processing**: Handle multiple reports efficiently

### Long-term

1. **Web Interface**: Build complete web app for report upload and analysis
2. **API Development**: Create REST API for SDG mapping service
3. **Advanced Analytics**: Generate insights and trends across multiple reports

## Technical Stack

- **PDF Processing**: Mistral Document AI
- **LLM**: OpenAI GPT-4
- **Data Storage**: JSON files (to be upgraded to vector DB)
- **Languages**: Python
- **UI**: Streamlit (for prototyping)

## Known Issues

- Missing keyword files for SDGs 3, 4, 7-17
- Manual text file handling between extraction and mapping
- No persistent storage for results
- Limited error handling and validation

## Resources

- UN SDG API: https://unstats.un.org/SDGAPI/
- Project Repository: [Current Directory]
