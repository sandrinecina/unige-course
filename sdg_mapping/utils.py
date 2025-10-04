"""
Utilities Module

This module provides shared utility functions for the SDG mapping package.
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path


def load_sdg_keywords(sdg_number: int, base_path: str = "sdg") -> Dict[str, Any]:
    """
    Load SDG keywords and indicators from JSON file.
    
    Args:
        sdg_number: The SDG number to load (1-17)
        base_path: Base path to SDG data files
        
    Returns:
        Dictionary containing goal info, indicators, and keywords
    """
    file_path = Path(base_path) / f"sdg_keywords_output_{sdg_number}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[f"Goal {sdg_number}"]


def load_all_sdg_data(base_path: str = "sdg") -> Dict[int, Dict[str, Any]]:
    """
    Load all SDG data from available JSON files.
    
    Args:
        base_path: Base path to SDG data files
        
    Returns:
        Dictionary mapping SDG numbers to their data
    """
    sdg_data = {}
    for sdg_num in range(1, 18):  # SDGs 1-17
        try:
            sdg_data[sdg_num] = load_sdg_keywords(sdg_num, base_path)
        except FileNotFoundError:
            print(f"Warning: SDG {sdg_num} data not found")
            continue
    return sdg_data


def extract_pdf_content(pdf_path: str) -> str:
    """
    Extract text content from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    # TODO: Integrate with pdf_extraction_ui.py functionality
    pass