"""
Utilities Module

This module provides shared utility functions for the SDG mapping package.
"""

import json
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np


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


def extract_pdf_content(pdf_url: str) -> str:
    """
    Extract text content from PDF file using Mistral Document AI.
    
    Args:
        pdf_url: URL to PDF file
        
    Returns:
        Extracted text content
    """
    # Import the necessary components from the pdf_extraction module
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'sdg'))
    
    try:
        from src.clients import get_mistral_document_ai_client
        
        with get_mistral_document_ai_client() as client:
            result = client.extract_text(pdf_url)
            
        # Parse the result to get text
        if isinstance(result, dict):
            if "text" in result:
                return result["text"]
            elif "pages" in result:
                # Concatenate text from all pages
                texts = []
                for page in result["pages"]:
                    if "text" in page:
                        texts.append(page["text"])
                return "\n\n".join(texts)
        
        return str(result)
        
    except Exception as e:
        raise Exception(f"Failed to extract PDF content: {str(e)}")


def extract_pdf_structured_content(pdf_url: str, properties: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract structured content from PDF file using Mistral Document AI.
    
    Args:
        pdf_url: URL to PDF file
        properties: Dictionary defining what properties to extract
        
    Returns:
        Dictionary with extracted structured data
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'sdg'))
    
    try:
        from src.clients import get_mistral_document_ai_client
        from src.clients.mistral_document_ai_client import AnnotationType
        
        with get_mistral_document_ai_client() as client:
            result = client.extract_with_prompt(
                pdf_url,
                properties=properties,
                required=list(properties.keys()),
                annotation_type=AnnotationType.DOCUMENT,
            )
        
        # Parse the result
        if isinstance(result, dict) and "annotation" in result:
            return result["annotation"]
        
        return result
        
    except Exception as e:
        raise Exception(f"Failed to extract structured PDF content: {str(e)}")


def create_sdg_descriptions(sdg_data: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
    """
    Create comprehensive SDG descriptions from loaded data.
    
    Args:
        sdg_data: Dictionary mapping SDG numbers to their data
        
    Returns:
        Dictionary mapping SDG numbers to comprehensive text descriptions
    """
    descriptions = {}
    
    for sdg_num, data in sdg_data.items():
        # Combine name, indicators, and keywords into a comprehensive description
        desc_parts = [
            f"SDG {sdg_num}: {data['name']}",
            f"This goal has {data['indicator_count']} indicators.",
            ""
        ]
        
        # Add a few key indicators
        desc_parts.append("Key indicators include:")
        for i, indicator in enumerate(data['indicators'][:3]):  # First 3 indicators
            desc_parts.append(f"- {indicator['code']}: {indicator['description']}")
        
        # Add keywords
        desc_parts.append("")
        desc_parts.append(f"Related keywords: {', '.join(data['keywords'][:10])}")
        
        descriptions[sdg_num] = "\n".join(desc_parts)
    
    return descriptions


def format_sdg_for_embedding(sdg_data: Dict[str, Any]) -> str:
    """
    Format SDG data into a single string suitable for embedding.
    
    Args:
        sdg_data: Data for a single SDG
        
    Returns:
        Formatted string combining all relevant information
    """
    parts = [
        sdg_data['name'],
        # Include all indicator descriptions
        " ".join([ind['description'] for ind in sdg_data['indicators']]),
        # Include all keywords
        " ".join(sdg_data['keywords'])
    ]
    
    return " ".join(parts)


def chunk_text_with_llamaparse(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Placeholder for LlamaParse chunking - to be implemented when LlamaParse is integrated.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # For now, simple paragraph-based chunking
    # TODO: Replace with LlamaParse implementation
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            words = current_chunk.split()[-overlap:] if overlap > 0 else []
            current_chunk = " ".join(words) + " " + para
        else:
            current_chunk += " " + para if current_chunk else para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def calculate_confidence_score(similarity_scores: List[float], method: str = "softmax") -> List[float]:
    """
    Calculate confidence scores from similarity scores.
    
    Args:
        similarity_scores: Raw similarity scores
        method: Method to use ("softmax", "normalize", "threshold")
        
    Returns:
        List of confidence scores between 0 and 1
    """
    scores = np.array(similarity_scores)
    
    if method == "softmax":
        exp_scores = np.exp(scores - np.max(scores))
        return (exp_scores / exp_scores.sum()).tolist()
    elif method == "normalize":
        if scores.max() - scores.min() > 0:
            return ((scores - scores.min()) / (scores.max() - scores.min())).tolist()
        else:
            return [1.0] * len(scores)
    elif method == "threshold":
        threshold = 0.7
        return [1.0 if s > threshold else s for s in scores]
    else:
        return scores.tolist()