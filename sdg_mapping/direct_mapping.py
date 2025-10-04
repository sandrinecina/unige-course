"""
Direct LLM Mapping Module

This module provides functionality to directly map NGO reports to SDGs
using LLM analysis without intermediate steps.
"""

from typing import List, Tuple
from langfuse.decorators import observe


@observe()
def direct_llm_mapping(report_text: str, sdg_descriptions: dict) -> List[Tuple[int, float, str]]:
    """
    Directly map report to SDGs using LLM analysis.
    
    Args:
        report_text: Complete report content
        sdg_descriptions: Dictionary of SDG numbers to descriptions
        
    Returns:
        List of (sdg_number, confidence_score, rationale) tuples
    """
    # TODO: Implement direct mapping logic
    pass