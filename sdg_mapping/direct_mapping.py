"""
Direct LLM Mapping Module

This module provides functionality to directly map NGO reports to SDGs
using LLM analysis without intermediate steps.
"""

import json
from typing import List, Tuple, Dict, Any
from langfuse import observe
from langfuse.openai import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from .config import get_config, PROMPTS
from .utils import chunk_text_with_llamaparse


@observe()
def direct_llm_mapping(report_text: str, sdg_descriptions: Dict[int, str]) -> List[Tuple[int, float, str]]:
    """
    Directly map report to SDGs using LLM analysis.
    
    Args:
        report_text: Complete report content
        sdg_descriptions: Dictionary of SDG numbers to descriptions
        
    Returns:
        List of (sdg_number, confidence_score, rationale) tuples
    """
    config = get_config()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model_name=config.llm_model,
        temperature=config.llm_temperature,  # Low temperature for more consistent results
        openai_api_key=config.openai_api_key
    )
    
    # Prepare the SDG descriptions
    sdg_desc_text = "\n\n".join([
        f"SDG {num}: {desc}" 
        for num, desc in sdg_descriptions.items()
    ])
    
    # If report is too long, chunk it
    if len(report_text) > config.max_report_length:
        # Take the most relevant parts or summarize
        chunks = chunk_text_with_llamaparse(report_text, chunk_size=config.chunk_size * 4, overlap=config.chunk_overlap * 4)
        # Process first few chunks for now
        report_text = "\n\n---\n\n".join(chunks[:5])
    
    # Create the prompt
    prompt = PROMPTS["direct_mapping"].format(
        report_text=report_text,
        sdg_descriptions=sdg_desc_text
    )
    
    messages = [
        SystemMessage(content="You are an expert in UN Sustainable Development Goals analysis."),
        HumanMessage(content=prompt)
    ]
    
    try:
        # Get LLM response
        response = llm.invoke(messages)
        
        # Parse JSON response
        mappings = json.loads(response.content)
        
        # Convert to expected format
        results = []
        for mapping in mappings:
            sdg_number = int(mapping['sdg_number'])
            confidence_score = float(mapping['confidence_score'])
            rationale = mapping['rationale']
            results.append((sdg_number, confidence_score, rationale))
        
        # Sort by confidence score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
        
    except json.JSONDecodeError:
        # Fallback: try to extract information from unstructured response
        return _parse_unstructured_response(response.content, sdg_descriptions)
    except Exception as e:
        raise Exception(f"Error in direct LLM mapping: {str(e)}")


def _parse_unstructured_response(response_text: str, sdg_descriptions: Dict[int, str]) -> List[Tuple[int, float, str]]:
    """
    Fallback parser for when LLM doesn't return proper JSON.
    
    Args:
        response_text: Raw LLM response
        sdg_descriptions: Dictionary of SDG numbers to descriptions
        
    Returns:
        List of (sdg_number, confidence_score, rationale) tuples
    """
    results = []
    
    # Simple heuristic parsing
    lines = response_text.split('\n')
    current_sdg = None
    current_confidence = 0.5  # Default confidence
    current_rationale = ""
    
    for line in lines:
        line = line.strip()
        
        # Look for SDG mentions
        for sdg_num in sdg_descriptions.keys():
            if f"SDG {sdg_num}" in line or f"Goal {sdg_num}" in line:
                # Save previous SDG if any
                if current_sdg is not None:
                    results.append((current_sdg, current_confidence, current_rationale.strip()))
                
                current_sdg = sdg_num
                current_confidence = 0.5
                current_rationale = line
                
                # Try to extract confidence if mentioned
                if "high confidence" in line.lower():
                    current_confidence = 0.9
                elif "medium confidence" in line.lower():
                    current_confidence = 0.6
                elif "low confidence" in line.lower():
                    current_confidence = 0.3
                
                break
        else:
            # Accumulate rationale
            if current_sdg is not None and line:
                current_rationale += " " + line
    
    # Don't forget the last one
    if current_sdg is not None:
        results.append((current_sdg, current_confidence, current_rationale.strip()))
    
    return results