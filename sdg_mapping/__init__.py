"""
SDG Mapping Package

This package provides functionality for mapping NGO annual reports to 
UN Sustainable Development Goals using multiple approaches:
- Direct LLM mapping
- Vector search with embeddings
- Hierarchical classification
"""

__version__ = "0.1.0"

from .direct_mapping import direct_llm_mapping
from .summarization import generate_project_summaries
from .vector_search import vector_search_sdgs
from .hierarchical_search import classify_sdg_goal, search_sdg_targets
from .llm_judge import llm_sdg_judge
from .evaluation import evaluate_vector_search_quality

__all__ = [
    'direct_llm_mapping',
    'generate_project_summaries',
    'vector_search_sdgs',
    'classify_sdg_goal',
    'search_sdg_targets',
    'llm_sdg_judge',
    'evaluate_vector_search_quality'
]