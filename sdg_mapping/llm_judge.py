"""
LLM Judge Module

This module provides functionality for using an LLM as a judge to evaluate
which SDGs apply to a project summary.
"""

from typing import List, Dict
from langfuse import observe


JUDGE_PROMPT_TEMPLATE = """
You are an expert in UN Sustainable Development Goals.

Given this project summary:
{summary}

Topics: {topics}

Evaluate which SDGs this project contributes to.
Consider both direct and indirect contributions.
Provide relevance scores (0-1) and clear justifications.
"""


@observe()
def llm_sdg_judge(summary: str, topics: List[str]) -> List[Dict[str, any]]:
    """
    Use LLM to judge which SDGs apply to a project summary.
    
    Args:
        summary: Project summary text
        topics: List of topics associated with the project
        
    Returns:
        List of dictionaries containing:
            - sdg: SDG number (int)
            - relevance_score: Score from 0 to 1 (float)
            - justification: Explanation for the assignment (str)
    """
    # TODO: Implement LLM judge logic
    pass