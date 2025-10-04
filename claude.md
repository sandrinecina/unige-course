# SDG Mapping Implementation Guide

## Overview

This guide outlines a multi-approach strategy for mapping NGOs annual reports to UN Sustainable Development Goals (SDGs) using LLMs and vector search.

## Implementation Approaches

### Approach 1: Direct LLM Mapping

**Objective**: Use LLM to directly output SDG mappings from the entire report.

```python
# Task: Direct SDG Mapping
# Input: Full report text
# Output: List of applicable SDGs with confidence scores

def direct_llm_mapping(report_text, sdg_descriptions):
    """
    Directly map report to SDGs using LLM analysis.

    Args:
        report_text: Complete report content
        sdg_descriptions: Dictionary of SDG numbers to descriptions

    Returns:
        List of (sdg_number, confidence_score, rationale)
    """
    prompt = f"""
    Given this report:
    {report_text}

    And these SDG descriptions:
    {sdg_descriptions}

    Identify which SDGs this report relates to.
    Return mappings with confidence scores (0-1) and brief rationales.
    """
    # LLM processing here
    pass
```

### Approach 2: Summary-Based Vector Search

#### Step 1: Report to Summary with Topics

```python
# Task: Generate project summaries and extract topics
# Input: Report text
# Output: Structured summaries with topic lists

def generate_project_summaries(report_text):
    """
    Extract project summaries and associated topics from report.

    Returns:
        List of {
            'project_id': str,
            'summary': str,
            'topics': List[str],
            'key_outcomes': List[str]
        }
    """
    pass
```

#### Step 2: LLM as Judge for SDG Assignment

```python
# Task: Create evaluator to assess SDG assignments
# Input: Project summary
# Output: SDG assignments with justification

def llm_sdg_judge(summary, topics):
    """
    Use LLM to judge which SDGs apply to a project summary.

    Returns:
        List of {
            'sdg': int,
            'relevance_score': float,
            'justification': str
        }
    """
    pass
```

#### Step 3: Vector Search Implementation

```python
# Task: Embed summaries and perform vector search against SDG embeddings
# Input: Summaries + topics
# Output: Similar SDGs ranked by relevance

def vector_search_sdgs(summary_with_topics, sdg_embeddings):
    """
    Find similar SDGs using vector search.

    Args:
        summary_with_topics: Combined text of summary and topics
        sdg_embeddings: Pre-computed embeddings of SDG descriptions

    Returns:
        List of (sdg_number, similarity_score)
    """
    pass
```

### Approach 3: Hierarchical Vector Search

#### Two-Stage Search Process

```python
# Stage 1: Goal-level classification
def classify_sdg_goal(summary):
    """
    First determine which main SDG goal(s) the project relates to.

    Returns:
        List of relevant SDG goals with confidence scores
    """
    pass

# Stage 2: Target-level search within identified goals
def search_sdg_targets(summary, identified_goals):
    """
    Within identified goals, search for specific targets.

    Returns:
        List of specific SDG targets with relevance scores
    """
    pass
```

## Evaluation Framework

### Vector Search Evaluator

```python
def evaluate_vector_search_quality(
    predicted_sdgs,
    llm_assigned_sdgs,
    summary
):
    """
    Evaluate how well vector search results match LLM judgments.

    Metrics:
    - Precision/Recall against LLM assignments
    - Ranking quality (NDCG)
    - Coverage of relevant SDGs
    """
    pass
```

## Implementation Pipeline

1. **Data Preparation**

   - Load report
   - Prepare SDG descriptions and embeddings
   - Set up evaluation framework

2. **Processing Pipeline**

   ```python
   # Main pipeline
   report = load_report()

   # Generate summaries
   summaries = generate_project_summaries(report)

   # Run parallel approaches
   direct_mappings = direct_llm_mapping(report, sdg_descriptions)

   for summary in summaries:
       # LLM judgment
       llm_sdgs = llm_sdg_judge(summary['summary'], summary['topics'])

       # Vector search
       vector_sdgs = vector_search_sdgs(
           summary['summary'] + ' ' + ' '.join(summary['topics']),
           sdg_embeddings
       )

       # Hierarchical search
       goals = classify_sdg_goal(summary['summary'])
       targets = search_sdg_targets(summary['summary'], goals)

       # Evaluate approaches
       evaluation = evaluate_vector_search_quality(
           vector_sdgs,
           llm_sdgs,
           summary
       )
   ```

3. **Results Aggregation**
   - Compare results across approaches
   - Generate confidence-weighted final mappings
   - Produce evaluation report

## Key Considerations

### Embedding Strategy

- Embed SDG titles, descriptions, and targets separately
- Create combined embeddings for comprehensive search
- Consider domain-specific fine-tuning

### Prompt Engineering for LLM Judge

```python
JUDGE_PROMPT_TEMPLATE = """
You are an expert in UN Sustainable Development Goals.

Given this project summary:
{summary}

Topics: {topics}

Evaluate which SDGs this project contributes to.
Consider both direct and indirect contributions.
Provide relevance scores (0-1) and clear justifications.
"""
```

### Hierarchical Search Benefits

- Reduces search space at each level
- Improves precision by constraining to relevant goals first
- Allows for more nuanced target-level matching

## Testing and Validation

1. **Test Cases**

   - Projects with clear single-SDG alignment
   - Projects spanning multiple SDGs
   - Edge cases with indirect SDG relationships

2. **Metrics to Track**

   - Agreement between approaches
   - LLM judge consistency
   - Vector search recall@k
   - Processing time per report

3. **Iterative Improvement**
   - Fine-tune embeddings based on evaluation results
   - Adjust LLM prompts for better judgment
   - Optimize vector search parameters

## Next Steps

1. Implement baseline direct LLM approach
2. Build summary generation pipeline
3. Create SDG embedding database
4. Develop evaluation framework
5. Run comparative analysis across approaches
6. Select optimal approach or ensemble method

## Code Structure

```
sdg_mapping/
├── __init__.py
├── direct_mapping.py      # Direct LLM approach
├── summarization.py       # Summary generation
├── vector_search.py       # Vector search implementation
├── hierarchical_search.py # Two-stage search
├── evaluation.py          # Evaluation metrics
├── llm_judge.py          # LLM as evaluator
└── pipeline.py           # Main orchestration
```
