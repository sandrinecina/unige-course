# SDG Mapping Implementation Tasks

## Overview
This document tracks the implementation tasks for the SDG mapping system described in claude.md. The system will map NGO annual reports to UN Sustainable Development Goals using multiple approaches including direct LLM mapping, vector search, and hierarchical classification.

## Implementation Tasks

### Phase 1: Core Infrastructure Setup

#### 1. Project Structure Creation
- [ ] Create `sdg_mapping/` directory
- [ ] Create `__init__.py` for package initialization
- [ ] Set up basic module files:
  - [ ] `direct_mapping.py`
  - [ ] `summarization.py`
  - [ ] `vector_search.py`
  - [ ] `hierarchical_search.py`
  - [ ] `evaluation.py`
  - [ ] `llm_judge.py`
  - [ ] `pipeline.py`
  - [ ] `utils.py` (for shared utilities)
  - [ ] `config.py` (for configuration management)

#### 2. Dependencies and Environment Setup
- [ ] Create/update `requirements.txt` with:
  - [ ] `langchain`
  - [ ] `llamaindex`
  - [ ] `llama-parse`
  - [ ] `chromadb` or alternative vector DB
  - [ ] `openai` or alternative LLM provider
  - [ ] `numpy`
  - [ ] `pandas`
  - [ ] `scikit-learn` (for metrics)
  - [ ] `pydantic` (for data models)
  - [ ] `langfuse` (for observability)
- [ ] Set up environment variables for API keys
- [ ] Configure Langfuse for tracking
- [ ] Create `.env.example` file

### Phase 2: Data Preparation

#### 3. SDG Data Processing
- [ ] Load and parse `sdg_indicators.json`
- [ ] Load existing SDG keywords from generated files (e.g., sdg_keywords_output_1.json)
- [ ] Create structured data models for SDGs
- [ ] Generate embeddings for:
  - [ ] SDG goal names (from "name" field)
  - [ ] SDG indicator descriptions (from "indicators[].description")
  - [ ] SDG keywords (from "keywords" array)
  - [ ] Combined representations (goal name + indicators + keywords)

#### 4. Vector Database Setup
- [ ] Initialize vector database (ChromaDB/Pinecone/Weaviate)
- [ ] Create collections/indexes for:
  - [ ] SDG embeddings
  - [ ] Report summaries
  - [ ] Project descriptions
- [ ] Implement embedding storage and retrieval functions

### Phase 3: Core Functionality Implementation

#### 5. Direct LLM Mapping (`direct_mapping.py`)
- [ ] Implement `direct_llm_mapping()` function with @observe decorator
- [ ] Create prompt templates for SDG identification
- [ ] Add confidence scoring logic
- [ ] Implement rationale generation
- [ ] Add error handling and retries

#### 6. Summarization Module (`summarization.py`)
- [ ] Implement `generate_project_summaries()` function with @observe decorator
- [ ] Create extraction logic for:
  - [ ] Project identification
  - [ ] Summary generation
  - [ ] Topic extraction
  - [ ] Key outcomes identification
- [ ] Add chunking strategy for long reports using LlamaParse
- [ ] Implement structured output parsing

#### 7. LLM Judge Implementation (`llm_judge.py`)
- [ ] Implement `llm_sdg_judge()` function with @observe decorator
- [ ] Create JUDGE_PROMPT_TEMPLATE
- [ ] Add relevance scoring logic
- [ ] Implement justification generation
- [ ] Add consistency checks

#### 8. Vector Search Module (`vector_search.py`)
- [ ] Implement `vector_search_sdgs()` function with @observe decorator
- [ ] Create embedding generation for summaries
- [ ] Implement similarity search
- [ ] Add ranking and filtering logic
- [ ] Optimize search parameters

#### 9. Hierarchical Search (`hierarchical_search.py`)
- [ ] Implement `classify_sdg_goal()` function with @observe decorator
- [ ] Implement `search_sdg_targets()` function with @observe decorator
- [ ] Create two-stage search pipeline
- [ ] Add confidence scoring for each stage
- [ ] Implement target filtering based on goals

### Phase 4: Evaluation and Pipeline

#### 10. Evaluation Framework (`evaluation.py`)
- [ ] Implement `evaluate_vector_search_quality()` function with @observe decorator
- [ ] Add metrics calculations:
  - [ ] Precision/Recall
  - [ ] NDCG (Normalized Discounted Cumulative Gain)
  - [ ] Coverage metrics
  - [ ] Agreement scores
- [ ] Create comparison functions between approaches
- [ ] Add statistical significance tests

#### 11. Main Pipeline (`pipeline.py`)
- [ ] Implement main orchestration function with @observe decorator
- [ ] Add parallel processing for approaches
- [ ] Create results aggregation logic
- [ ] Implement confidence weighting
- [ ] Add progress tracking and logging
- [ ] Integrate Langfuse traces for debugging and monitoring
- [ ] Create final output formatting

### Phase 5: Integration and Testing

#### 12. Integration with Existing Code
- [ ] Connect with `sdg_data_service.py`
- [ ] Integrate with `agent.py` functionality
- [ ] Use existing SDG keywords from generated files
- [ ] Leverage PDF processing from `pdf_extraction_ui.py`

#### 13. Test Suite Development
- [ ] Create test data:
  - [ ] Sample reports with known SDG mappings
  - [ ] Edge cases (ambiguous mappings)
  - [ ] Multi-SDG projects
- [ ] Write unit tests for each module
- [ ] Create integration tests
- [ ] Add performance benchmarks

#### 14. Documentation and Examples
- [ ] Create usage examples
- [ ] Document API interfaces
- [ ] Add configuration guide
- [ ] Create troubleshooting guide

### Phase 6: Optimization and Deployment

#### 15. Performance Optimization
- [ ] Profile code for bottlenecks
- [ ] Optimize embedding generation
- [ ] Implement caching strategies
- [ ] Add batch processing capabilities
- [ ] Optimize vector search parameters

#### 16. Results and Reporting
- [ ] Create visualization tools for results
- [ ] Implement export functions (JSON, CSV, PDF)
- [ ] Add comparison dashboards
- [ ] Create confidence score visualizations

## Priority Order

### High Priority (Start Immediately)
1. Project structure creation
2. Direct LLM mapping implementation
3. Summarization module
4. Vector database setup
5. SDG embeddings creation
6. LLM judge implementation
7. Basic evaluation metrics
8. Main pipeline orchestration

### Medium Priority (After Core Features)
1. Hierarchical search implementation
2. Advanced evaluation metrics
3. Integration with existing code
4. Test suite development
5. Results aggregation logic

### Low Priority (Nice to Have)
1. Visualization tools
2. Advanced reporting features
3. Performance optimizations
4. Comprehensive documentation

## Success Criteria

1. **Accuracy**: System should achieve >80% agreement with human SDG mappings
2. **Performance**: Process a full annual report in <2 minutes
3. **Coverage**: Identify all relevant SDGs (high recall)
4. **Precision**: Minimize false positive SDG assignments
5. **Explainability**: Provide clear justifications for each mapping

## Next Steps

1. Set up the project structure and install dependencies
2. Begin with direct LLM mapping as the baseline
3. Implement summarization to enable other approaches
4. Build evaluation framework early to track progress
5. Iterate based on evaluation results

## Notes

- Start with a simple implementation and iterate
- Use existing SDG data from `sdg_indicators.json`
- Leverage the agent framework from `04-Agents-exercise/`
- Use Langfuse @observe decorators for all main functions to track performance and debug issues
- Consider using Langfuse for tracking experiments (as mentioned in NOTES.md)
- Follow the RAG best practices outlined in NOTES.md for vector search implementation