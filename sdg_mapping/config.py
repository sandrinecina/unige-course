"""
Configuration Module

This module provides configuration settings for the SDG mapping system.
"""

import os
from typing import Dict, Any
from pydantic import BaseSettings, Field


class SDGMappingConfig(BaseSettings):
    """Configuration for SDG Mapping system"""
    
    # LLM settings
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Vector database settings
    vector_db_type: str = Field(default="chromadb", env="VECTOR_DB_TYPE")
    vector_db_path: str = Field(default="./chroma_db", env="VECTOR_DB_PATH")
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    
    # LlamaParse settings
    llama_parse_api_key: str = Field(default="", env="LLAMA_PARSE_API_KEY")
    
    # Langfuse settings
    langfuse_public_key: str = Field(default="", env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")
    
    # Processing settings
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # Evaluation settings
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_config() -> SDGMappingConfig:
    """Get configuration instance"""
    return SDGMappingConfig()


# Default prompts
PROMPTS = {
    "direct_mapping": """Given this report:
{report_text}

And these SDG descriptions:
{sdg_descriptions}

Identify which SDGs this report relates to.
Return mappings with confidence scores (0-1) and brief rationales.

Format your response as a JSON array with objects containing:
- sdg_number: integer
- confidence_score: float between 0 and 1
- rationale: brief explanation
""",
    
    "summarization": """Extract distinct projects from this report:
{report_text}

For each project, provide:
- A unique project identifier
- A concise summary (100-200 words)
- Key topics/themes (as a list)
- Main outcomes or impacts

Format as JSON array of project objects.
""",
}