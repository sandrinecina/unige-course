"""
Configuration Module

This module provides configuration settings for the SDG mapping system.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field, validator


class SDGMappingConfig(BaseSettings):
    """Configuration for SDG Mapping system"""
    
    # API Keys and Secrets (from environment)
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    llama_parse_api_key: str = Field(default="", env="LLAMA_PARSE_API_KEY")
    langfuse_public_key: str = Field(default="", env="LANGFUSE_PUBLIC_KEY", description="LF_PUBLIC_KEY also supported")
    langfuse_secret_key: str = Field(default="", env="LANGFUSE_SECRET_KEY", description="LF_SECRET_KEY also supported")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")
    
    # LLM settings (configurable defaults)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    
    # Vector database settings (configurable defaults)
    vector_db_type: str = "chromadb"
    vector_db_path: str = "./chroma_db"
    embedding_model: str = "text-embedding-ada-002"
    
    # Processing settings (application behavior)
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_retries: int = 3
    max_report_length: int = 10000  # Characters before chunking
    
    # Evaluation settings (application behavior)
    confidence_threshold: float = 0.5
    top_k_results: int = 5
    
    # Model parameters
    embedding_dimension: int = 1536  # For text-embedding-ada-002
    similarity_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @validator('langfuse_public_key', pre=True)
    def get_langfuse_public_key(cls, v):
        if not v:
            # Check for alternative env var name
            v = os.getenv('LF_PUBLIC_KEY', '')
        return v
    
    @validator('langfuse_secret_key', pre=True)
    def get_langfuse_secret_key(cls, v):
        if not v:
            # Check for alternative env var name
            v = os.getenv('LF_SECRET_KEY', '')
        return v


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