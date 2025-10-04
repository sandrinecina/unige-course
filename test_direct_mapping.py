#!/usr/bin/env python3
"""
Test script for direct mapping functionality
"""

import os
from dotenv import load_dotenv
from sdg_mapping.utils import load_all_sdg_data, create_sdg_descriptions
from sdg_mapping.direct_mapping import direct_llm_mapping

# Load environment variables
load_dotenv()

def test_direct_mapping():
    """Test the direct mapping functionality with sample text"""
    
    # Load SDG data
    print("Loading SDG data...")
    sdg_data = load_all_sdg_data()
    sdg_descriptions = create_sdg_descriptions(sdg_data)
    
    print(f"Loaded {len(sdg_data)} SDGs")
    
    # Sample report text (you can replace with actual report content)
    sample_report = """
    Annual Report 2023 - Water for All Initiative
    
    Our organization has successfully implemented clean water access programs in 
    rural communities across Sub-Saharan Africa. This year, we:
    
    1. Installed 50 new water wells providing clean drinking water to 25,000 people
    2. Trained 200 local technicians in water system maintenance
    3. Reduced waterborne diseases by 60% in target communities
    4. Provided sanitation facilities to 15 schools, benefiting 3,000 students
    5. Empowered women by reducing water collection time by 75%
    
    Our education programs have also expanded, with new schools built in 
    underserved areas and teacher training programs reaching 500 educators.
    
    Environmental sustainability remains a core focus, with reforestation 
    projects planting 10,000 trees near water sources to prevent erosion.
    """
    
    print("\nTesting direct LLM mapping...")
    print("Sample report preview:")
    print(sample_report[:200] + "...")
    
    try:
        # Run direct mapping
        results = direct_llm_mapping(sample_report, sdg_descriptions)
        
        print(f"\nFound {len(results)} SDG mappings:")
        print("-" * 60)
        
        for sdg_num, confidence, rationale in results[:5]:  # Show top 5
            print(f"\nSDG {sdg_num}: {sdg_data[sdg_num]['name']}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Rationale: {rationale[:200]}...")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Make sure you have set the following environment variables:")
        print("- OPENAI_API_KEY")
        print("- LANGFUSE_PUBLIC_KEY (optional)")
        print("- LANGFUSE_SECRET_KEY (optional)")


if __name__ == "__main__":
    test_direct_mapping()