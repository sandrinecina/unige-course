#!/usr/bin/env python3
"""
Simple SDG mapping using OpenAI to map text to SDG goals
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_sdg_data():
    """Load SDG indicators from JSON file"""
    with open('sdg/sdg_indicators.json', 'r') as f:
        data = json.load(f)
    
    # Create a simple mapping of SDG number to name and description
    sdg_info = {}
    for goal in data:
        goal_id = int(goal['goalId'])
        sdg_info[goal_id] = {
            'name': goal['goalName'],
            'indicator_count': len(goal['indicators']),
            'sample_indicators': [ind['description'] for ind in goal['indicators'][:3]]
        }
    
    return sdg_info

def map_text_to_sdgs(text, sdg_info):
    """Map a text to relevant SDGs using OpenAI"""
    
    # Create a simple description of each SDG
    sdg_descriptions = ""
    for num, info in sdg_info.items():
        sdg_descriptions += f"\nSDG {num}: {info['name']}"
        sdg_descriptions += f"\n  Examples: {'; '.join(info['sample_indicators'][:2])}\n"
    
    prompt = f"""Given this text from an NGO report:

{text}

And these UN Sustainable Development Goals:
{sdg_descriptions}

Which SDGs does this text relate to? For each relevant SDG:
1. List the SDG number
2. Give a confidence score (0-1)
3. Provide a brief explanation

Format as JSON array like: [{{"sdg": 6, "confidence": 0.9, "reason": "..."}}]"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in UN Sustainable Development Goals."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    try:
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
    except:
        # If JSON parsing fails, return raw response
        return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    # Load SDG data
    print("Loading SDG data...")
    sdg_info = load_sdg_data()
    print(f"Loaded {len(sdg_info)} SDGs\n")
    
    # Sample text
    sample_text = """
    Our organization has successfully implemented clean water access programs in 
    rural communities across Sub-Saharan Africa. This year, we:
    - Installed 50 new water wells providing clean drinking water to 25,000 people
    - Trained 200 local technicians in water system maintenance
    - Reduced waterborne diseases by 60% in target communities
    """
    
    print("Sample text:")
    print(sample_text)
    print("\nMapping to SDGs...")
    
    # Get mappings
    mappings = map_text_to_sdgs(sample_text, sdg_info)
    
    print("\nResults:")
    if isinstance(mappings, list):
        for mapping in mappings:
            sdg_num = mapping['sdg']
            print(f"\nSDG {sdg_num}: {sdg_info[sdg_num]['name']}")
            print(f"Confidence: {mapping['confidence']}")
            print(f"Reason: {mapping['reason']}")
    else:
        print(mappings)