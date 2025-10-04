#!/usr/bin/env python3
"""
Granular SDG mapping using OpenAI to map text to specific SDG indicators
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
    
    # Create a detailed mapping including all indicators
    sdg_info = {}
    all_indicators = []
    
    for goal in data:
        goal_id = int(goal['goalId'])
        indicators = []
        
        for ind in goal['indicators']:
            indicator = {
                'code': ind['code'],
                'description': ind['description'],
                'goal_id': goal_id,
                'goal_name': goal['goalName']
            }
            indicators.append(indicator)
            all_indicators.append(indicator)
        
        sdg_info[goal_id] = {
            'name': goal['goalName'],
            'indicators': indicators
        }
    
    return sdg_info, all_indicators

def map_text_to_indicators(text, sdg_info, all_indicators):
    """Map text to specific SDG indicators using OpenAI"""
    
    # Create a structured list of indicators
    indicators_text = ""
    for sdg_num, info in sdg_info.items():
        indicators_text += f"\nSDG {sdg_num}: {info['name']}\n"
        for ind in info['indicators']:
            indicators_text += f"  [{ind['code']}] {ind['description']}\n"
    
    prompt = f"""Given this text from an NGO report:

{text}

Analyze which specific SDG indicators this text relates to. Consider the following indicators:

{indicators_text}

For each relevant indicator:
1. Provide the indicator code (e.g., "6.1.1")
2. Give a relevance score (0-1) 
3. Explain specifically how the text relates to this indicator

Format as JSON array like: 
[{{"indicator": "6.1.1", "relevance": 0.9, "explanation": "The text directly mentions providing clean drinking water access, which is the core of indicator 6.1.1"}}]

Be specific and only include indicators that are clearly relevant."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in UN SDG indicators. Be precise in matching text to specific indicators."},
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

def display_results(mappings, sdg_info, all_indicators):
    """Display the mapping results in a structured way"""
    
    # Create a lookup for indicators
    indicator_lookup = {ind['code']: ind for ind in all_indicators}
    
    # Group by SDG
    sdg_mappings = {}
    for mapping in mappings:
        ind_code = mapping['indicator']
        if ind_code in indicator_lookup:
            ind_info = indicator_lookup[ind_code]
            goal_id = ind_info['goal_id']
            
            if goal_id not in sdg_mappings:
                sdg_mappings[goal_id] = []
            
            sdg_mappings[goal_id].append({
                'code': ind_code,
                'description': ind_info['description'],
                'relevance': mapping['relevance'],
                'explanation': mapping['explanation']
            })
    
    # Display results
    for goal_id in sorted(sdg_mappings.keys()):
        print(f"\n{'='*60}")
        print(f"SDG {goal_id}: {sdg_info[goal_id]['name']}")
        print(f"{'='*60}")
        
        for ind in sorted(sdg_mappings[goal_id], key=lambda x: x['relevance'], reverse=True):
            print(f"\nIndicator {ind['code']} (Relevance: {ind['relevance']:.2f})")
            print(f"Description: {ind['description']}")
            print(f"Match reason: {ind['explanation']}")

# Example usage
if __name__ == "__main__":
    # Load SDG data
    print("Loading SDG data...")
    sdg_info, all_indicators = load_sdg_data()
    print(f"Loaded {len(sdg_info)} SDGs with {len(all_indicators)} indicators\n")
    
    # Sample texts with different focus areas
    sample_texts = [
        """
        Our water access program has achieved remarkable results:
        - Installed 50 new water wells providing safe drinking water to 25,000 people
        - 95% of households now have access to safely managed drinking water services
        - Water quality testing shows 0% contamination in all new wells
        - Reduced time spent collecting water from 3 hours to 30 minutes daily
        """,
        """
        Our women's empowerment initiative focuses on:
        - Training 500 women in leadership roles in local government
        - 40% of local council positions are now held by women
        - Established legal aid centers to combat gender-based violence
        - Provided land ownership documentation to 200 women farmers
        """
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{'#'*60}")
        print(f"SAMPLE TEXT {i}:")
        print(text.strip())
        print(f"\nMapping to specific SDG indicators...")
        
        # Get mappings
        mappings = map_text_to_indicators(text, sdg_info, all_indicators)
        
        if isinstance(mappings, list):
            display_results(mappings, sdg_info, all_indicators)
        else:
            print("Error parsing results:")
            print(mappings)
        
        print(f"\n{'#'*60}\n")