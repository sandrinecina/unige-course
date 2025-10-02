#!/usr/bin/env python3
"""
Script to fetch SDG data from API and save it locally
Run this once to create the sdg_indicators.json file
"""

from sdg_data_service import SDGDataService
import json

def fetch_and_save():
    service = SDGDataService()
    print("Fetching SDG data from API...")
    
    data = service.fetch_indicators_all_countries()
    
    if data:
        with open('sdg_indicators.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} SDG goals to sdg_indicators.json")
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    fetch_and_save()