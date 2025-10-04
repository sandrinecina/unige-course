#!/usr/bin/env python3
"""
Simple script to extract PDF text to file
"""

import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add path for imports
sys.path.append('sdg_mapping_simple')

from sdg_mapping_simple.clients import get_mistral_document_ai_client

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_simple.py <pdf_url> [output_file]")
        sys.exit(1)
    
    pdf_url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_text.txt"
    
    print(f"Extracting from: {pdf_url}")
    
    try:
        with get_mistral_document_ai_client() as client:
            saved_file = client.extract_text_to_file(pdf_url, output_file)
            print(f"\n✅ Success! Text saved to: {saved_file}")
            print("\nYou can now use this file with:")
            print(f"python simple_sdg_mapping_indicators.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()