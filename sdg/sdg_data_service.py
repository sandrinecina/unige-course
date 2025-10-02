import requests
import json

class SDGDataService:
    def __init__(self):
        self.base_url = "https://unstats.un.org/SDGAPI/v1/sdg/DataAvailability/GetIndicatorsAllCountries"
        self.headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
    
    def fetch_indicators_all_countries(self, data_point_type=1, country_id=0, nature_of_data="string"):
        """
        Fetch SDG indicators data for all countries
        
        Args:
            data_point_type (int): Type of data point (default: 1)
            country_id (int): Country ID (default: 0 for all countries)
            nature_of_data (str): Nature of data (default: "string")
        
        Returns:
            dict: API response containing indicators data
        """
        data = {
            'dataPointType': data_point_type,
            'countryId': country_id,
            'natureOfData': nature_of_data
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

# Example usage
if __name__ == "__main__":
    service = SDGDataService()
    data = service.fetch_indicators_all_countries()
    if data:
        print(json.dumps(data, indent=2))