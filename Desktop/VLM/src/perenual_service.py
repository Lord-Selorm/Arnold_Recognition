import requests
import os
import json

class PerenualService:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("PLANT_API_KEY")
        self.base_url = "https://perenual.com/api"
        self._cache = {}

    def get_disease_details(self, disease_name):
        """
        Fetch disease details from Perenual API based on a search query.
        Since Perenual uses a list endpoint, we first search/filter for the disease.
        """
        if not self.api_key:
            return None

        # Clean up the name for search (e.g., 'Apple___Apple_scab' -> 'Apple Scab')
        query = disease_name.replace("___", " ").replace("_", " ").split("(")[0].strip()
        
        # Check cache
        if query in self._cache:
            return self._cache[query]

        try:
            # Step 1: Search for disease ID (Using the list endpoint)
            # Perenual doesn't have a direct 'search' for diseases documented publicly similarly to species,
            # but we can try the pest-disease-list and filter in memory or via q param if supported.
            # We will fetch a generic list and match locally to start, or query if possible.
            # Documentation implies generic list.
            
            url = f"{self.base_url}/pest-disease-list?key={self.api_key}&q={query}"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('data', [])
                
                # Find best match
                for item in results:
                    common_name = item.get('common_name', '').lower()
                    if query.lower() in common_name:
                        # Found a likely match. In a real full integration, we'd fetch details by ID.
                        # For this specific API tier, the list might contain the info we need.
                        
                        detail = {
                            'description': item.get('description', 'No detailed description available from external database.'),
                            'treatment': item.get('solution') or item.get('treatment', []),
                            # Normalize treatment to list if it's a string
                        }
                        
                        if isinstance(detail['treatment'], str):
                             detail['treatment'] = [t.strip() for t in detail['treatment'].split('.') if t.strip()]
                        
                        self._cache[query] = detail
                        return detail
            
            return None

        except Exception as e:
            print(f"Perenual API Error: {e}")
            return None
