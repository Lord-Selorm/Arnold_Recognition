import requests
import json

API_KEY = "sk-7LiM695f79b64eb8814239"

def check_disease_list():
    """Check if we can list diseases (Data API)"""
    url = f"https://perenual.com/api/pest-disease-list?key={API_KEY}"
    try:
        print(f"Checking Data API: {url}...")
        resp = requests.get(url)
        if resp.status_code == 200:
            print("✅ Data API Success!")
            data = resp.json()
            print(json.dumps(data, indent=2)[:500] + "...")
            return True
        else:
            print(f"❌ Data API Failed: {resp.status_code}")
            print(resp.text)
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Verifying Perenual API Key...")
    check_disease_list()
