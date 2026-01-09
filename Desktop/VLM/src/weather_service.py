import requests
import os
from datetime import datetime

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv('WEATHER_API_KEY', 'demo_key')
        self.base_url = 'http://api.openweathermap.org/data/2.5'
    
    def get_weather_data(self, lat=40.7128, lon=-74.0060):  # Default to NYC
        """Get current weather data for disease risk assessment"""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(f"{self.base_url}/weather", params=params)
            if response.status_code == 200:
                data = response.json()
                return self._analyze_weather_for_disease_risk(data)
            else:
                return self._get_demo_weather_data()
        except:
            return self._get_demo_weather_data()
    
    def _analyze_weather_for_disease_risk(self, weather_data):
        """Analyze weather conditions for disease risk factors"""
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data.get('wind', {}).get('speed', 0)
        weather_condition = weather_data['weather'][0]['main'].lower()
        
        risk_factors = {
            'temperature': self._assess_temperature_risk(temp),
            'humidity': self._assess_humidity_risk(humidity),
            'wind': self._assess_wind_risk(wind_speed),
            'conditions': self._assess_weather_conditions_risk(weather_condition)
        }
        
        overall_risk = sum(risk_factors.values()) / len(risk_factors)
        
        return {
            'current_temp': temp,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'conditions': weather_condition,
            'risk_factors': risk_factors,
            'overall_disease_risk': overall_risk,
            'recommendations': self._generate_weather_recommendations(risk_factors, weather_condition)
        }
    
    def _assess_temperature_risk(self, temp):
        """Assess disease risk based on temperature"""
        if 15 <= temp <= 25:
            return 0.8  # Optimal for fungal diseases
        elif 10 <= temp < 15 or 25 < temp <= 30:
            return 0.5  # Moderate risk
        else:
            return 0.2  # Low risk
    
    def _assess_humidity_risk(self, humidity):
        """Assess disease risk based on humidity"""
        if humidity >= 70:
            return 0.9  # High risk for fungal diseases
        elif 50 <= humidity < 70:
            return 0.6  # Moderate risk
        else:
            return 0.3  # Lower risk
    
    def _assess_wind_risk(self, wind_speed):
        """Assess disease spread risk based on wind"""
        if wind_speed > 15:
            return 0.8  # High spread risk
        elif 5 <= wind_speed <= 15:
            return 0.5  # Moderate spread risk
        else:
            return 0.2  # Low spread risk
    
    def _assess_weather_conditions_risk(self, conditions):
        """Assess risk based on weather conditions"""
        high_risk_conditions = ['rain', 'drizzle', 'mist', 'fog']
        moderate_risk_conditions = ['clouds', 'haze']
        
        if conditions in high_risk_conditions:
            return 0.8
        elif conditions in moderate_risk_conditions:
            return 0.5
        else:
            return 0.2
    
    def _generate_weather_recommendations(self, risk_factors, conditions):
        """Generate recommendations based on weather analysis"""
        recommendations = []
        
        if risk_factors['humidity'] > 0.7:
            recommendations.append("High humidity increases fungal disease risk - ensure good air circulation")
        
        if risk_factors['temperature'] > 0.7:
            recommendations.append("Optimal temperature for disease development - monitor plants closely")
        
        if risk_factors['wind'] > 0.6:
            recommendations.append("Windy conditions may spread disease - consider protective measures")
        
        if 'rain' in conditions:
            recommendations.append("Rainy conditions promote disease - avoid overhead watering")
        
        return recommendations
    
    def _get_demo_weather_data(self):
        """Get demo weather data when API is unavailable"""
        return {
            'current_temp': 22.5,
            'humidity': 75,
            'wind_speed': 8,
            'conditions': 'clouds',
            'risk_factors': {
                'temperature': 0.8,
                'humidity': 0.9,
                'wind': 0.5,
                'conditions': 0.5
            },
            'overall_disease_risk': 0.675,
            'recommendations': [
                "High humidity increases fungal disease risk - ensure good air circulation",
                "Optimal temperature for disease development - monitor plants closely",
                "Cloudy conditions promote disease - avoid overhead watering"
            ]
        }
