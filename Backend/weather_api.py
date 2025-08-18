import requests
from typing import Dict, Any, Optional, List
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

class WeatherAPI:
    def __init__(self, base_url: str = "https://api.open-meteo.com/v1/forecast"):
        self.base_url = base_url
        self.default_location = {"latitude": 28.6139, "longitude": 77.2090}  # Delhi
        self.mandi_base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        self.mandi_api_key = os.getenv("MANDI_API_KEY")  # make sure to set in .env

    def _get_nearby_states(self, latitude: float, longitude: float) -> List[str]:
        """Return nearby states based on rough lat/lon mapping"""
        # Minimal hardcoded mapping for demo, can expand
        if 27 <= latitude <= 29 and 76 <= longitude <= 78:
            return ["DELHI", "HARYANA", "UTTAR PRADESH", "RAJASTHAN"]
        elif 18 <= latitude <= 20 and 73 <= longitude <= 76:
            return ["MAHARASHTRA", "MADHYA PRADESH", "KARNATAKA"]
        else:
            return ["DELHI"]  # fallback

    def _fetch_mandi_prices(self, states: List[str]) -> str:
        """Fetch mandi price info for given states"""
        if not self.mandi_api_key:
            return "\n[Mandi Data Error: No API Key set in environment]\n"

        mandi_info = "\nNEARBY STATE MANDI PRICES (Real-time from data.gov.in API):\n"
        for state in states:
            try:
                params = {
                    "format": "json",
                    "limit": 5,
                    "api-key": self.mandi_api_key,
                    "filters[state]": state
                }
                resp = requests.get(self.mandi_base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

                records = data.get("records", [])
                if not records:
                    mandi_info += f"{state}: No mandi price data found.\n"
                    continue

                mandi_info += f"\n{state}:\n"
                for rec in records:
                    commodity = rec.get("commodity", "N/A")
                    variety = rec.get("variety", "N/A")
                    min_price = rec.get("min_price", "N/A")
                    max_price = rec.get("max_price", "N/A")
                    modal_price = rec.get("modal_price", "N/A")
                    mandi_info += f"  - {commodity} ({variety}): Min {min_price}, Max {max_price}, Modal {modal_price}\n"

            except Exception as e:
                mandi_info += f"{state}: Error fetching mandi data ({e})\n"

        return mandi_info

    async def get_current_weather_context(self, latitude: float = None, longitude: float = None) -> str:
        """Get current weather + mandi price data as context string for LLM"""
        try:
            # Weather params
            params = {
                "latitude": latitude or self.default_location["latitude"],
                "longitude": longitude or self.default_location["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum",
                "current_weather": "true",
                "forecast_days": 7,
                "timezone": "auto"
            }

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            weather_data = response.json()

            current = weather_data.get("current_weather", {})
            daily = weather_data.get("daily", {})

            # --- WEATHER OUTPUT (unchanged) ---
            context = f"""
CURRENT WEATHER DATA (Delhi, Real-time from Open-Meteo API):
Current Temperature: {current.get('temperature', 'N/A')}°C
Current Wind Speed: {current.get('windspeed', 'N/A')} km/h
Current Weather Code: {current.get('weathercode', 'N/A')}

7-Day Forecast:
"""
            for i, (date, tmax, tmin, precip) in enumerate(zip(
                daily.get("time", [])[:7], 
                daily.get("temperature_2m_max", [])[:7], 
                daily.get("temperature_2m_min", [])[:7],
                daily.get("precipitation_sum", [])[:7]
            )):
                day_label = "Today" if i == 0 else f"Day +{i}"
                context += f"{day_label} ({date}): Max {tmax}°C, Min {tmin}°C, Precipitation {precip}mm\n"

            context += "\nNote: This is CURRENT/FUTURE weather data only. For historical weather data, this API cannot provide past information.\n"

            # --- ADD MANDI PRICES HERE ---
            states = self._get_nearby_states(latitude or self.default_location["latitude"], 
                                             longitude or self.default_location["longitude"])
            context += self._fetch_mandi_prices(states)

            return context

        except Exception as e:
            return f"Weather API Error: {str(e)}. Current weather data unavailable."
