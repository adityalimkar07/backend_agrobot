import requests
from typing import Dict, Any, Optional, List
import asyncio
import os
from dotenv import load_dotenv
from geopy.geocoders import Nominatim

load_dotenv()
# Ensure you have MANDI_API_KEY set in your .env file

class WeatherAPI:
    def __init__(self, base_url: str = "https://api.open-meteo.com/v1/forecast"):
        self.base_url = base_url
        self.default_location = {"latitude": 19.8762, "longitude": 75.3433}  # Aurangabad, MH
        self.mandi_base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        self.mandi_api_key = os.getenv("MANDI_API_KEY")  # make sure to set in .env
        self.geolocator = Nominatim(user_agent="mandi_locator")

    def _get_location_name(self, latitude: float, longitude: float) -> str:
        """
        # NEW: Get a user-friendly location name (e.g., City, State) from coordinates.
        """
        try:
            location = self.geolocator.reverse((latitude, longitude), language="en")
            if not location or not location.raw.get("address"):
                return "Unknown Location"
            
            address = location.raw.get("address", {})
            # Prioritize city, then town/village, then state
            city = address.get("city", address.get("town", address.get("village")))
            state = address.get("state")
            
            if city and state:
                return f"{city}, {state}"
            return state or city or "Unknown Location"
        except Exception:
            return "Unknown Location"

    def _reverse_geocode(self, latitude: float, longitude: float) -> Optional[str]:
        """Get state name from coordinates using reverse geocoding"""
        try:
            location = self.geolocator.reverse((latitude, longitude), language="en")
            if not location:
                return None
            address = location.raw.get("address", {})
            return address.get("state")  # e.g., "Maharashtra"
        except Exception:
            return None

    def _get_nearby_states(self, latitude: float, longitude: float) -> List[str]:
        """Get state + neighboring states via shifted coordinates"""
        states = set()
        shifts = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]  # center, N, S, E, W
        for dlat, dlon in shifts:
            state = self._reverse_geocode(latitude + dlat, longitude + dlon)
            if state:
                states.add(state.upper())
        # # REMOVED: Fallback to Delhi is no longer necessary if coordinates are always provided
        return list(states)

    def _fetch_mandi_prices(self, states: List[str]) -> str:
        """Fetch mandi price info for given states"""
        if not self.mandi_api_key:
            return "\n[Mandi Data Error: No API Key set in environment]\n"

        if not states:
            return "\nNEARBY STATE MANDI PRICES: Could not determine state from coordinates.\n"

        mandi_info = "\nNEARBY STATE MANDI PRICES (Real-time from data.gov.in API):\n"
        for state in states:
            try:
                params = {
                    "format": "json",
                    "limit": 10,
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
                    state_name = rec.get("state", "N/A")
                    district = rec.get("district", "N/A")
                    market = rec.get("market", "N/A")
                    commodity = rec.get("commodity", "N/A")
                    variety = rec.get("variety", "N/A")
                    grade = rec.get("grade", "N/A")
                    arrival_date = rec.get("arrival_date", "N/A")
                    min_price = rec.get("min_price", "N/A")
                    max_price = rec.get("max_price", "N/A")
                    modal_price = rec.get("modal_price", "N/A")

                    mandi_info += (
                        f"  - On {arrival_date}, in {district} district of {state_name} "
                        f"(market: {market}), {commodity} (variety: {variety}, grade: {grade}) "
                        f"was sold at a minimum of {min_price} Rs/100kg, "
                        f"maximum of {max_price} Rs/100kg, "
                        f"with a modal price of {modal_price} Rs/100kg.\n"
                    )

            except Exception as e:
                mandi_info += f"{state}: Error fetching mandi data ({e})\n"

        return mandi_info

    async def get_current_weather_context(self, latitude: float, longitude: float) -> str:
        """
        # CHANGED: This method now requires latitude and longitude.
        Get current weather + mandi price data as context string for LLM.
        """
        try:
            # Weather params
            params = {
                "latitude": latitude,
                "longitude": longitude,
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

            # --- DYNAMIC LOCATION NAME ---
            # CHANGED: Get location name from coordinates and remove hardcoded "Delhi".
            location_name = self._get_location_name(latitude, longitude)
            
            context = f"""
CURRENT WEATHER DATA ({location_name}, Real-time from Open-Meteo API):
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
            states = self._get_nearby_states(latitude, longitude)
            context += self._fetch_mandi_prices(states)

            return context

        except Exception as e:
            return f"Weather API Error: {str(e)}. Current weather data unavailable."