# README

## Consumption
File: Actual_consumption_201701010000_202601010000_Quarterhour 

Source: https://www.smard.de/en/downloadcenter/download-market-data/

Timeframe: 01.01.2017 - 01.01.2026, Resolution = 15min

Region: TransnetBW, Baden-Württemberg

Key Variables: 
- "grid load [MWh]"

## Price data
File: Day-ahead_prices_201701010000_202601010000_Quarterhour

Source: https://www.smard.de/en/downloadcenter/download-market-data/

Timeframe: 01.08.2018 - 31.12.2025, Resolution = 60min

Region: Germany

Key Variables: 
- "Germany/Luxembourg [€/MWh]"

## Weather Data
File: open-meteo-48.83N9.15E293m.xlsx

Source (with description of variables): https://open-meteo.com/en/docs/historical-weather-api

Timeframe: 01.01.2017, 02:00 -01.01.2026, Resolution = 15min

Region: Stuttgart

Key Variables:
- "temperature_2m (°C)", temperature 2 meter above ground
- "relative_humidity_2m (%)", relative humidity 2 meter above ground
- "precipitation (mm)", Total precipitation (rain, showers, snow) sum of the preceding 15min
- "shortwave_radiation (W/m²)", shortwave solar radiation as average of the preceding 15min, solar panels primarily work with shortwave radiation
- "wind_speed_10m (km/h)", Wind speed at 10 meters above ground. Wind speed on 10 meters is the standard level
- dew_point_2m (°C), Dew point temperature at 2 meters above ground
- "weather_code (wmo code)"
	- "0": "Clear sky"
	- "1": "Mainly clear",
	- "2": "Partly cloudy",
	- "3": "Overcast",
	- "45": "Fog",
	- "48": "Depositing rime fog",
	- "51": "Light drizzle",
	- "53": "Moderate drizzle",
	- "55": "Dense drizzle",
	- "56": "Light freezing drizzle",
	- "57": "Dense freezing drizzle",
	- "61": "Slight rain",
	- "63": "Moderate rain",
	- "65": "Heavy rain",
	- "66": "Light freezing rain",
	- "67": "Heavy freezing rain",
	- "71": "Slight snowfall",
	- "73": "Moderate snowfall",
	- "75": "Heavy snowfall",
	- "77": "Snow grains",
	- "80": "Slight rain showers",
	- "81": "Moderate rain showers",
	- "82": "Violent rain showers",
	- "85": "Slight snow showers",
	- "86": "Heavy snow showers",
	- "95": "Thunderstorm",
	- "96": "Thunderstorm with slight hail",
	- "97": "Heavy thunderstorm",
	- "99": "Thunderstorm with heavy hail" 


    
    