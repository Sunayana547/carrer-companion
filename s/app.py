from flask import Flask, request, jsonify, render_template
import requests
import os
import numpy as np
from backend import YieldModel


app = Flask(__name__, static_folder='assets', static_url_path='/assets', template_folder='templates')
MODEL = YieldModel('crop yield data sheet.xlsx')


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/crop/<crop_name>')
def get_crop_info(crop_name):
    """
    Get temperature information for a specific crop.
    Example: /api/crop/wheat
    """
    try:
        crop = crop_name.lower()
        if hasattr(MODEL, 'crop_temperature_ranges') and crop in MODEL.crop_temperature_ranges:
            temp_info = MODEL.crop_temperature_ranges[crop]
            return jsonify({
                'success': True,
                'crop': crop.capitalize(),
                'temperature': {
                    'min': temp_info['min'],
                    'max': temp_info['max'],
                    'optimal': temp_info['optimal']
                },
                'message': f"{crop.capitalize()} grows best between {temp_info['min']}-{temp_info['max']}°C with an optimal temperature of {temp_info['optimal']}°C"
            })
        else:
            return jsonify({
                'success': False,
                'message': f"No temperature available for {crop}"
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error getting crop information: {str(e)}"
        }), 500


@app.post('/api/predict')
def api_predict():
    """
    Predict crop yield based on input parameters.
    Expected parameters:
    - crop: (optional) Name of the crop (e.g., 'wheat', 'rice', 'maize')
    - temp: (optional) Temperature in Celsius
    - rain: (optional) Rainfall in mm
    - fertilizer: (optional) Amount of fertilizer
    - nitrogen: (optional) Nitrogen level
    - phosphorus: (optional) Phosphorus level
    - potassium: (optional) Potassium level
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        result = MODEL.predict(payload)
        
        # Add crop information if available
        crop = payload.get('crop', '').lower()
        if crop and hasattr(MODEL, 'crop_temperature_ranges') and crop in MODEL.crop_temperature_ranges:
            temp_info = MODEL.crop_temperature_ranges[crop]
            result['crop_info'] = {
                'name': crop.capitalize(),
                'optimal_temp': temp_info['optimal'],
                'min_temp': temp_info['min'],
                'max_temp': temp_info['max']
            }
            
            # Add temperature effect explanation
            temp = payload.get('temp')
            if temp is not None:
                if temp < temp_info['min'] or temp > temp_info['max']:
                    result['temperature_effect_explanation'] = f"Temperature {temp}°C is outside the optimal range for {crop} ({temp_info['min']}-{temp_info['max']}°C). Yield is significantly reduced."
                else:
                    temp_diff = abs(temp - temp_info['optimal'])
                    if temp_diff <= 2:
                        result['temperature_effect_explanation'] = f"Temperature {temp}°C is ideal for {crop}."
                    elif temp_diff <= 5:
                        result['temperature_effect_explanation'] = f"Temperature {temp}°C is within acceptable range for {crop}, but not optimal."
                    else:
                        result['temperature_effect_explanation'] = f"Temperature {temp}°C is at the edge of the acceptable range for {crop}."
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.post('/api/simulate')
def api_simulate():
    """
    Simulate yield predictions for a range of values for a specific feature.
    Returns a list of yield predictions for the given range.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        
        # Get crop information if provided
        crop = data.get('crop', '').lower()
        
        # Get base values from the request
        base_values = {
            'crop': crop,
            'temp': float(data.get('temp', 25)),
            'rain': float(data.get('rain', 50)),
            'fertilizer': int(data.get('fertilizer', 50)),
            'nitrogen': int(data.get('nitrogen', 50)),
            'phosphorus': int(data.get('phosphorus', 30)),
            'potassium': int(data.get('potassium', 20)),
        }
        
        # Get the feature to simulate and its current value
        sim_feature = data.get('sim_feature', 'temp')
        current_value = float(data.get('current_value', base_values.get(sim_feature, 25)))
        
        # Define ranges for each feature
        feature_ranges = {
            'temp': (0, 50, 1),        # min, max, step
            'rain': (0, 300, 10),      # mm
            'fertilizer': (0, 200, 5),  # kg/acre
            'nitrogen': (0, 200, 5),    # kg/ha
            'phosphorus': (0, 200, 5),  # kg/ha
            'potassium': (0, 200, 5)    # kg/ha
        }
        
        # Get the range for the current feature
        min_val, max_val, step = feature_ranges[sim_feature]
        
        # If simulating temperature and crop is specified, adjust range based on crop's temperature tolerance
        if sim_feature == 'temp' and crop and hasattr(MODEL, 'crop_temperature_ranges') and crop in MODEL.crop_temperature_ranges:
            crop_info = MODEL.crop_temperature_ranges[crop]
            min_val = max(min_val, crop_info['min'] - 5)  # Show 5 degrees below min
            max_val = min(max_val, crop_info['max'] + 5)  # Show 5 degrees above max
        
        # Generate values around the current value (5 points: current-2 to current+2)
        values = np.linspace(
            max(min_val, current_value - 2 * step),
            min(max_val, current_value + 2 * step),
            5
        )
        
        # Find the index of the current value (or closest to it)
        current_index = np.abs(values - current_value).argmin()
        
        # Generate predictions for each value
        predictions = []
        for i, val in enumerate(values):
            # Create a copy of base values and update the simulated feature
            sim_values = base_values.copy()
            sim_values[sim_feature] = val
            
            # Get prediction
            prediction = MODEL.predict(sim_values)
            
            # Add some realistic variation (simulating model uncertainty)
            if i != current_index:  # Don't add noise to the current value
                noise = np.random.normal(0, prediction.get('std', 1) * 0.2)
                prediction['yield'] = max(0, prediction.get('yield', 0) + noise)
            
            predictions.append({
                'x': round(float(val), 1) if sim_feature == 'temp' else int(round(val)),
                'yield': round(float(prediction.get('yield', 0)), 2),
                'std': round(float(prediction.get('std', 0)), 2)
            })
        
        response = {
            'data': predictions,
            'current_index': int(current_index),
            'feature': sim_feature,
            'unit': '°C' if sim_feature == 'temp' else 'mm' if sim_feature == 'rain' 
                    else 'kg/acre' if sim_feature == 'fertilizer' else 'kg/ha'
        }
        
        # Add crop temperature info if available
        if crop and hasattr(MODEL, 'crop_temperature_ranges') and crop in MODEL.crop_temperature_ranges:
            response['crop_info'] = {
                'name': crop.capitalize(),
                'optimal_temp': MODEL.crop_temperature_ranges[crop]['optimal'],
                'min_temp': MODEL.crop_temperature_ranges[crop]['min'],
                'max_temp': MODEL.crop_temperature_ranges[crop]['max']
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        return jsonify({
            'error': str(e),
            'data': [],
            'current_index': 0
        }), 400


@app.get('/api/weather')
def api_weather():
    city = request.args.get('city', 'Visakhapatnam')
    api_key = os.environ.get('OPENWEATHER_API_KEY', '6783ff27525de6477470a6bf93fc9a1a')
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        j = res.json()
        return jsonify({
            'city': city, 
            'temp': j.get('main', {}).get('temp'), 
            'rain': (j.get('rain') or {}).get('1h', 0),
            'humidity': j.get('main', {}).get('humidity'),
            'wind_speed': j.get('wind', {}).get('speed'),
            'description': j.get('weather', [{}])[0].get('description', '').title()
        })
    except Exception as e:
        print(f"Weather API error: {str(e)}")
        return jsonify({
            'city': city, 
            'temp': None, 
            'rain': None,
            'error': 'Failed to fetch weather data'
        }), 502


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


