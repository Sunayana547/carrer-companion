import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple


class YieldModel:
    def __init__(self, file_path: str):
        self.data = pd.read_excel(file_path)
        # Temperature ranges for different crops (in Celsius)
        self.crop_temperature_ranges = {
            'wheat': {'min': 10, 'max': 25, 'optimal': 20},
            'rice': {'min': 20, 'max': 35, 'optimal': 28},
            'maize': {'min': 15, 'max': 35, 'optimal': 25},
            'sugarcane': {'min': 20, 'max': 35, 'optimal': 30},
            'cotton': {'min': 21, 'max': 40, 'optimal': 30},
            'soybean': {'min': 15, 'max': 35, 'optimal': 25},
            # Add more crops as needed
        }
        self._prepare()

    def get_crop_temperature_info(self, crop_name: str) -> Dict:
        """Get temperature information for a specific crop."""
        crop = crop_name.lower()
        if crop in self.crop_temperature_ranges:
            return self.crop_temperature_ranges[crop]
        return None
        
    def _prepare(self) -> None:
        df = self.data
        df.fillna(df.mean(numeric_only=True), inplace=True)
        if 'Temperatue' in df.columns:
            df.rename(columns={'Temperatue': 'Temperature'}, inplace=True)
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')

        target_col = 'Yeild (Q/acre)'
        if target_col not in df.columns:
            raise RuntimeError(f"Target column not found: {target_col}")

        X = df.drop(target_col, axis=1).select_dtypes(include=[np.number])
        y = df[target_col]

        self.feature_columns = X.columns.tolist()
        self.avg_yield = float(y.mean())
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_scaled, y)

    def predict(self, payload: dict) -> dict:
        crop = str(payload.get('crop', '')).lower()
        temp = payload.get('temp')
        
        # If crop is specified and temperature is provided, adjust prediction based on temperature
        temp_effect = 1.0  # Default multiplier
        if crop and crop in self.crop_temperature_ranges and temp is not None:
            temp_ranges = self.crop_temperature_ranges[crop]
            if temp < temp_ranges['min'] or temp > temp_ranges['max']:
                temp_effect = 0.1  # Very low yield if outside range
            else:
                # Calculate temperature effect (1.0 at optimal, decreasing towards min/max)
                temp_diff = abs(temp - temp_ranges['optimal'])
                temp_range = (temp_ranges['max'] - temp_ranges['min']) / 2
                temp_effect = max(0.1, 1 - (temp_diff / temp_range) * 0.5)
        
        inputs = {
            'Rain Fall (mm)': payload.get('rain'),
            'Fertilizer': payload.get('fertilizer'),
            'Temperature': temp,
            'Nitrogen (N)': payload.get('nitrogen'),
            'Phosphorus (P)': payload.get('phosphorus'),
            'Potassium (K)': payload.get('potassium'),
        }
        df = pd.DataFrame([inputs]).reindex(columns=self.feature_columns)
        for col in self.feature_columns:
            if col not in inputs or pd.isna(df[col].iloc[0]):
                df[col] = pd.to_numeric(self.data[col], errors='coerce').mean()
        x_scaled = self.scaler.transform(df)
        individual = [tree.predict(x_scaled) for tree in self.model.estimators_]
        pred = float(np.mean(individual))
        pred_std = float(np.std(individual))
        
        # Apply temperature effect if crop is specified
        if 'crop' in payload and payload['crop'].lower() in self.crop_temperature_ranges:
            pred *= temp_effect
            
        return {
            'yield': pred,
            'std': pred_std,
            'avg_yield': self.avg_yield,
            'feature_order': self.feature_columns,
            'temperature_effect': f"{temp_effect*100:.1f}%" if 'crop' in payload and payload['crop'] else None,
            'suitable_temperature': self.crop_temperature_ranges.get(crop, {}) if 'crop' in payload and payload['crop'] else None
        }


