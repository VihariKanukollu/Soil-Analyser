# soil_analyzer/core.py
"""
Core module for the Soil Analyzer system.
This module includes:
1. Simulated hardware interface
2. Spectral data processing
3. Soil parameter prediction
4. Simple reporting

To use:
1. Create a SoilAnalyzer instance
2. Call analyze_sample() to get results
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import random
import base64
from io import BytesIO

# Directory for sample data (create these in your project)
SAMPLE_DATA_DIR = "./sample_data"
MODEL_COEFFICIENTS_FILE = "./model_data/model_coefficients.json"
RESULTS_DIR = "./results"

# Create necessary directories
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_COEFFICIENTS_FILE), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===============================================================
# HARDWARE SIMULATION LAYER
# ===============================================================

class SimulatedSpectrometer:
    """Simulates a spectrometer by providing sample spectral data"""
    
    def __init__(self):
        """Initialize the simulated spectrometer"""
        self.connected = False
        self._create_sample_data_if_needed()
        
    def connect(self):
        """Simulate connecting to the hardware"""
        print("Connecting to simulated spectrometer...")
        self.connected = True
        return self.connected
        
    def capture_spectrum(self):
        """Return simulated spectral data"""
        if not self.connected:
            raise Exception("Spectrometer not connected. Call connect() first.")
            
        # Select a random sample file
        sample_files = [f for f in os.listdir(SAMPLE_DATA_DIR) if f.endswith('.npy')]
        if not sample_files:
            raise Exception("No sample data files found in sample_data directory")
            
        sample_file = os.path.join(SAMPLE_DATA_DIR, random.choice(sample_files))
        
        # Load the spectral data
        spectral_data = np.load(sample_file)
        
        # Add some random noise to simulate variations
        noise = np.random.normal(0, 0.02, spectral_data.shape)
        spectral_data = spectral_data + noise
        
        # Ensure values stay in reasonable range
        spectral_data = np.clip(spectral_data, 0, 1)
        
        print(f"Captured simulated spectrum with {len(spectral_data)} data points")
        return spectral_data
    
    def disconnect(self):
        """Simulate disconnecting from hardware"""
        self.connected = False
        print("Disconnected from simulated spectrometer")
        
    def _create_sample_data_if_needed(self):
        """Create sample spectral data if none exists"""
        sample_files = [f for f in os.listdir(SAMPLE_DATA_DIR) if f.endswith('.npy')]
        
        if len(sample_files) >= 5:
            return  # We already have enough sample files
            
        print("Generating sample spectral data...")
        
        # Create wavelength range (200-2500nm with 5nm intervals)
        wavelengths = np.arange(200, 2500, 5)
        
        # Define soil types with different characteristics
        soil_types = [
            {"name": "clay_high_nitrogen", "peaks": [(580, 0.7), (1400, 0.6)], "nitrogen": 250},
            {"name": "sandy_low_nutrients", "peaks": [(650, 0.8), (1200, 0.3)], "nitrogen": 80},
            {"name": "loam_balanced", "peaks": [(500, 0.5), (1300, 0.5)], "nitrogen": 150},
            {"name": "high_organic_matter", "peaks": [(450, 0.4), (1600, 0.7)], "nitrogen": 220},
            {"name": "acidic_clay", "peaks": [(520, 0.6), (1350, 0.5)], "nitrogen": 180}
        ]
        
        # Generate spectral data for each soil type
        for soil in soil_types:
            # Create base spectrum
            spectrum = np.zeros_like(wavelengths, dtype=float)
            
            # Add characteristics peaks
            for peak_pos, peak_height in soil["peaks"]:
                # Create a gaussian peak
                peak = peak_height * np.exp(-(wavelengths - peak_pos)**2 / (2 * 50**2))
                spectrum += peak
            
            # Add some baseline
            baseline = 0.2 + 0.1 * np.sin(wavelengths / 500)
            spectrum += baseline
            
            # Normalize to 0-1 range
            spectrum = spectrum / np.max(spectrum)
            
            # Save the spectrum
            output_file = os.path.join(SAMPLE_DATA_DIR, f"{soil['name']}.npy")
            np.save(output_file, spectrum)
            
            # Also save the metadata
            metadata = {
                "soil_type": soil["name"],
                "wavelength_range": [int(wavelengths[0]), int(wavelengths[-1])],
                "data_points": len(spectrum),
                "soil_properties": {
                    "nitrogen": soil["nitrogen"],
                    "phosphorus": int(soil["nitrogen"] * 0.4 + random.uniform(-20, 20)),
                    "potassium": int(soil["nitrogen"] * 0.8 + random.uniform(-30, 30)),
                    "ph": round(6.5 + random.uniform(-1.5, 1.5), 1),
                    "organic_matter": round(2.0 + soil["nitrogen"] / 100, 1)
                }
            }
            
            with open(output_file.replace('.npy', '.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
                
        # Create model coefficients based on the sample data
        self._create_model_coefficients()
                
    def _create_model_coefficients(self):
        """Create simple model coefficients for the simulator"""
        if os.path.exists(MODEL_COEFFICIENTS_FILE):
            return  # Model already exists
            
        # Generate simple coefficients for each soil parameter
        wavelengths = np.arange(200, 2500, 5)
        num_coefficients = len(wavelengths)
        
        # Each parameter gets a set of coefficients that emphasize different spectral regions
        model_data = {
            "wavelengths": wavelengths.tolist(),
            "parameters": {
                "nitrogen": {
                    "intercept": 50.0,
                    "coefficients": self._generate_param_coefficients(num_coefficients, [400, 800])
                },
                "phosphorus": {
                    "intercept": 20.0,
                    "coefficients": self._generate_param_coefficients(num_coefficients, [500, 700])
                },
                "potassium": {
                    "intercept": 100.0,
                    "coefficients": self._generate_param_coefficients(num_coefficients, [600, 900])
                },
                "ph": {
                    "intercept": 6.5,
                    "coefficients": self._generate_param_coefficients(num_coefficients, [300, 550])
                },
                "organic_matter": {
                    "intercept": 2.0,
                    "coefficients": self._generate_param_coefficients(num_coefficients, [1000, 1600])
                }
            }
        }
        
        # Save model data
        os.makedirs(os.path.dirname(MODEL_COEFFICIENTS_FILE), exist_ok=True)
        with open(MODEL_COEFFICIENTS_FILE, 'w') as f:
            json.dump(model_data, f, indent=2)
            
    def _generate_param_coefficients(self, size, peak_range):
        """Generate coefficients with emphasis on specific wavelength ranges"""
        coeffs = np.zeros(size)
        
        # Find indices corresponding to the peak range
        start_idx = int((peak_range[0] - 200) / 5)
        end_idx = int((peak_range[1] - 200) / 5)
        
        # Create coefficients with peaks in the specified range
        coeffs[start_idx:end_idx] = np.random.normal(0, 0.1, end_idx - start_idx)
        
        # Add some small values elsewhere
        other_indices = list(range(0, start_idx)) + list(range(end_idx, size))
        coeffs[other_indices] = np.random.normal(0, 0.01, len(other_indices))
        
        return coeffs.tolist()

# ===============================================================
# SOIL ANALYSIS ENGINE
# ===============================================================

class SpectralProcessor:
    """Processes spectral data and extracts features"""
    
    def __init__(self):
        """Initialize the spectral processor"""
        pass
        
    def process(self, spectral_data):
        """Process raw spectral data"""
        # Apply preprocessing techniques
        smoothed = self._smooth_data(spectral_data)
        baseline_corrected = self._correct_baseline(smoothed)
        normalized = self._normalize(baseline_corrected)
        
        return normalized
        
    def _smooth_data(self, data):
        """Apply smoothing filter to reduce noise"""
        # Simple moving average filter
        window_size = 5
        smoothed = np.zeros_like(data)
        
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed[i] = np.mean(data[start:end])
            
        return smoothed
        
    def _correct_baseline(self, data):
        """Correct baseline drift"""
        # Simple baseline correction by subtracting minimum
        baseline = np.min(data)
        return data - baseline
        
    def _normalize(self, data):
        """Normalize data to 0-1 range"""
        if np.max(data) - np.min(data) > 0:
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        return data


class SoilPredictor:
    """Predicts soil parameters from spectral data"""
    
    def __init__(self):
        """Initialize the soil predictor"""
        self._load_model()
        
    def predict(self, processed_spectrum):
        """Predict soil parameters from processed spectral data"""
        # Ensure the spectrum length matches model expectations
        if len(processed_spectrum) != len(self.model_data["wavelengths"]):
            raise ValueError(f"Spectral data length ({len(processed_spectrum)}) does not match model expected length ({len(self.model_data['wavelengths'])})")
        
        # Calculate predictions for each parameter
        predictions = {}
        for param, model in self.model_data["parameters"].items():
            # Simple linear model: intercept + weighted sum of spectrum values
            value = model["intercept"] + np.sum(np.array(model["coefficients"]) * processed_spectrum)
            
            # Apply reasonable constraints
            if param == "ph":
                value = max(3.5, min(9.0, value))
            elif param == "organic_matter":
                value = max(0.1, min(10.0, value))
            else:  # Nutrients (N, P, K)
                value = max(0, value)
                
            predictions[param] = value
            
        return predictions
        
    def _load_model(self):
        """Load model coefficients"""
        if not os.path.exists(MODEL_COEFFICIENTS_FILE):
            raise FileNotFoundError(f"Model file not found: {MODEL_COEFFICIENTS_FILE}")
            
        with open(MODEL_COEFFICIENTS_FILE, 'r') as f:
            self.model_data = json.load(f)


class ReportGenerator:
    """Generates soil analysis reports"""
    
    def __init__(self):
        """Initialize the report generator"""
        pass
        
    def generate_report(self, sample_id, soil_parameters):
        """Generate a comprehensive soil analysis report"""
        # Create report structure
        report = {
            "sample_id": sample_id,
            "analysis_date": datetime.now().isoformat(),
            "soil_parameters": self._format_parameters(soil_parameters),
            "recommendations": self._generate_recommendations(soil_parameters),
            "visualizations": self._generate_visualizations(soil_parameters)
        }
        
        return report
        
    def _format_parameters(self, params):
        """Format raw parameters with units and interpretations"""
        formatted = {}
        
        # Parameter definitions with units and optimal ranges
        param_defs = {
            "nitrogen": {"unit": "mg/kg", "optimal_range": [120, 250]},
            "phosphorus": {"unit": "mg/kg", "optimal_range": [20, 50]},
            "potassium": {"unit": "mg/kg", "optimal_range": [120, 250]},
            "ph": {"unit": "pH", "optimal_range": [6.0, 7.0]},
            "organic_matter": {"unit": "%", "optimal_range": [2.5, 5.0]}
        }
        
        # Format each parameter
        for param, value in params.items():
            if param in param_defs:
                def_data = param_defs[param]
                
                # Determine status based on optimal range
                if value < def_data["optimal_range"][0]:
                    status = "Low"
                elif value > def_data["optimal_range"][1]:
                    status = "High"
                else:
                    status = "Optimal"
                
                formatted[param] = {
                    "value": round(value, 2) if param != "ph" else round(value, 1),
                    "unit": def_data["unit"],
                    "status": status,
                    "optimal_range": def_data["optimal_range"]
                }
                
        return formatted
        
    def _generate_recommendations(self, params):
        """Generate recommendations based on soil parameters"""
        recommendations = {
            "fertilizer": self._recommend_fertilizer(params),
            "lime_application": self._recommend_lime(params),
            "organic_amendments": self._recommend_organic(params),
            "suitable_crops": self._recommend_crops(params)
        }
        
        return recommendations
        
    def _recommend_fertilizer(self, params):
        """Recommend fertilizer based on nutrient levels"""
        recommendations = []
        
        # Nitrogen recommendation
        if params["nitrogen"] < 120:
            deficit = 120 - params["nitrogen"]
            recommendations.append({
                "nutrient": "Nitrogen",
                "recommendation": f"Apply approximately {int(deficit * 2.5)} kg/ha of nitrogen fertilizer",
                "reasoning": "Nitrogen levels are below optimal range"
            })
            
        # Phosphorus recommendation
        if params["phosphorus"] < 20:
            deficit = 20 - params["phosphorus"]
            recommendations.append({
                "nutrient": "Phosphorus",
                "recommendation": f"Apply approximately {int(deficit * 2)} kg/ha of phosphate fertilizer",
                "reasoning": "Phosphorus levels are below optimal range"
            })
            
        # Potassium recommendation
        if params["potassium"] < 120:
            deficit = 120 - params["potassium"]
            recommendations.append({
                "nutrient": "Potassium",
                "recommendation": f"Apply approximately {int(deficit * 1.5)} kg/ha of potassium fertilizer",
                "reasoning": "Potassium levels are below optimal range"
            })
            
        if not recommendations:
            recommendations.append({
                "nutrient": "All",
                "recommendation": "No additional fertilizer needed at this time",
                "reasoning": "Nutrient levels are within optimal ranges"
            })
            
        return recommendations
        
    def _recommend_lime(self, params):
        """Recommend lime application based on pH"""
        if params["ph"] < 6.0:
            deficit = 6.0 - params["ph"]
            return {
                "recommendation": f"Apply approximately {int(deficit * 1000)} kg/ha of agricultural lime",
                "reasoning": f"pH is {params['ph']}, which is below the optimal range"
            }
        elif params["ph"] > 7.5:
            excess = params["ph"] - 7.5
            return {
                "recommendation": "Consider applying sulfur products to reduce soil pH",
                "reasoning": f"pH is {params['ph']}, which is above the optimal range for most crops"
            }
        else:
            return {
                "recommendation": "No lime application needed",
                "reasoning": f"pH is {params['ph']}, which is within optimal range"
            }
            
    def _recommend_organic(self, params):
        """Recommend organic amendments based on organic matter content"""
        if params["organic_matter"] < 2.5:
            deficit = 2.5 - params["organic_matter"]
            return {
                "recommendation": f"Apply {int(deficit * 10)} tons/ha of compost or well-rotted manure",
                "reasoning": f"Organic matter content is {params['organic_matter']}%, which is below optimal"
            }
        else:
            return {
                "recommendation": "No additional organic amendments needed",
                "reasoning": f"Organic matter content is {params['organic_matter']}%, which is adequate"
            }
            
    def _recommend_crops(self, params):
        """Recommend suitable crops based on soil parameters"""
        # Simple crop recommendations based on pH and nutrient levels
        suitable_crops = []
        
        # pH-based recommendations
        if 5.5 <= params["ph"] <= 7.0:
            suitable_crops.extend(["Corn", "Wheat", "Soybeans"])
        if 5.0 <= params["ph"] <= 6.5:
            suitable_crops.extend(["Potatoes", "Strawberries"])
        if 6.0 <= params["ph"] <= 7.5:
            suitable_crops.extend(["Alfalfa", "Barley"])
            
        # Nutrient-based additions
        if params["nitrogen"] > 150 and params["phosphorus"] > 30:
            suitable_crops.append("Leafy Vegetables")
        if params["potassium"] > 150:
            suitable_crops.append("Root Vegetables")
            
        # Organic matter consideration
        if params["organic_matter"] > 3.0:
            suitable_crops.append("Specialty Vegetables")
            
        # Remove duplicates and sort
        suitable_crops = sorted(list(set(suitable_crops)))
        
        return suitable_crops
        
    def _generate_visualizations(self, params):
        """Generate visualizations of soil parameters"""
        visualizations = {
            "nutrient_chart": self._generate_nutrient_chart(params),
            "soil_health_gauge": self._generate_soil_health_gauge(params)
        }
        
        return visualizations
        
    def _generate_nutrient_chart(self, params):
        """Generate nutrient level chart"""
        # Extract nutrient values
        nutrients = ["nitrogen", "phosphorus", "potassium"]
        values = [params[n] for n in nutrients]
        
        # Create figure
        plt.figure(figsize=(8, 5))
        
        # Create bars
        bars = plt.bar(nutrients, values, width=0.6)
        
        # Add optimal range indicators
        optimal_ranges = {
            "nitrogen": [120, 250],
            "phosphorus": [20, 50],
            "potassium": [120, 250]
        }
        
        # Color bars based on optimal ranges
        for i, nutrient in enumerate(nutrients):
            opt_min, opt_max = optimal_ranges[nutrient]
            if values[i] < opt_min:
                bars[i].set_color('orange')  # Deficient
            elif values[i] > opt_max:
                bars[i].set_color('red')     # Excess
            else:
                bars[i].set_color('green')   # Optimal
        
        # Add labels and title
        plt.ylabel('Concentration (mg/kg)')
        plt.title('Soil Nutrient Levels')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Convert plot to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return plot_data
        
    def _generate_soil_health_gauge(self, params):
        """Generate soil health gauge visualization"""
        # Calculate overall health score (0-100)
        score = self._calculate_soil_health_score(params)
        
        # Create figure for gauge
        plt.figure(figsize=(6, 3))
        
        # Health categories and colors
        categories = ['Poor', 'Fair', 'Good', 'Excellent']
        colors = ['red', 'orange', 'lightgreen', 'darkgreen']
        
        # Create simple horizontal gauge
        plt.barh(0, 100, height=0.5, color='lightgray')
        plt.barh(0, score, height=0.5, color=colors[min(3, score // 25)])
        
        # Add marker for score
        plt.scatter(score, 0, color='black', s=100, zorder=5)
        
        # Add category labels
        for i in range(4):
            plt.text(i * 25 + 12.5, -0.25, categories[i], ha='center')
        
        # Add score text
        plt.text(50, 0.25, f"Soil Health Score: {score}/100", 
                ha='center', fontsize=12, fontweight='bold')
        
        # Remove axes and set limits
        plt.axis('off')
        plt.xlim(0, 100)
        plt.ylim(-0.5, 0.5)
        
        # Convert plot to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return plot_data
        
    def _calculate_soil_health_score(self, params):
        """Calculate overall soil health score"""
        # Component scores (0-100 each)
        scores = {
            "nitrogen": self._score_in_range(params["nitrogen"], 120, 250),
            "phosphorus": self._score_in_range(params["phosphorus"], 20, 50),
            "potassium": self._score_in_range(params["potassium"], 120, 250),
            "ph": self._score_in_range(params["ph"], 6.0, 7.0, True),
            "organic_matter": self._score_in_range(params["organic_matter"], 2.5, 5.0)
        }
        
        # Weighted average
        weights = {
            "nitrogen": 0.2,
            "phosphorus": 0.2, 
            "potassium": 0.2,
            "ph": 0.25,
            "organic_matter": 0.15
        }
        
        overall_score = sum(scores[param] * weights[param] for param in scores)
        
        return int(overall_score)
        
    def _score_in_range(self, value, optimal_min, optimal_max, is_critical=False):
        """Score a parameter based on its position relative to the optimal range"""
        # Perfect score if in optimal range
        if optimal_min <= value <= optimal_max:
            return 100
            
        # Calculate how far outside the range we are
        if value < optimal_min:
            deviation = (optimal_min - value) / optimal_min
        else:
            deviation = (value - optimal_max) / optimal_max
            
        # Cap deviation at 1.0 (100% deviation)
        deviation = min(1.0, deviation)
        
        # More severe penalty for critical parameters
        penalty_factor = 1.0 if not is_critical else 1.25
        
        # Calculate score
        score = 100 - (deviation * 100 * penalty_factor)
        
        # Ensure score is in 0-100 range
        return max(0, min(100, score))

# ===============================================================
# MAIN SOIL ANALYZER CLASS
# ===============================================================

class SoilAnalyzer:
    """Main soil analyzer class that coordinates the analysis process"""
    
    def __init__(self, use_simulated_hardware=True):
        """Initialize the soil analyzer"""
        self.use_simulated_hardware = use_simulated_hardware
        
        # Initialize components
        if use_simulated_hardware:
            self.spectrometer = SimulatedSpectrometer()
        else:
            # This would be replaced with actual hardware in the future
            raise NotImplementedError("Real hardware interface not implemented yet")
            
        self.processor = SpectralProcessor()
        self.predictor = SoilPredictor()
        self.report_generator = ReportGenerator()
        
    def analyze_sample(self, sample_metadata=None):
        """Analyze a soil sample and generate a report"""
        # Connect to spectrometer
        self.spectrometer.connect()
        
        try:
            # Capture spectral data
            spectrum = self.spectrometer.capture_spectrum()
            
            # Process the spectrum
            processed_spectrum = self.processor.process(spectrum)
            
            # Predict soil parameters
            soil_parameters = self.predictor.predict(processed_spectrum)
            
            # Generate sample ID
            sample_id = f"SAMPLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate report
            report = self.report_generator.generate_report(sample_id, soil_parameters)
            
            # Add metadata if provided
            if sample_metadata:
                report["metadata"] = sample_metadata
                
            # Save report
            self._save_report(sample_id, report)
            
            return report
            
        finally:
            # Always disconnect
            self.spectrometer.disconnect()
            
    def _save_report(self, sample_id, report):
        """Save the analysis report to disk"""
        # Create output directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Save as JSON file
        output_file = os.path.join(RESULTS_DIR, f"{sample_id}.json")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Report saved to {output_file}")

# ===============================================================
# DEMO USAGE
# ===============================================================

def run_demo():
    """Run a simple demonstration of the soil analyzer"""
    print("=== Soil Analyzer Demo ===")
    
    # Initialize analyzer
    analyzer = SoilAnalyzer(use_simulated_hardware=True)
    
    # Sample metadata
    metadata = {
        "location": "Test Farm, Field A",
        "depth": "0-15 cm",
        "collector": "Demo User",
        "notes": "Demo sample for testing"
    }
    
    # Run analysis
    print("\nAnalyzing sample...")
    report = analyzer.analyze_sample(metadata)
    
    # Print results
    print("\n=== Analysis Results ===")
    for param, data in report["soil_parameters"].items():
        status_color = '\033[32m' if data["status"] == "Optimal" else '\033[33m' if data["status"] == "Low" else '\033[31m'
        reset_color = '\033[0m'
        print(f"{param.capitalize()}: {data['value']} {data['unit']} - {status_color}{data['status']}{reset_color}")

    # Print recommendations
    print("\n=== Recommendations ===")
    
    # Fertilizer
    print("Fertilizer Recommendations:")
    for rec in report["recommendations"]["fertilizer"]:
        print(f"- {rec['recommendation']}")
    
    # Lime
    print("\nLime Application:")
    print(f"- {report['recommendations']['lime_application']['recommendation']}")
    
    # Organic matter
    print("\nOrganic Amendments:")
    print(f"- {report['recommendations']['organic_amendments']['recommendation']}")
    
    # Crops
    print("\nSuitable Crops:")
    for crop in report["recommendations"]["suitable_crops"]:
        print(f"- {crop}")
    
    print(f"\nFull report saved to {os.path.join(RESULTS_DIR, report['sample_id'] + '.json')}")
    print("\nReport includes visualization images (base64 encoded) for the nutrient chart and soil health gauge")

if __name__ == "__main__":
    run_demo()
