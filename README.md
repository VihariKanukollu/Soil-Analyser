# Soil-Analyser
# Soil Analyzer - Installation & Usage Guide

## Overview

This soil analysis system provides a software simulation of the AI-enabled soil testing device developed by Vihari Kanukollu. The software simulates the spectroscopy hardware and provides soil analysis capabilities including:

- Soil nutrient analysis (N, P, K)
- pH measurement
- Organic matter content
- Fertilizer recommendations
- Crop suitability analysis
- Visualization of results

## Installation

### Prerequisites

- Python 3.8 or newer
- pip (Python package manager)

### Setup Instructions

1. **Create a new directory for the project**

```bash
mkdir soil-analyzer
cd soil-analyzer
```

2. **Create the required directory structure**

```bash
mkdir -p soil_analyzer/sample_data
mkdir -p soil_analyzer/model_data
mkdir -p results
```

3. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install required packages**

```bash
pip install numpy pandas matplotlib
```

5. **Copy the source files**

Create the following files with the code provided in the code artifacts:
- `soil_analyzer/core.py` - Core analysis engine
- `demo.py` - Demo application

### File Structure

Your project should have the following structure:

```
soil-analyzer/
├── soil_analyzer/
│   ├── __init__.py
│   ├── core.py
│   ├── sample_data/
│   └── model_data/
├── results/
├── demo.py
└── venv/
```

Create an empty `__init__.py` file in the soil_analyzer directory:

```bash
touch soil_analyzer/__init__.py
```

## Usage

### Running the Demo

To start the soil analyzer demo application:

```bash
python demo.py
```

This will launch a text-based interface that allows you to:

1. Analyze new soil samples (simulated)
2. View recent analysis results
3. Export results to CSV
4. View system information

### First-time Use

When you first run the application, it will automatically:

1. Generate simulated spectral data for different soil types
2. Create a basic model for soil parameter prediction
3. Set up the necessary file structure

### Analyzing a Sample

1. From the main menu, select `1. Analyze New Soil Sample`
2. Enter the requested metadata for the sample:
   - Field name
   - Location
   - Sample depth
   - Collector name
   - (Optional) Target crop
   - (Optional) Additional notes
3. The system will simulate spectral data capture and analysis
4. Results will be displayed, including:
   - Soil parameter values
   - Status assessment (Low, Optimal, High)
   - Recommendations for fertilizers and amendments
   - Suitable crops

### Viewing Previous Results

1. From the main menu, select `2. View Recent Analysis Results`
2. Select a result from the list by entering its number
3. The full analysis report will be displayed

### Exporting Data

1. From the main menu, select `3. Export Results to CSV`
2. The system will export all analysis results to a CSV file in the current directory
3. This file can be opened in Excel or any spreadsheet application

## Extending the System

### Adapting to Real Hardware

When you have real spectroscopy hardware available, you'll need to:

1. Create a new hardware interface class in `soil_analyzer/hardware.py`
2. Implement the necessary methods to communicate with your specific hardware
3. Update the `SoilAnalyzer` class to use your real hardware interface

Example implementation for real hardware:

```python
class RealSpectrometerInterface:
    def __init__(self, device_id, port="/dev/ttyUSB0"):
        self.device_id = device_id
        self.port = port
        self.connected = False
        
    def connect(self):
        # Real hardware connection code
        # Example: self.device = serial.Serial(self.port, 9600)
        self.connected = True
        return self.connected
        
    def capture_spectrum(self):
        if not self.connected:
            raise Exception("Spectrometer not connected")
            
        # Code to capture real spectral data from hardware
        # ...
        
        return spectral_data
```

Then update the `SoilAnalyzer` initialization:

```python
# In soil_analyzer/core.py
def __init__(self, use_simulated_hardware=True, device_id=None):
    self.use_simulated_hardware = use_simulated_hardware
    
    if use_simulated_hardware:
        self.spectrometer = SimulatedSpectrometer()
    else:
        from soil_analyzer.hardware import RealSpectrometerInterface
        self.spectrometer = RealSpectrometerInterface(device_id)
```

### Training with Real Data

To improve the model with real soil samples and lab data:

1. Create a training module that:
   - Collects spectral data from real samples
   - Pairs it with lab analysis results
   - Trains a proper machine learning model
   - Saves the model for use in predictions

2. Replace the simplified linear model with your trained model

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Ensure you're running the application from the project root directory
   - Check that you've activated the virtual environment
   - Verify all required packages are installed

2. **FileNotFoundError**
   - Make sure all directories are created as specified
   - Check file paths in the code match your system

3. **Visualization Errors**
   - Ensure matplotlib is properly installed
   - Try updating matplotlib: `pip install --upgrade matplotlib`

### Getting Help

For assistance with this software, contact:
- Email: vihari@urbankisaan.com
- Documentation: www.farmos.ai

## Next Steps

1. **Collect Real Samples**
   - Gather soil samples from different fields
   - Obtain laboratory analysis results for each sample

2. **Develop Calibration System**
   - Create a dataset of spectral readings paired with lab results
   - Train a robust machine learning model for prediction

3. **Integrate with Real Hardware**
   - Select appropriate spectrometer components
   - Develop hardware interface to capture real spectral data
   - Replace the simulation with real hardware integration
