# demo.py
"""
A simple demonstration script for the Soil Analyzer.
This script creates a command-line interface to interact with the analyzer.
"""

import os
import sys
import json
from soil_analyzer.core import SoilAnalyzer, RESULTS_DIR

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    print("\n" + "=" * 60)
    print("               SOIL ANALYZER SYSTEM")
    print("                 Simulation Mode")
    print("=" * 60 + "\n")

def print_menu():
    """Print the main menu"""
    print("\nMAIN MENU:")
    print("1. Analyze New Soil Sample")
    print("2. View Recent Analysis Results")
    print("3. Export Results to CSV")
    print("4. System Information")
    print("5. Exit")
    print("\nEnter your choice (1-5): ", end="")

def collect_sample_metadata():
    """Collect metadata for a new sample analysis"""
    print("\n--- New Soil Sample Analysis ---")
    print("Please enter sample information:\n")
    
    metadata = {}
    metadata["field_name"] = input("Field Name: ").strip()
    metadata["location"] = input("Location (GPS or description): ").strip()
    metadata["depth"] = input("Sample Depth (cm): ").strip()
    metadata["collector"] = input("Collected By: ").strip()
    
    # Optional target crop
    target_crop = input("Target Crop (optional): ").strip()
    if target_crop:
        metadata["target_crop"] = target_crop
    
    notes = input("Additional Notes: ").strip()
    if notes:
        metadata["notes"] = notes
    
    return metadata

def run_analysis(analyzer):
    """Run a new soil analysis"""
    metadata = collect_sample_metadata()
    
    print("\nPreparing analyzer...")
    print("Collecting spectral data...\n")
    
    # Run the analysis
    report = analyzer.analyze_sample(metadata)
    
    # Display results
    display_analysis_results(report)
    
    return report

def display_analysis_results(report):
    """Display the analysis results"""
    clear_screen()
    print_header()
    print(f"ANALYSIS RESULTS - Sample ID: {report['sample_id']}")
    print(f"Date: {report['analysis_date'][:10]}")
    
    if "metadata" in report:
        print(f"Field: {report['metadata'].get('field_name', 'N/A')}")
        print(f"Location: {report['metadata'].get('location', 'N/A')}")
    
    print("\n--- SOIL PARAMETERS ---")
    print("-" * 60)
    print(f"{'Parameter':<15} {'Value':<10} {'Unit':<8} {'Status':<10} {'Optimal Range'}")
    print("-" * 60)
    
    for param, data in report["soil_parameters"].items():
        status_marker = "✓" if data["status"] == "Optimal" else "⚠" if data["status"] == "Low" else "⚠"
        print(f"{param.capitalize():<15} {data['value']:<10.1f} {data['unit']:<8} {status_marker} {data['status']:<8} {data['optimal_range'][0]}-{data['optimal_range'][1]}")
    
    print("\n--- RECOMMENDATIONS ---")
    
    # Fertilizer
    print("\nFertilizer Recommendations:")
    for rec in report["recommendations"]["fertilizer"]:
        print(f"• {rec['recommendation']}")
    
    # Lime
    print("\nLime Application:")
    print(f"• {report['recommendations']['lime_application']['recommendation']}")
    
    # Organic matter
    print("\nOrganic Amendments:")
    print(f"• {report['recommendations']['organic_amendments']['recommendation']}")
    
    # Crops
    print("\nSuitable Crops:")
    for crop in report["recommendations"]["suitable_crops"]:
        print(f"• {crop}")
    
    print("\nPress Enter to continue...", end="")
    input()

def list_recent_analyses():
    """List recent analyses and allow viewing them"""
    if not os.path.exists(RESULTS_DIR):
        print("\nNo analysis results found.")
        return
    
    # Get all result files
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
    
    if not result_files:
        print("\nNo analysis results found.")
        return
    
    # Sort by date (newest first)
    result_files.sort(reverse=True)
    
    clear_screen()
    print_header()
    print("RECENT ANALYSES")
    print("\n{:<5} {:<20} {:<25} {:<20}".format("No.", "Sample ID", "Date", "Field/Location"))
    print("-" * 70)
    
    results_data = []
    for i, filename in enumerate(result_files[:10]):  # Show 10 most recent
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            sample_id = data.get("sample_id", "Unknown")
            date = data.get("analysis_date", "")[:10]
            
            field = "N/A"
            location = "N/A"
            
            if "metadata" in data:
                field = data["metadata"].get("field_name", "N/A")
                location = data["metadata"].get("location", "N/A")
            
            display_loc = field if field != "N/A" else location
            print("{:<5} {:<20} {:<25} {:<20}".format(i+1, sample_id, date, display_loc[:20]))
            
            results_data.append(data)
            
        except Exception as e:
            print(f"{i+1:<5} {filename:<20} [Error reading file]")
    
    if not results_data:
        print("\nNo valid analysis results found.")
        return
    
    print("\nEnter the number to view details (or 0 to return): ", end="")
    choice = input().strip()
    
    try:
        choice = int(choice)
        if 1 <= choice <= len(results_data):
            display_analysis_results(results_data[choice-1])
        elif choice != 0:
            print("\nInvalid choice.")
    except ValueError:
        print("\nInvalid input. Please enter a number.")

def export_to_csv():
    """Export analysis results to CSV"""
    if not os.path.exists(RESULTS_DIR):
        print("\nNo analysis results found to export.")
        return
    
    # Get all result files
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
    
    if not result_files:
        print("\nNo analysis results found to export.")
        return
    
    print("\nExporting analysis results to CSV...")
    
    # Create output file
    output_file = os.path.join(os.getcwd(), "soil_analysis_results.csv")
    
    # CSV header
    header = ["Sample ID", "Date", "Field", "Location", "Depth", 
              "Nitrogen (mg/kg)", "Phosphorus (mg/kg)", "Potassium (mg/kg)", 
              "pH", "Organic Matter (%)"]
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(",".join(header) + "\n")
        
        # Process each file
        for filename in result_files:
            filepath = os.path.join(RESULTS_DIR, filename)
            try:
                with open(filepath, 'r') as json_file:
                    data = json.load(json_file)
                
                # Prepare row data
                row = [
                    data.get("sample_id", ""),
                    data.get("analysis_date", "")[:10],
                    data.get("metadata", {}).get("field_name", ""),
                    data.get("metadata", {}).get("location", ""),
                    data.get("metadata", {}).get("depth", ""),
                ]
                
                # Add soil parameters
                soil_params = data.get("soil_parameters", {})
                row.append(str(soil_params.get("nitrogen", {}).get("value", "")))
                row.append(str(soil_params.get("phosphorus", {}).get("value", "")))
                row.append(str(soil_params.get("potassium", {}).get("value", "")))
                row.append(str(soil_params.get("ph", {}).get("value", "")))
                row.append(str(soil_params.get("organic_matter", {}).get("value", "")))
                
                # Write row
                f.write(",".join(row) + "\n")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nExport complete. File saved at: {output_file}")
    print("\nPress Enter to continue...", end="")
    input()

def show_system_info():
    """Display system information"""
    clear_screen()
    print_header()
    print("SYSTEM INFORMATION")
    
    print("\nSoil Analyzer Simulator")
    print("Version: 1.0.0")
    print("Mode: Simulation (Hardware Emulation)")
    
    # Count analyses
    analysis_count = 0
    if os.path.exists(RESULTS_DIR):
        analysis_count = len([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')])
    
    print(f"\nTotal Analyses: {analysis_count}")
    
    print("\nCapabilities:")
    print("• Spectral Analysis: UV-Visible-NIR (200-2500nm)")
    print("• Soil Parameters: Nitrogen, Phosphorus, Potassium, pH, Organic Matter")
    print("• Recommendations: Fertilizer, Lime, Organic Amendments, Crop Suitability")
    
    print("\nPress Enter to continue...", end="")
    input()

def main():
    """Main application function"""
    # Initialize the analyzer
    analyzer = SoilAnalyzer(use_simulated_hardware=True)
    
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        try:
            choice = input().strip()
            
            if choice == '1':
                run_analysis(analyzer)
            elif choice == '2':
                list_recent_analyses()
            elif choice == '3':
                export_to_csv()
            elif choice == '4':
                show_system_info()
            elif choice == '5':
                print("\nExiting Soil Analyzer. Thank you for using our system!")
                sys.exit(0)
            else:
                print("\nInvalid choice. Please enter a number between 1 and 5.")
                print("Press Enter to continue...", end="")
                input()
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled. Returning to main menu...")
            continue
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Press Enter to continue...", end="")
            input()

if __name__ == "__main__":
    main()
