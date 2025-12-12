"""
Test script for detect_metric_anomalies method

Tests the detect_metric_anomalies method across multiple time ranges on March 4th (UTC+8).
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.local_metric_tool import LocalMetricAnalysisTool
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_detect_anomalies():
    """Test detect_metric_anomalies for multiple time ranges."""
    print("=" * 80)
    print("Testing detect_metric_anomalies")
    print("=" * 80)
    
    # Initialize tool
    print("\nInitializing LocalMetricAnalysisTool...")
    tool = LocalMetricAnalysisTool(config={"dataset_path": "datasets/OpenRCA/Bank"})
    tool.initialize()
    
    # Test time ranges (all on March 4th, UTC+8)
    test_ranges = [
        ("2021-03-04T01:00:00", "2021-03-04T01:30:00", "1:00 - 1:30"),
        ("2021-03-04T05:30:00", "2021-03-04T06:00:00", "5:30 - 6:00"),
        ("2021-03-04T07:00:00", "2021-03-04T07:30:00", "7:00 - 7:30"),
        ("2021-03-04T13:30:00", "2021-03-04T14:00:00", "13:30 - 14:00"),
    ]
    
    for start_time, end_time, time_label in test_ranges:
        print("\n" + "=" * 80)
        print(f"Time Range: {time_label} (March 4th, UTC+8)")
        print(f"Start: {start_time}")
        print(f"End:   {end_time}")
        print("=" * 80)
        
        try:
            # Call detect_metric_anomalies with default parameters
            result = tool.detect_metric_anomalies(
                start_time=start_time,
                end_time=end_time,
                method="both",  # Use both ruptures and zscore methods
                component_id=None,  # Analyze all candidate components
                sensitivity=3.0,
                top=10,
                ruptures_algorithm="pelt",
                ruptures_model="rbf",
                pen=5.0,
                z_threshold=None,
                min_data_points_ruptures=10,
                min_data_points_zscore=5,
                min_consecutive=3
            )
            
            # Parse JSON result
            anomalies = json.loads(result)
            
            print(f"\nTotal anomalies detected: {len(anomalies)}")
            
            if len(anomalies) == 0:
                print("No anomalies detected in this time range.")
            else:
                print("\nDetected Anomalies:")
                print("-" * 80)
                
                for idx, anomaly in enumerate(anomalies, 1):
                    print(f"\n[{idx}] Component: {anomaly.get('component_name', 'N/A')}")
                    print(f"     KPI: {anomaly.get('faulty_kpi', 'N/A')}")
                    print(f"     Fault Start Time: {anomaly.get('fault_start_time', 'N/A')}")
                    print(f"     Severity: {anomaly.get('severity_score', 'N/A')}")
                    print(f"     Deviation: {anomaly.get('deviation_pct', 0):.2f}%")
                    print(f"     Method: {anomaly.get('method', 'N/A')}")
                    print(f"     Change Index: {anomaly.get('change_idx', 'N/A')}")
                
                # Summary statistics
                print("\n" + "-" * 80)
                print("Summary Statistics:")
                print(f"  - Components affected: {len(set(a.get('component_name') for a in anomalies))}")
                print(f"  - Methods used: {set(a.get('method') for a in anomalies)}")
                
                # Group by severity
                severity_counts = {}
                for anomaly in anomalies:
                    severity = anomaly.get('severity_score', 'Unknown')
                    # Extract severity level from severity_score string
                    if '严重' in severity:
                        severity_level = '严重'
                    elif '显著' in severity:
                        severity_level = '显著'
                    elif '中等' in severity:
                        severity_level = '中等'
                    else:
                        severity_level = '未知'
                    severity_counts[severity_level] = severity_counts.get(severity_level, 0) + 1
                
                print(f"  - Severity distribution:")
                for level, count in severity_counts.items():
                    print(f"    * {level}: {count}")
                
                # Top 3 anomalies by deviation percentage
                sorted_anomalies = sorted(anomalies, key=lambda x: x.get('deviation_pct', 0), reverse=True)
                print(f"\n  - Top 3 anomalies by deviation:")
                for idx, anomaly in enumerate(sorted_anomalies[:3], 1):
                    print(f"    {idx}. {anomaly.get('component_name')} - {anomaly.get('faulty_kpi')}: {anomaly.get('deviation_pct', 0):.2f}%")
        
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 80)
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_detect_anomalies()

