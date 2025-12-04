"""
Test script for LocalMetricAnalysisTool
"""

from src.tools.local_metric_tool import LocalMetricAnalysisTool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_metrics():
    print("Initializing LocalMetricAnalysisTool...")
    tool = LocalMetricAnalysisTool(config={"dataset_path": "datasets/OpenRCA/Bank"})
    tool.initialize()
    
    # Test parameters (based on available data)
    start_time = "2021-03-04T14:00:00"
    end_time = "2021-03-04T15:00:00"
    
    print("\n=== Testing Application Metrics ===")
    
    print("\n1. Testing get_service_performance:")
    print(tool.get_service_performance(start_time, end_time))
    
    print("\n2. Testing find_slow_services:")
    print(tool.find_slow_services(start_time, end_time, threshold_ms=100))
    
    print("\n3. Testing find_low_success_rate_services:")
    print(tool.find_low_success_rate_services(start_time, end_time, threshold_percent=100))
    
    print("\n=== Testing Infrastructure Metrics ===")
    
    print("\n4. Testing get_resource_metrics (CPU for Tomcat04):")
    print(tool.get_resource_metrics("Tomcat04", "CPU", start_time, end_time))
    
    print("\n5. Testing find_high_resource_usage (CPU pattern):")
    print(tool.find_high_resource_usage("CPU", start_time, end_time, threshold=10, top=5))
    
    print("\n6. Testing detect_metric_anomalies:")
    print(tool.detect_metric_anomalies(start_time, end_time, "Tomcat04", sensitivity=2.0, top=5))
    
    print("\n7. Testing get_component_health_summary (all components, CPU only):")
    print(tool.get_component_health_summary(start_time, end_time, metric_pattern="CPU", warning_threshold=50, critical_threshold=80))
    
    print("\n=== Testing Discovery Methods ===")
    
    print("\n8. Testing get_available_components:")
    print(tool.get_available_components(start_time, end_time))
    
    print("\n9. Testing get_available_metrics (all components):")
    result = tool.get_available_metrics(start_time, end_time)
    # Only print first 500 chars to avoid too much output
    print(result[:500] + "..." if len(result) > 500 else result)
    
    print("\n10. Testing get_available_metrics (Tomcat04 only):")
    print(tool.get_available_metrics(start_time, end_time, "Tomcat04"))
    
    print("\n11. Testing get_available_metrics (Tomcat04, CPU pattern, top=5):")
    print(tool.get_available_metrics(start_time, end_time, "Tomcat04", metric_pattern="CPU", top=5))

if __name__ == "__main__":
    test_metrics()
