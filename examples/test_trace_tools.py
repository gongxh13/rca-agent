"""
Test script for LocalTraceAnalysisTool
"""

from src.tools.local_trace_tool import LocalTraceAnalysisTool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_traces():
    print("Initializing LocalTraceAnalysisTool...")
    tool = LocalTraceAnalysisTool(config={"dataset_path": "datasets/OpenRCA/Bank"})
    tool.initialize()
    
    # Test parameters (based on available data in 2021_03_04)
    start_time = "2021-03-04T14:00:00"
    end_time = "2021-03-04T14:05:00"
    
    print("\n=== Testing Trace Analysis ===")
    
    print("\n1. Testing find_slow_spans:")
    # Using a high threshold to find interesting ones
    print(tool.find_slow_spans(start_time, end_time, min_duration_ms=500, limit=5))
    
    print("\n2. Testing identify_bottlenecks:")
    print(tool.identify_bottlenecks(start_time, end_time, min_impact_percentage=5.0))
    
    print("\n3. Testing get_service_dependencies:")
    print(tool.get_service_dependencies(start_time, end_time, service_name="dockerA2"))
    
    print("\n4. Testing detect_latency_anomalies:")
    print(tool.detect_latency_anomalies(start_time, end_time, sensitivity=0.9))
    
    print("\n5. Testing analyze_call_chain (picking a trace ID from slow spans if possible):")
    # Let's first get a trace ID from the slow spans output or just pick one we know exists
    # trace_id: gw0120210304220007497209 (found in previous run)
    trace_id = "gw0120210304220007497209"
    print(f"Analyzing trace: {trace_id}")
    print(tool.analyze_call_chain(trace_id, start_time, end_time))

    start_time = "2021-03-04T00:00:00"
    end_time = "2021-03-04T23:59:59"
    trace_id = "gw0120210304091259219332"
    print(f"Analyzing trace: {trace_id}")
    print(tool.analyze_call_chain(trace_id, start_time, end_time))

if __name__ == "__main__":
    test_traces()
