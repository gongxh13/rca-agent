"""
Test script for Local Log Analysis Tool
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.local_log_tool import LocalLogAnalysisTool

def test_log_tools():
    print("Initializing Local Log Analysis Tool...")
    tool = LocalLogAnalysisTool()
    tool.initialize()
    
    # Define a time range that we know has data (based on file exploration)
    # Files are like 2021_03_04, so let's pick a range within that day
    start_time = "2021-03-04T00:00:00"
    end_time = "2021-03-04T23:59:59"
    
    print(f"\nTesting with time range: {start_time} to {end_time}")
    
    # 1. Test get_log_summary
    print("\n--- Testing get_log_summary ---")
    summary = tool.get_log_summary(start_time=start_time, end_time=end_time)
    print(summary)
    
    # 2. Test find_error_patterns
    print("\n--- Testing find_error_patterns ---")
    patterns = tool.find_error_patterns(start_time=start_time, end_time=end_time)
    print(patterns)
    
    # 3. Test detect_anomalies
    print("\n--- Testing detect_anomalies ---")
    anomalies = tool.detect_anomalies(start_time=start_time, end_time=end_time, sensitivity=0.5)
    print(anomalies)
    
    # 4. Test analyze_error_frequency
    print("\n--- Testing analyze_error_frequency ---")
    freq = tool.analyze_error_frequency(start_time=start_time, end_time=end_time)
    print(freq)
    
    # 5. Test find_correlated_events
    # Use a pattern that likely exists, e.g., "GC" or "CMS" from the sample we saw
    print("\n--- Testing find_correlated_events (Reference: 'CMS-concurrent-mark-start') ---")
    correlated = tool.find_correlated_events(
        reference_event="CMS-concurrent-mark-start",
        start_time=start_time,
        end_time=end_time,
        time_window_seconds=5
    )
    print(correlated)

if __name__ == "__main__":
    test_log_tools()
