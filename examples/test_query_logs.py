"""
Test script for query_logs functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.local_log_tool import LocalLogAnalysisTool

def test_query_logs():
    print("Initializing Local Log Analysis Tool...")
    tool = LocalLogAnalysisTool()
    tool.initialize()
    
    # Test 1: Query logs in a specific time range
    print("\n=== Test 1: Query logs in a 5-minute window ===")
    result = tool.query_logs(
        start_time="2021-03-04T00:00:00",
        end_time="2021-03-04T00:05:00",
        limit=5
    )
    print(result)
    
    # Test 2: Query logs for a specific service
    print("\n\n=== Test 2: Query logs for Tomcat01 service ===")
    result = tool.query_logs(
        start_time="2021-03-04T00:00:00",
        end_time="2021-03-04T00:05:00",
        service_name="Tomcat01",
        limit=5
    )
    print(result)
    
    # Test 3: Query logs matching a pattern
    print("\n\n=== Test 3: Query logs containing 'Allocation Failure' ===")
    result = tool.query_logs(
        start_time="2021-03-04T00:00:00",
        end_time="2021-03-04T01:00:00",
        pattern="Allocation Failure",
        limit=3
    )
    print(result)
    
    # Test 4: Query logs for apache services
    print("\n\n=== Test 4: Query apache logs ===")
    result = tool.query_logs(
        start_time="2021-03-04T00:00:00",
        end_time="2021-03-04T00:05:00",
        pattern="apache",
        limit=5
    )
    print(result)

if __name__ == "__main__":
    test_query_logs()
