
import sys
import os
import pandas as pd
from src.tools.local_log_tool import LocalLogAnalysisTool

def debug_patterns():
    tool = LocalLogAnalysisTool()
    tool.initialize()
    
    start_time = "2021-03-04T00:00:00"
    end_time = "2021-03-04T00:10:00"
    
    print(f"Loading logs from {start_time} to {end_time}...")
    df = tool.loader.load_logs_for_time_range(start_time, end_time)
    print(f"Loaded {len(df)} logs.")
    
    # Filter for errors manually to see what we have
    error_logs = df[df['value'].str.contains('error|exception|fail', case=False, na=False)].copy()
    print(f"Found {len(error_logs)} raw error logs.")
    
    if error_logs.empty:
        print("No error logs found matching criteria.")
        return

    # Apply masking logic from the tool
    # 1. Mask ISO timestamps
    error_logs['pattern'] = error_logs['value'].str.replace(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4}', '<TIMESTAMP>', regex=True)
    # 2. Mask other common timestamp formats (e.g. 3748839.786)
    error_logs['pattern'] = error_logs['pattern'].str.replace(r'\d+\.\d+:', '<TIME>:', regex=True)
    # 3. Mask numbers (e.g. sizes, durations)
    error_logs['pattern'] = error_logs['pattern'].str.replace(r'\b\d+\b', '<NUM>', regex=True)
    # 4. Mask hex IDs (simple heuristic)
    error_logs['pattern'] = error_logs['pattern'].str.replace(r'\b[0-9a-fA-F]{8,}\b', '<ID>', regex=True)
    
    print("\nTop 5 Patterns:")
    print(error_logs['pattern'].value_counts().head(5))
    
    print("\nSample Raw vs Masked:")
    for i in range(min(3, len(error_logs))):
        print(f"Raw: {error_logs.iloc[i]['value']}")
        print(f"Msk: {error_logs.iloc[i]['pattern']}")
        print("-" * 20)

if __name__ == "__main__":
    debug_patterns()
