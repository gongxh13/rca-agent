#!/usr/bin/env python3
import sys
import time
import subprocess
from datetime import datetime
from common import run_cmd, get_device_sectors, DM_DEV_NAME, DM_DEV_PATH

def inject():
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Injecting BAD DISK fault on {DM_DEV_NAME}...")

    # 1. Get size
    try:
        sectors = get_device_sectors(DM_DEV_PATH)
        print(f"Target device size: {sectors} sectors")
    except Exception as e:
        print(f"Error getting device size: {e}")
        print(f"Is the device {DM_DEV_NAME} set up?")
        sys.exit(1)

    # 2. Suspend I/O
    print("Suspending device...")
    run_cmd(["sudo", "dmsetup", "suspend", DM_DEV_NAME], check=True)

    # 3. Load error table
    # Table format: 0 <sectors> error
    error_table = f"0 {sectors} error"
    print(f"Loading error table: {error_table}")
    try:
        run_cmd(["sudo", "dmsetup", "load", DM_DEV_NAME, "--table", error_table], check=True)
    except Exception as e:
        print(f"Failed to load table: {e}")
        print("Resuming device to restore state...")
        run_cmd(["sudo", "dmsetup", "resume", DM_DEV_NAME], check=False)
        sys.exit(1)

    # 4. Resume I/O
    print("Resuming device (Fault Active)...")
    run_cmd(["sudo", "dmsetup", "resume", DM_DEV_NAME], check=True)

    print("\nFAULT INJECTED. All I/O to the disk should now fail.")

    print("Waiting 2 seconds to capture kernel logs...")
    time.sleep(2)
    
    log_file = "kernel.log"
    print(f"Collecting kernel logs since {start_time} into {log_file}...")
    try:
        with open(log_file, "w") as f:
            subprocess.run(["sudo", "journalctl", "-k", "--since", start_time, "--no-pager"], stdout=f, stderr=subprocess.STDOUT, check=False)
        print(f"Logs saved to {log_file}")
    except Exception as e:
        print(f"Error collecting logs: {e}")

if __name__ == "__main__":
    inject()
