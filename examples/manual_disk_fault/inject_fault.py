#!/usr/bin/env python3
import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from common import run_cmd, get_device_sectors, DM_DEV_NAME, DM_DEV_PATH, find_scsi_debug_device

def inject_bad_disk(start_time):
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

def inject_slow_disk(start_time):
    print(f"Injecting SLOW DISK fault on underlying scsi_debug device...")
    
    # 1. Find underlying device
    try:
        device = find_scsi_debug_device()
        print(f"Found backing device: {device}")
        device_name = Path(device).name
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Is scsi_debug module loaded?")
        sys.exit(1)

    # 2. Inject parameters
    print("Setting scsi_debug parameters for slow response...")
    try:
        # every_nth: 1 means every access is delayed? Or every 1st access?
        # scsi_debug documentation says: "every_nth: (default 0) when > 0, every nth command is ..."
        # if 1, then every command.
        run_cmd(["sudo", "sh", "-c", "echo 1 > /sys/bus/pseudo/drivers/scsi_debug/every_nth"], check=True)
        
        # delay: delay in jiffies (or ms). 5000 is significant.
        run_cmd(["sudo", "sh", "-c", "echo 5000 > /sys/bus/pseudo/drivers/scsi_debug/delay"], check=True)
        
        # timeout: set on block device side to avoid quick timeout if desired?
        # disk_fault_injector sets it to 30.
        run_cmd(["sudo", "sh", "-c", f"echo 30 > /sys/block/{device_name}/device/timeout"], check=True)
        
    except Exception as e:
        print(f"Error setting parameters: {e}")
        sys.exit(1)

    print("\nFAULT INJECTED. Disk I/O should now be slow (delayed).")

def collect_logs(start_time):
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

def main():
    parser = argparse.ArgumentParser(description="Inject disk faults manually.")
    parser.add_argument("type", nargs="?", default="bad", choices=["bad", "slow"], help="Fault type: 'bad' (default) or 'slow'")
    args = parser.parse_args()

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if args.type == "bad":
        inject_bad_disk(start_time)
    elif args.type == "slow":
        inject_slow_disk(start_time)
    
    collect_logs(start_time)

if __name__ == "__main__":
    main()
