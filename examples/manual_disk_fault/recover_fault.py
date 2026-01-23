#!/usr/bin/env python3
import sys
from common import run_cmd, find_scsi_debug_device, get_device_sectors, DM_DEV_NAME, DM_DEV_PATH

def recover():
    print(f"Recovering {DM_DEV_NAME} to normal state...")

    # 1. Find underlying device again
    try:
        device = find_scsi_debug_device()
        print(f"Found backing device: {device}")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Get size
    try:
        sectors = get_device_sectors(DM_DEV_PATH)
        print(f"Target device size: {sectors} sectors")
    except Exception as e:
        print(f"Error getting device size: {e}")
        sys.exit(1)

    # 3. Suspend
    print("Suspending device...")
    run_cmd(["sudo", "dmsetup", "suspend", DM_DEV_NAME], check=True)

    # 4. Load linear table
    linear_table = f"0 {sectors} linear {device} 0"
    print(f"Loading linear table: {linear_table}")
    try:
        run_cmd(["sudo", "dmsetup", "load", DM_DEV_NAME, "--table", linear_table], check=True)
    except Exception as e:
        print(f"Failed to load table: {e}")
        print("Resuming anyway...")
        run_cmd(["sudo", "dmsetup", "resume", DM_DEV_NAME], check=False)
        sys.exit(1)

    # 5. Resume
    print("Resuming device (Normal State)...")
    run_cmd(["sudo", "dmsetup", "resume", DM_DEV_NAME], check=True)

    print("\nRECOVERY COMPLETE. I/O should succeed now.")

if __name__ == "__main__":
    recover()
