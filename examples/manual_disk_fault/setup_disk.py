#!/usr/bin/env python3
import sys
import time
from common import run_cmd, find_scsi_debug_device, get_device_sectors, DM_DEV_NAME, MOUNT_POINT, DM_DEV_PATH

def setup():
    print(">>> Step 1: Loading scsi_debug module...")
    # Clean up first just in case
    run_cmd(["sudo", "modprobe", "-r", "scsi_debug"], check=False)
    # Load with 1GB size
    run_cmd(["sudo", "modprobe", "scsi_debug", "dev_size_mb=1024"], check=True)
    
    # Wait for device to appear
    time.sleep(1)
    
    try:
        device = find_scsi_debug_device()
        print(f"Found scsi_debug device at: {device}")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    sectors = get_device_sectors(device)
    print(f"Device size: {sectors} sectors")

    print(f">>> Step 2: Creating DM Linear device '{DM_DEV_NAME}'...")
    # Check if exists
    proc = run_cmd(["sudo", "dmsetup", "info", DM_DEV_NAME], check=False, capture_output=True)
    if proc.returncode == 0:
        print(f"DM device {DM_DEV_NAME} already exists. Removing it...")
        run_cmd(["sudo", "dmsetup", "remove", DM_DEV_NAME], check=True)

    # Create linear mapping: 0 <sectors> linear <device> 0
    table = f"0 {sectors} linear {device} 0"
    run_cmd(["sudo", "dmsetup", "create", DM_DEV_NAME], input=table, check=True)
    
    print(f">>> Step 3: Formatting {DM_DEV_PATH}...")
    run_cmd(["sudo", "mkfs.ext4", "-F", DM_DEV_PATH], check=True)

    print(f">>> Step 4: Mounting to {MOUNT_POINT}...")
    run_cmd(["sudo", "mkdir", "-p", MOUNT_POINT], check=True)
    run_cmd(["sudo", "mount", DM_DEV_PATH, MOUNT_POINT], check=True)
    
    # Fix permissions so our app can write without sudo if needed, or just chmod it
    run_cmd(["sudo", "chmod", "777", MOUNT_POINT], check=True)

    print("\nSUCCESS: Disk setup complete.")
    print(f"Device: {DM_DEV_PATH}")
    print(f"Mounted at: {MOUNT_POINT}")
    print("You can now run 'python3 business_app.py'")

if __name__ == "__main__":
    setup()
