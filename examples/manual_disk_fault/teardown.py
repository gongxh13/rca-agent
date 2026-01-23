#!/usr/bin/env python3
import time
from common import run_cmd, DM_DEV_NAME, MOUNT_POINT

def teardown():
    print("Cleaning up...")
    
    # 1. Unmount
    print(f"Unmounting {MOUNT_POINT}...")
    run_cmd(["sudo", "umount", "-f", MOUNT_POINT], check=False)
    
    # 2. Remove DM device
    print(f"Removing DM device {DM_DEV_NAME}...")
    run_cmd(["sudo", "dmsetup", "remove", DM_DEV_NAME], check=False)
    
    # 3. Remove scsi_debug
    print("Removing scsi_debug module...")
    # Try a few times as it might be busy
    for i in range(3):
        proc = run_cmd(["sudo", "modprobe", "-r", "scsi_debug"], check=False)
        if proc.returncode == 0:
            break
        time.sleep(1)
    
    print("Cleanup complete.")

if __name__ == "__main__":
    teardown()
