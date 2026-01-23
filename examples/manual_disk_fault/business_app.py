#!/usr/bin/env python3
import time
import os
import datetime
import sys
import logging
from common import MOUNT_POINT

# Setup logging to output to both console and file
logger = logging.getLogger("business_app")
logger.setLevel(logging.INFO)

# Formatter to just pass through the message (since we manually format timestamp)
formatter = logging.Formatter('%(message)s')

# File Handler (app.log)
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console Handler (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

FILE_PATH = os.path.join(MOUNT_POINT, "business_data.dat")

def main():
    logger.info(f"Starting Business App Simulation...")
    logger.info(f"Target file: {FILE_PATH}")
    logger.info("Press Ctrl+C to stop.")

    if not os.path.exists(MOUNT_POINT):
        logger.info(f"Error: Mount point {MOUNT_POINT} does not exist. Did you run setup_disk.py?")
        sys.exit(1)

    counter = 0
    while True:
        # Use Syslog format (Local Time) to match system logs
        # Format example: Jan 23 10:00:00
        now_local = datetime.datetime.now()
        timestamp = now_local.strftime('%b %d %H:%M:%S')
        
        try:
            # Simulate a critical business transaction (Write + Fsync)
            content = f"transaction_id={counter} timestamp={timestamp} payload={'x'*100}\n"
            
            with open(FILE_PATH, "a") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno()) # Critical: force sync to hit the disk immediately
            
            # Format: TIMESTAMP INFO message
            logger.info(f"{timestamp} {os.uname().nodename} business_app: Transaction {counter} committed successfully.")
            
        except OSError as e:
            # This is what we expect to see during fault injection
            logger.info(f"{timestamp} {os.uname().nodename} business_app: Transaction {counter} FAILED! Disk Error: {e}")
        except Exception as e:
            logger.info(f"{timestamp} {os.uname().nodename} business_app: Unexpected error: {e}")
        
        counter += 1
        time.sleep(1.0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nBusiness App stopped.")
