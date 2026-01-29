#!/usr/bin/env python3
import time
import os
import datetime
import sys
import logging
import signal
from common import MOUNT_POINT

# Setup logging to output to both console and file
logger = logging.getLogger("order_service")
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

FILE_PATH = os.path.join(MOUNT_POINT, "orders.dat")

def timeout_handler(signum, frame):
    raise TimeoutError("Disk write timed out after 30 seconds")

def main():
    logger.info(f"Starting Order Service Simulation...")
    logger.info(f"Target file: {FILE_PATH}")
    logger.info("Press Ctrl+C to stop.")

    if not os.path.exists(MOUNT_POINT):
        logger.info(f"Error: Mount point {MOUNT_POINT} does not exist. Did you run setup_disk.py?")
        sys.exit(1)

    # Register signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)

    counter = 0
    while True:
        # Use Syslog format (Local Time) to match system logs
        # Format example: Jan 23 10:00:00
        now_local = datetime.datetime.now()
        timestamp = now_local.strftime('%b %d %H:%M:%S')
        
        try:
            # Simulate a critical business transaction (Write + Fsync)
            content = f"order_id={counter} timestamp={timestamp} item_id=ITEM-{counter%100} quantity=1 status=PLACED\n"
            
            # Set timeout for 15 seconds
            signal.alarm(15)
            
            with open(FILE_PATH, "a") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno()) # Critical: force sync to hit the disk immediately
            
            # Disable alarm if successful
            signal.alarm(0)
            
            # Format: TIMESTAMP INFO message
            logger.info(f"{timestamp} {os.uname().nodename} order_service: Order {counter} processed successfully.")
            
        except TimeoutError as e:
            logger.info(f"{timestamp} {os.uname().nodename} order_service: Order {counter} FAILED! Timeout: {e}")
        except OSError as e:
            try:
                signal.alarm(0) # Ensure alarm is disabled
            except TimeoutError:
                # If timeout occurs during error handling, ignore it to ensure logging proceeds
                pass
            # This is what we expect to see during fault injection
            logger.info(f"{timestamp} {os.uname().nodename} order_service: Order {counter} FAILED! Disk Error: {e}")
        except Exception as e:
            try:
                signal.alarm(0) # Ensure alarm is disabled
            except TimeoutError:
                pass
            logger.info(f"{timestamp} {os.uname().nodename} order_service: Unexpected error: {e}")
        
        counter += 1
        time.sleep(1.0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nOrder Service stopped.")
