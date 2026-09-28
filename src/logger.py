import logging
import os
import sys
from datetime import datetime

LOG_FORMAT = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# Always log to stdout. On Lambda, stdout goes straight to CloudWatch.
handlers = [logging.StreamHandler(sys.stdout)]

# Also log to a file when running somewhere writable (your laptop, Docker locally).
# Serverless filesystems are read-only, so skip the file if we can't create it.
try:
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{datetime.now():%m_%d_%Y_%H_%M_%S}.log")
    handlers.append(logging.FileHandler(log_file))
except OSError:
    pass

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)