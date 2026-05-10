#!/usr/bin/env python3
import os
import time
from pathlib import Path

SEARCH_DIR = Path(__file__).parent.parent
MAX_AGE_DAYS = 7
MAX_AGE_SECONDS = MAX_AGE_DAYS * 86400

now = time.time()
deleted = []

for log_file in SEARCH_DIR.rglob("*.log"):
    try:
        age = now - log_file.stat().st_mtime
        if age > MAX_AGE_SECONDS:
            log_file.unlink()
            deleted.append(log_file)
    except OSError as e:
        print(f"Error processing {log_file}: {e}")

if deleted:
    print(f"Deleted {len(deleted)} log file(s) older than {MAX_AGE_DAYS} days:")
    for f in deleted:
        print(f"  {f}")
else:
    print(f"No log files older than {MAX_AGE_DAYS} days found.")
