"""
Creative Relight - Logging Utility
Saves console output to log files for debugging
"""

import sys
import os
from datetime import datetime
from pathlib import Path

class TeeLogger:
    """
    Dual output: prints to console AND saves to file
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

class ErrorLogger:
    """
    Dual output for stderr: prints to console AND saves to file
    """
    def __init__(self, log_file):
        self.terminal = sys.stderr
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def setup_logging(log_dir="logs", prefix="creative_relight"):
    """
    Set up logging to save all console output to a file
    
    Args:
        log_dir: Directory to save logs (default: "logs")
        prefix: Prefix for log filename (default: "creative_relight")
    
    Returns:
        tuple: (log_file_path, tee_logger, error_logger)
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{prefix}_{timestamp}.log"
    
    # Write header
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Creative Relight - Log File\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    
    # Redirect stdout and stderr to log file (while still showing in console)
    tee_logger = TeeLogger(log_file)
    error_logger = ErrorLogger(log_file)
    
    sys.stdout = tee_logger
    sys.stderr = error_logger
    
    print(f"📝 Logging to: {log_file}")
    print()
    
    return str(log_file), tee_logger, error_logger

def stop_logging(tee_logger, error_logger):
    """
    Stop logging and restore normal stdout/stderr
    
    Args:
        tee_logger: TeeLogger instance
        error_logger: ErrorLogger instance
    """
    sys.stdout = tee_logger.terminal
    sys.stderr = error_logger.terminal
    tee_logger.close()
    error_logger.close()

def get_latest_log(log_dir="logs"):
    """
    Get the path to the most recent log file
    
    Args:
        log_dir: Directory containing logs
    
    Returns:
        Path to latest log file or None if no logs exist
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    
    log_files = list(log_path.glob("*.log"))
    if not log_files:
        return None
    
    # Sort by modification time, most recent first
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(log_files[0])

if __name__ == "__main__":
    # Test logging
    log_file, tee, err = setup_logging()
    
    print("This goes to both console and log file")
    print("[DEBUG] Testing debug message")
    print("[ERROR] Testing error message")
    
    stop_logging(tee, err)
    
    print(f"\nLog saved to: {log_file}")
    print(f"Latest log: {get_latest_log()}")
