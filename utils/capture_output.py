import sys
from datetime import datetime
from contextlib import contextmanager

class StreamTee:
    """A wrapper class to tee stdout/stderr to a file."""
    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        # Write to the original stream (e.g., terminal)
        self.stream.write(data)
        # Write to the log file
        self.file.write(data)
        # Flush both to ensure real-time logging
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, attr):
        """Pass any other calls to the original stream."""
        return getattr(self.stream, attr)

@contextmanager
def tee_output(log_filename):
    """
    A context manager to redirect stdout and stderr to a log file
    while still printing to the terminal.
    """
    log_file = None
    try:
        # Open the log file in append mode ('a') with line buffering (buffering=1)
        log_filename = f"{log_filename}-{datetime.now().strftime("[%Y-%m-%d][%H:%M:%S]")}"
        log_file = open(log_filename, 'w', encoding='utf-8', buffering=1)
        
        # Store original streams
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Create and assign the tee objects
        sys.stdout = StreamTee(original_stdout, log_file)
        sys.stderr = StreamTee(original_stderr, log_file)
        
        # Yield control back to the 'with' block
        yield
        
    finally:
        # Restore original streams
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
        if 'original_stderr' in locals():
            sys.stderr = original_stderr
            
        # Close the log file
        if log_file:
            log_file.close()