"""
Robust Tee implementation for simultaneous console and file output.
Handles encoding errors gracefully to prevent OSError on Windows.
"""

import sys


class Tee:
    """
    Tee stdout/stderr to both console and file with robust encoding handling.
    
    Usage:
        with Tee('output.log'):
            print("This goes to both console and file")
    """
    
    def __init__(self, path, mode='a'):
        """
        Initialize Tee.
        
        Args:
            path: Path to log file
            mode: File mode ('a' for append, 'w' for write)
        """
        self.console = sys.__stdout__
        self.file = open(path, mode, encoding='utf-8', errors='replace')
    
    def write(self, s):
        """Write to both console and file with encoding error handling."""
        # Write to console, never crash on encoding
        try:
            self.console.write(s)
        except (UnicodeEncodeError, OSError):
            # Fallback: encode to console encoding with replacement
            console_enc = getattr(self.console, 'encoding', 'cp1252') or 'cp1252'
            try:
                safe_str = s.encode(console_enc, 'replace').decode(console_enc, 'ignore')
                self.console.write(safe_str)
            except Exception:
                # Last resort: skip problematic output
                pass
        
        # Write to file safely
        try:
            self.file.write(s)
        except Exception:
            # Last-ditch: replace problematic chars
            try:
                safe_str = s.encode('utf-8', 'replace').decode('utf-8')
                self.file.write(safe_str)
            except Exception:
                pass
    
    def flush(self):
        """Flush both console and file."""
        try:
            self.console.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass
    
    def close(self):
        """Close the file handle."""
        try:
            self.file.close()
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager entry - redirect stdout/stderr."""
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, *exc):
        """Context manager exit - restore stdout/stderr."""
        sys.stdout = self._old_out
        sys.stderr = self._old_err
        self.close()
        return False  # Don't suppress exceptions


if __name__ == "__main__":
    # Test the tee
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    try:
        print("Testing Tee class...")
        with Tee(temp_path):
            print("This line goes to both console and file")
            print("Another line with numbers: 123.45")
            print("Unicode test (should be safe): EUR/USD")
        
        print("\nReading back from file:")
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
        
        print("[OK] Tee test passed!")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
