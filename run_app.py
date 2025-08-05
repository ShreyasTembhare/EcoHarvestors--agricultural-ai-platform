#!/usr/bin/env python3
"""
Run script for EcoHarvestors application.
Starts the Streamlit web application.
"""

import subprocess
import sys
import os

def main():
    """Run the EcoHarvestors Streamlit application."""
    try:
        # Change to the app directory
        app_dir = os.path.join(os.path.dirname(__file__), 'app')
        
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'
        ], cwd=app_dir, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main() 