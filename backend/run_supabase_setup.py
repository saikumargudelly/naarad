#!/usr/bin/env python3
"""
Simple script to run Supabase setup with proper .env loading
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Now run the Supabase setup
if __name__ == "__main__":
    print("🔧 Loading environment variables...")
    print(f"✅ SUPABASE_URL: {'Set' if os.getenv('SUPABASE_URL') else 'Not set'}")
    print(f"✅ SUPABASE_KEY: {'Set' if os.getenv('SUPABASE_KEY') else 'Not set'}")
    
    if not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_KEY'):
        print("❌ Supabase credentials not found in .env file")
        sys.exit(1)
    
    print("🚀 Running Supabase setup...")
    
    # Import and run the setup
    from scripts.supabase_setup import main
    main() 