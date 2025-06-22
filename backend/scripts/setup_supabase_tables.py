#!/usr/bin/env python3
"""
Supabase Database Tables Setup Script

This script automatically creates the necessary database tables in your Supabase project.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, environment variables may not load properly")

def read_sql_file(file_path: str) -> str:
    """Read the SQL file content."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ SQL file not found: {file_path}")
        return None

async def setup_supabase_tables():
    """Set up Supabase database tables."""
    print("🚀 Setting up Supabase database tables...")
    
    # Check if Supabase credentials are available
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials not found in environment variables.")
        print("   Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
        return False
    
    try:
        from supabase import create_client
        
        # Create Supabase client
        print("🔗 Connecting to Supabase...")
        supabase = create_client(supabase_url, supabase_key)
        
        # Read the SQL setup script
        sql_file_path = backend_dir / "scripts" / "supabase_setup.sql"
        sql_content = read_sql_file(str(sql_file_path))
        
        if not sql_content:
            return False
        
        # Split SQL into individual statements
        sql_statements = []
        current_statement = ""
        
        for line in sql_content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('--') or not line:
                continue
            
            current_statement += line + " "
            
            # Check if statement ends with semicolon
            if line.endswith(';'):
                sql_statements.append(current_statement.strip())
                current_statement = ""
        
        # Execute SQL statements
        print("📝 Creating database tables and functions...")
        
        for i, statement in enumerate(sql_statements, 1):
            try:
                print(f"   Executing statement {i}/{len(sql_statements)}...")
                
                # Execute the SQL statement
                result = supabase.rpc('exec_sql', {'sql': statement}).execute()
                
                print(f"   ✅ Statement {i} executed successfully")
                
            except Exception as e:
                # Some statements might fail if they already exist, which is okay
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    print(f"   ⚠️  Statement {i} skipped (already exists)")
                else:
                    print(f"   ❌ Statement {i} failed: {e}")
                    print(f"   SQL: {statement[:100]}...")
        
        print("\n✅ Database setup completed!")
        
        # Test the setup by trying to access the table
        print("\n🧪 Testing database setup...")
        try:
            # Try to select from the table (should work even if empty)
            result = supabase.table("conversation_memories").select("count").limit(1).execute()
            print("✅ Table 'conversation_memories' is accessible!")
            
            # Test inserting a sample record
            sample_data = {
                'id': 'test_setup_user_test_conv',
                'user_id': 'test_setup_user',
                'conversation_id': 'test_conv',
                'messages': [{'role': 'user', 'content': 'Test message'}],
                'metadata': {'title': 'Setup Test', 'setup_test': True}
            }
            
            insert_result = supabase.table("conversation_memories").insert(sample_data).execute()
            print("✅ Sample data inserted successfully!")
            
            # Clean up test data
            supabase.table("conversation_memories").delete().eq("id", "test_setup_user_test_conv").execute()
            print("✅ Test data cleaned up!")
            
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            return False
        
        return True
        
    except ImportError:
        print("❌ Supabase library not installed. Please run: pip install supabase")
        return False
    except Exception as e:
        print(f"❌ Failed to set up Supabase tables: {e}")
        return False

def main():
    """Main function."""
    print("🗄️  Supabase Database Tables Setup")
    print("=" * 50)
    
    # Check environment
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    print(f"✅ Supabase URL: {'Set' if supabase_url else 'Not set'}")
    print(f"✅ Supabase Key: {'Set' if supabase_key else 'Not set'}")
    
    if not supabase_url or not supabase_key:
        print("\n❌ Please set your Supabase credentials in the .env file:")
        print("   SUPABASE_URL=https://your-project-id.supabase.co")
        print("   SUPABASE_KEY=your_supabase_anon_key_here")
        return
    
    # Run the setup
    success = asyncio.run(setup_supabase_tables())
    
    if success:
        print("\n🎉 Supabase database setup completed successfully!")
        print("   You can now use your Naarad AI Assistant with Supabase storage.")
        print("\n📊 To verify in Supabase dashboard:")
        print("   1. Go to your Supabase project dashboard")
        print("   2. Navigate to Table Editor")
        print("   3. You should see the 'conversation_memories' table")
        print("   4. Check the SQL Editor to see the functions and views")
    else:
        print("\n❌ Supabase database setup failed.")
        print("   Please check the error messages above and try again.")

if __name__ == "__main__":
    main() 