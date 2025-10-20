"""
Quick script to test database connection
"""
from urllib.parse import quote_plus
import psycopg2

# Your credentials
user = "postgres.ugowtgrsotloudlludcm"
password = ",j3Kn#wd_EKH"
host = "aws-1-us-east-1.pooler.supabase.com"
port = 5432
database = "postgres"

# URL encode the password
encoded_password = quote_plus(password)
print(f"Original password: {password}")
print(f"Encoded password: {encoded_password}")

# Test connection
try:
    print("\n🔄 Attempting to connect to Supabase...")
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        sslmode="require"  # Try with SSL
    )
    print("✅ Connection successful!")
    print(f"Database version: {conn.server_version}")
    
    # Test a query
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"PostgreSQL version: {version[0]}")
    
    cursor.close()
    conn.close()
    
    # Print the correct connection string
    print("\n✅ Use this connection string in your .env file:")
    print(f"SUPABASE_DB_URI=postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}")
    
except psycopg2.OperationalError as e:
    print(f"❌ Connection failed: {e}")
    print("\n💡 Possible issues:")
    print("   1. Your Supabase project might be paused (check dashboard)")
    print("   2. The credentials might be incorrect")
    print("   3. Your IP might not be whitelisted")
    print("\n📝 Suggested fix:")
    print("   - Go to https://supabase.com/dashboard")
    print("   - Check if your project is active")
    print("   - Get a new connection string from Settings → Database")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

