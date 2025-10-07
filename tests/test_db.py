# test_db.py
import sqlalchemy
from sqlalchemy.exc import OperationalError
from app.config.settings import settings # <-- Import your settings object

print(f"Attempting to connect to the database at: {settings.DB_URI}")

try:
    engine = sqlalchemy.create_engine(str(settings.DB_URI))
    with engine.connect() as connection:
        print("✅ Connection successful!")
except OperationalError as e:
    print("❌ Connection failed!")
    print(f"   Error details: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")