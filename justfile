set shell := ["powershell.exe", "-c"]

# Default recipe - show available commands
default:
    @just --list

# Create initial database migration
migrate-create message:
    uv run alembic revision --autogenerate -m "{{message}}"

# Apply all pending migrations
migrate-up:
    uv run alembic upgrade head

# Rollback last migration
migrate-down:
    uv run alembic downgrade -1

# Start MinIO server
minio-start:
    podman run -p 9000:9000 -p 9001:9001 quay.io/minio/minio server /data --console-address ":9001"

# Stop MinIO server
minio-stop:
    podman stop minio

# Show migration history
migrate-history:
    uv run alembic history

# Show current migration status
migrate-status:
    uv run alembic current

# Initialize providers and models in database
init-providers:
    @echo "Initializing providers and models..."
    uv run python scripts/init_providers.py

# Check database tables
check-tables:
    uv run python -m scripts.check_tables

# Check if migrations are applied
check-migrations:
    @echo "Checking migration status..."
    uv run alembic current
    @echo "Checking if tables exist..."
    uv run python -m scripts.check_tables

# Full database setup (migrate + init providers)
setup-db: migrate-up init-providers
    @echo "Database setup complete!"

check-settings:
    uv run python -m app.config.settings

# Complete database initialization (start + migrate + init + check)
init-complete: start-db migrate-up init-providers check-tables
    @echo "🎉 Complete database initialization finished!"
    @echo "✅ Database is running and ready to use"

# Reset database (drop all tables and recreate)
reset-db: migrate-down migrate-up init-providers
    @echo "Database reset complete!"

# Install dependencies
install:
    uv sync

# Start PostgreSQL database container
start-db:
    docker run --name cortex-postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=cortex -p 5433:5432 -d postgres:15

query-db:
    docker exec -it cortex-postgres psql -U postgres -d cortex

# Stop PostgreSQL database container
stop-db:
    docker stop cortex-postgres

# Remove PostgreSQL database container
remove-db:
    docker rm cortex-postgres

# Restart PostgreSQL database container
restart-db: stop-db start-db
    @echo "Database restarted!"

# Show database status
db-status:
    docker ps --filter name=cortex-postgres

# Run the FastAPI server
serve:
    uv run python main.py

# Run tests
test:
    uv run python -m pytest tests/

# Format code
format:
    uv run black .
    uv run isort .

# Lint code
lint:
    uv run flake8 .
    uv run mypy .

# Clean up Python cache files
clean:
    Get-ChildItem -Path . -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Recurse -Name "*.pyc" | Remove-Item -Force
    @echo "Cleaned up Python cache files"

# Show database connection info
db-info:
    @echo "Database URI: {{env('DB_URI', 'Not set')}}"
    @echo "Current working directory: {{justfile_directory()}}"

# Help command
help:
    @echo "Available commands:"
    @echo "  migrate-create <message>  - Create new migration with message"
    @echo "  migrate-up               - Apply all pending migrations"
    @echo "  migrate-down             - Rollback last migration"
    @echo "  migrate-history          - Show migration history"
    @echo "  migrate-status           - Show current migration status"
    @echo "  init-providers           - Initialize providers and models"
    @echo "  check-tables             - Check database tables"
    @echo "  setup-db                - Full database setup (migrate + init)"
    @echo "  init-complete           - Complete DB init (start + migrate + init + check)"
    @echo "  reset-db                 - Reset database completely"
    @echo "  start-db                 - Start PostgreSQL container"
    @echo "  stop-db                  - Stop PostgreSQL container"
    @echo "  restart-db               - Restart PostgreSQL container"
    @echo "  remove-db                - Remove PostgreSQL container"
    @echo "  db-status                - Show database container status"
    @echo "  install                  - Install dependencies"
    @echo "  serve                    - Run FastAPI server"
    @echo "  test                     - Run tests"
    @echo "  format                   - Format code"
    @echo "  lint                     - Lint code"
    @echo "  clean                    - Clean Python cache files"
    @echo "  db-info                  - Show database connection info"
    @echo "  create-agents-table      - Create agents table manually"
    @echo "  help                     - Show this help message"
