# PowerShell script to create agents table
Write-Host "Creating agents table..." -ForegroundColor Green

# Run the SQL script
docker exec -i cortex-postgres psql -U postgres -d cortex < scripts/create_agents_table.sql

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Agents table created successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Error creating agents table" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Database setup complete!" -ForegroundColor Green
