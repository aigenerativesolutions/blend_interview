# MLOps Pipeline - Docker Runner PowerShell
Write-Host "🐳 MLOps Pipeline - Docker Runner" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker version *>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not available"
    }
} catch {
    Write-Host "❌ Docker no está disponible. Asegúrate de que Docker Desktop esté ejecutándose." -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""
Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow
docker build -f Dockerfile.pipeline -t mlops-pipeline .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""
Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
Write-Host ""

do {
    Write-Host "Selecciona qué ejecutar:" -ForegroundColor White
    Write-Host "1. Validar pipeline (recomendado primero)" -ForegroundColor White
    Write-Host "2. Ejecutar pipeline completo" -ForegroundColor White
    Write-Host "3. Debug tests" -ForegroundColor White
    Write-Host "4. Modo interactivo (bash)" -ForegroundColor White
    Write-Host "5. Salir" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "Ingresa tu opción (1-5)"
    
    switch ($choice) {
        "1" {
            Write-Host ""
            Write-Host "🧪 Ejecutando validación de pipeline..." -ForegroundColor Yellow
            docker run --rm -v "${PWD}\artifacts:/app/artifacts" mlops-pipeline python validate_pipeline.py
        }
        "2" {
            Write-Host ""
            Write-Host "🚀 Ejecutando pipeline completo..." -ForegroundColor Green
            docker run --rm -v "${PWD}\artifacts:/app/artifacts" mlops-pipeline python quick_start.py
        }
        "3" {
            Write-Host ""
            Write-Host "🔍 Ejecutando debug tests..." -ForegroundColor Yellow
            docker run --rm -v "${PWD}\artifacts:/app/artifacts" mlops-pipeline python debug_test.py
        }
        "4" {
            Write-Host ""
            Write-Host "🐚 Modo interactivo - puedes ejecutar comandos dentro del container" -ForegroundColor Cyan
            Write-Host "Para salir del container, escribe: exit" -ForegroundColor Cyan
            docker run --rm -it -v "${PWD}\artifacts:/app/artifacts" mlops-pipeline bash
        }
        "5" {
            Write-Host ""
            Write-Host "👋 ¡Hasta luego!" -ForegroundColor Green
            break
        }
        default {
            Write-Host "Opción inválida, intenta de nuevo." -ForegroundColor Red
        }
    }
    Write-Host ""
} while ($choice -ne "5")