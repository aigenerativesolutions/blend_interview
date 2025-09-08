# MLOps Pipeline Execution Script for Windows PowerShell
# Soluciona problemas de volúmenes Docker en Windows

Write-Host "==================================================" -ForegroundColor Green
Write-Host "     MLOps Pipeline - Marketing Campaign" -ForegroundColor Green  
Write-Host "==================================================" -ForegroundColor Green

# Obtener directorio actual
$CurrentDir = Get-Location
Write-Host "Directorio actual: $CurrentDir" -ForegroundColor Yellow

# Método 1: Volúmenes con sintaxis PowerShell
Write-Host "`n[METODO 1] Intentando con volúmenes PowerShell..." -ForegroundColor Cyan

try {
    docker run --rm `
        -v "${CurrentDir}/data:/app/data" `
        -v "${CurrentDir}/artifacts:/app/artifacts" `
        marketing-ml-pipeline `
        python pipeline/run_pipeline.py --data-path data/marketing_campaign.csv --artifacts-dir artifacts
    
    Write-Host "Pipeline completado con volúmenes" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "Método 1 falló, intentando método alternativo..." -ForegroundColor Yellow
}

# Método 2: Sin volúmenes, extraer con docker cp
Write-Host "`n[METODO 2] Ejecutando sin volúmenes..." -ForegroundColor Cyan

# Eliminar contenedor previo si existe
docker rm ml-pipeline-temp 2>$null

# Ejecutar pipeline
$result = docker run --name ml-pipeline-temp marketing-ml-pipeline python pipeline/run_pipeline.py --data-path data/marketing_campaign.csv --artifacts-dir artifacts

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Pipeline falló" -ForegroundColor Red
    docker rm ml-pipeline-temp 2>$null
    exit 1
}

# Crear directorio artifacts si no existe
if (!(Test-Path "artifacts")) {
    New-Item -ItemType Directory -Path "artifacts"
}

# Extraer resultados del contenedor
Write-Host "`nExtrayendo artifacts del contenedor..." -ForegroundColor Cyan
docker cp ml-pipeline-temp:/app/artifacts ./

# Limpiar contenedor
docker rm ml-pipeline-temp

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "Pipeline completado exitosamente" -ForegroundColor Green
Write-Host "Artifacts guardados en: $CurrentDir\artifacts" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Mostrar archivos generados
Write-Host "`nArchivos generados:" -ForegroundColor Yellow
Get-ChildItem -Path "artifacts" -Recurse | Select-Object FullName