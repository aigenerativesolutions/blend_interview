@echo off
REM MLOps Pipeline Execution Script for Windows
REM Soluciona problemas de volúmenes Docker en Windows

echo ==================================================
echo      MLOps Pipeline - Marketing Campaign
echo ==================================================

REM Obtener directorio actual
set CURRENT_DIR=%cd%
echo Directorio actual: %CURRENT_DIR%

REM Método 1: Ejecutar pipeline sin volúmenes y extraer resultados
echo.
echo [METODO 1] Ejecutando pipeline sin volúmenes...
echo.

REM Eliminar contenedor previo si existe
docker rm ml-pipeline-temp 2>nul

REM Ejecutar pipeline
docker run --name ml-pipeline-temp marketing-ml-pipeline python pipeline/run_pipeline.py --data-path data/marketing_campaign.csv --artifacts-dir artifacts

REM Verificar si el contenedor se ejecutó correctamente
if %errorlevel% neq 0 (
    echo ERROR: Pipeline falló
    docker rm ml-pipeline-temp 2>nul
    exit /b 1
)

REM Crear directorio artifacts si no existe
if not exist artifacts mkdir artifacts

REM Extraer resultados del contenedor
echo.
echo Extrayendo artifacts del contenedor...
docker cp ml-pipeline-temp:/app/artifacts ./

REM Limpiar contenedor
docker rm ml-pipeline-temp

echo.
echo ==================================================
echo Pipeline completado exitosamente
echo Artifacts guardados en: %CURRENT_DIR%\artifacts
echo ==================================================

REM Mostrar archivos generados
echo.
echo Archivos generados:
dir artifacts /s /b

pause