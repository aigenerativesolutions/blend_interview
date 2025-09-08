@echo off
echo 🐳 MLOps Pipeline - Docker Runner
echo ================================

echo.
echo 🔨 Building Docker image...
docker build -f Dockerfile.pipeline -t mlops-pipeline .

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo.
echo ✅ Docker image built successfully!
echo.

:menu
echo Selecciona qué ejecutar:
echo 1. Validar pipeline (recomendado primero)
echo 2. Ejecutar pipeline completo
echo 3. Debug tests
echo 4. Modo interactivo (bash)
echo 5. Salir
echo.
set /p choice="Ingresa tu opción (1-5): "

if "%choice%"=="1" goto validate
if "%choice%"=="2" goto fullpipeline
if "%choice%"=="3" goto debug
if "%choice%"=="4" goto interactive
if "%choice%"=="5" goto end
echo Opción inválida, intenta de nuevo.
goto menu

:validate
echo.
echo 🧪 Ejecutando validación de pipeline...
docker run --rm -v "%cd%\artifacts:/app/artifacts" mlops-pipeline python validate_pipeline.py
goto menu

:fullpipeline
echo.
echo 🚀 Ejecutando pipeline completo...
docker run --rm -v "%cd%\artifacts:/app/artifacts" mlops-pipeline python quick_start.py
goto menu

:debug
echo.
echo 🔍 Ejecutando debug tests...
docker run --rm -v "%cd%\artifacts:/app/artifacts" mlops-pipeline python debug_test.py
goto menu

:interactive
echo.
echo 🐚 Modo interactivo - puedes ejecutar comandos dentro del container
echo Para salir del container, escribe: exit
docker run --rm -it -v "%cd%\artifacts:/app/artifacts" mlops-pipeline bash
goto menu

:end
echo.
echo 👋 ¡Hasta luego!
pause