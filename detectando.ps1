# simple jaja
#cd d:/cesareots/uev-master-ia/0-tfm/proy_tfm/
#& C:/Users/cesareots/.conda/envs/uev-tfm/python.exe detectando.py




# --- Configuración y Rutas (Buenas Prácticas) ---
# ruta base del entorno Conda
$condaEnvBasePath = "C:/Users/cesareots/.conda/envs/uev-tfm"
# ruta base del proyecto
$projectBasePath = "d:/cesareots/uev-master-ia/0-tfm/proy_tfm"
# ruta script dentro del proyecto
$pythonScriptPath = "detectando.py"

# ruta completa al ejecutable de Python de forma robusta
$pythonExecutable = Join-Path -Path $condaEnvBasePath -ChildPath "python.exe"
# ruta completa al script Python
$pythonScript = Join-Path -Path $projectBasePath -ChildPath $pythonScriptPath

# --- Verificaciones de Existencia ---
# Verifica si el ejecutable de Python existe antes de intentar usarlo
if (-not (Test-Path -Path "$pythonExecutable")) {
    Write-Error "Error: El ejecutable de Python NO se encontro en '$pythonExecutable'. Verifica la ruta del entorno Conda."
    exit 1
}

# Verifica si el script Python existe
if (-not (Test-Path -Path "$pythonScript")) {
    Write-Error "Error: El script Python principal NO se encontro en '$pythonScript'. Verifica la ruta del script."
    exit 1
}

# --- Ejecutar el Script Python ---
# operador '&' y rutas entrecomilladas.
cd "$projectBasePath"
& "$pythonExecutable" "$pythonScriptPath"

# --- Manejo Básico de Errores del Script Python ---
# $LASTEXITCODE contiene el código de salida del último programa externo ejecutado. 0 generalmente significa éxito, cualquier otro número indica un error.
if ($LASTEXITCODE -ne 0) {
    Write-Error "El script Python '$pythonScript' termino con codigo de salida $LASTEXITCODE. Puede que haya habido un error interno en el script."
    # Puedes añadir lógica adicional aquí, como notificar o registrar el error.
} else {
    Write-Host "El script Python '$pythonScript' parece haber terminado correctamente (codigo de salida 0)."
}

# --- Evitar que la ventana se cierre automáticamente ---
# Añade una pausa al final para que puedas ver la salida y los mensajes de error.
Write-Host "`nPresiona Enter para cerrar esta ventana..."
Read-Host # Este comando espera a que el usuario presione la tecla Enter
