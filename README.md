# EEG Spectrogram Particles (CUDA + cuFFT + OpenGL)

Breve
- Visualizador 3D de espectrogramas en tiempo real usando CUDA/cuFFT y OpenGL (GLEW + freeglut).

Objetivo
- Repo reproducible para usuarios con GPU NVIDIA: compilación Release x64 en Visual Studio 2022, sin rutas absolutas, con argumentos CLI `--csv` y `--fs`.

Requisitos
- Windows 10/11 (x64)
- Visual Studio 2022 (Desktop C++ workload)
- CUDA Toolkit (ej. 11.8) y driver NVIDIA compatibles
- vcpkg (para instalar `glew` y `freeglut`)

Estructura recomendada
- `Particle_EEG_spectogram/` — código fuente (contiene `kernel.cu`)
- `datasets_csv/` — muestras pequeñas (incluye `sample.csv`)
- `.github/` — plantillas de issues/PR
- `vcpkg.json` — manifiesto de dependencias
- `LICENSE`, `README.md`, `.gitignore`

Instalación de dependencias con vcpkg (1 vez)
1. git clone https://github.com/microsoft/vcpkg.git
2. cd vcpkg
3. bootstrap-vcpkg.bat
4. .\vcpkg integrate install
5. .\vcpkg install glew freeglut --triplet x64-windows

Build en Visual Studio 2022
1. Abre la solución (.sln) en la raíz del repo.
2. Selecciona `Configuration = Release` y `Platform = x64`.
3. Opcional (recomendado para Run desde VS): Project → Properties → Debugging → Working Directory = `$(SolutionDir)`.
4. Opcional (pasar args): Project → Properties → Debugging → Command Arguments =
   `--csv datasets_csv/sample.csv --fs 256`
5. Build → Build Solution. Luego Start (F5) o ejecutar el exe.

Run desde terminal
- Desde la carpeta del ejecutable o desde cualquier carpeta (resolución robusta):
  `Particle_EEG_spectogram.exe --csv datasets_csv/sample.csv --fs 256`

Resolución de rutas (detalle importante)
- El ejecutable resuelve rutas relativas así:
  - Si pasas `--csv` con ruta absoluta, se usa tal cual.
  - Si pasas ruta relativa (o no pasas nada), el programa busca la carpeta `datasets_csv` subiendo desde el directorio del ejecutable (hasta 8 niveles). Ese directorio se considera `repo root` y se resuelve `repo_root/datasets_csv/…`.
- Por eso ejecutar desde `x64\Release` o desde otra carpeta funciona sin rutas absolutas.

CLI args soportados
- `--csv <ruta>`  : ruta al CSV (relativa o absoluta). Default: `datasets_csv/sample.csv`.
- `--fs <valor>`  : frecuencia de muestreo (float). Default: `256`.

Dataset y reproducibilidad
- Incluye `datasets_csv/sample.csv` (muy pequeño) para pruebas rápidas.
- Regla de oro: no subir datasets grandes al repo. Proveer scripts de descarga o usar Git LFS para datos grandes.

Problemas comunes y soluciones rápidas
- "No pude leer CSV": revisa la salida en consola — verás líneas:
  `Repo root (detected): <ruta>`
  `Using CSV: <ruta completa>`
  Si la carpeta detectada no es la esperada, ajustar Working Directory en VS a `$(SolutionDir)` o pasar `--csv` con ruta absoluta.
- "No encuentra glew/freeglut" al ejecutar: asegúrate de haber corrido `vcpkg integrate install` y, si hace falta, copia los .dll desde `vcpkg_installed\x64-windows\bin` a la carpeta del exe o añade esa carpeta al PATH.
- Errores de `<filesystem>`: si el compilador falla por `std::filesystem` con C++14, cambia temporalmente la standard a C++17 en Project → C/C++ → Language.

Checklist rápido (verificación)
- [ ] Compila en Release x64
- [ ] `datasets_csv/sample.csv` se carga correctamente
- [ ] `Repo root (detected)` y `Using CSV` muestran rutas válidas
- [ ] No rutas absolutas hardcodeadas en el código

Contribuir
- Usa `vcpkg.json` si añades dependencias nuevas.
- Abre PR con descripción, pasos para compilar y prueba.

Licencia
- MIT (archivo `LICENSE`).
+++++++++++++++++++++++++++++++++++++++++++++        