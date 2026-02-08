# ANÁLISIS DETALLADO DEL CÓDIGO kernel.cu

## SECCIÓN 1: HEADERS Y MACROS DE SEGURIDAD (líneas 1-80)
```cpp
// ===== Windows macro safety =====
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
```
**Propósito**: Evitar conflictos de macros de Windows con min/max de la STL.

```cpp
#define GLEW_NO_GLU
#include <GL/glew.h>
#include <GL/freeglut.h>
```
**Propósito**: Incluir GLEW (OpenGL Extension Wrangler) y GLUT (graphics toolkit).

```cpp
#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { ... } \
} while(0)

#define CUFFT_CHECK(call) do { \
  cufftResult r = (call); \
  if (r != CUFFT_SUCCESS) { ... } \
} while(0)
```
**Propósito**: Macros para verificar errores CUDA y cuFFT. Si hay error, imprime y sale del programa.

---

## SECCIÓN 2: VARIABLES GLOBALES - TRIPLE BUFFERING (líneas 83-112)

```cpp
static constexpr int kBuffers = 3;
```
**Propósito**: 3 buffers para triple buffering (CPU comienza, 1 en GPU, 1 lista para presentar).

```cpp
static cudaStream_t gStream = nullptr;
static cudaEvent_t  gReadyEvent[kBuffers] = { nullptr, nullptr, nullptr };
static bool         gHasData[kBuffers] = { false, false, false };
```
**Propósito**: 
- `gStream`: Stream CUDA no-bloqueante para operaciones asíncronas.
- `gReadyEvent[i]`: Evento que se marca cuando el buffer `i` está listo.
- `gHasData[i]`: Flag indicando si el buffer `i` tiene datos válidos.

```cpp
static int gPresent = 0;    // Buffer actualmente visible
static int gQueued = 0;     // Buffer en cola para mostrar
static int gHeadForBuffer[kBuffers] = { 0,0,0 };  // Head del ring buffer para cada uno
```
**Propósito**: Gestión de qué buffer se ve, qué se procesa.

---

## SECCIÓN 3: DATOS DE ESCALA Y HOVER (líneas 114-156)

```cpp
static float* h_scalePinned = nullptr;  // 2 floats (pinned memory = CPU acceso rápido a GPU)
static float  gScaleHost[2] = { -20.0f, 40.0f };  // Escala dB min/max
static cudaEvent_t gScaleEvent = nullptr;  // Sincronización de escala
```
**Propósito**: Pasar escala dB desde GPU a CPU de manera asíncrona y eficiente.

```cpp
static float gRmsRaw = 0.0f;    // RMS de señal original
static float gRmsFilt = 0.0f;   // RMS de señal filtrada
static float gRmsRemoved = 0.0f;  // RMS de ruido removido
static float gRemovedPct = 0.0f;  // % de ruido removido
```
**Propósito**: Métricas mostradas en HUD para feedback del filtro.

```cpp
static float gPlaySpeed = 0.25f;  // Velocidad de reproducción (0.25 = 4x más lento)
static float gSampleFrac = 0.0f;  // Fracción de muestra (para avance suave)
```
**Propósito**: Control de velocidad de playback.

---

## SECCIÓN 4: PICKING (HOVER CON RATÓN) (líneas 158-185)

```cpp
static GLuint pickFBO = 0;      // Framebuffer Object para picking
static GLuint pickTex = 0;      // Texture con IDs codificados en RGB
static GLuint pickDepth = 0;    // Renderbuffer de profundidad
static GLuint pickProg = 0;     // Programa shader para picking
static GLint  uPickMVPLoc = -1; // Location del uniform MVP en el shader
static GLint  uPickPointSizeLoc = -1;  // Location del uniform point size
```
**Propósito**: Sistema de picking: renderizar a una texturas especial donde cada píxel contiene el ID de la partícula más cercana.

```cpp
static std::vector<int> gColSample0;  // gColSample0[col] = índice de sample del primer FFT en esa columna
static int gPickRadius = 6;  // Radio de búsqueda alrededor del ratón (2R+1)x(2R+1)
```
**Propósito**: Metadatos para saber qué muestra corresponde a cada columna visible.

```cpp
static float* h_hoverPinned = nullptr;  // 1 float (pinned) para el valor dB en hover
static cudaEvent_t gHoverEvent = nullptr;  // Sincronización de lectura hover
static bool        gHoverPending = false;  // ¿Esperando lectura desde GPU?
static bool        gHoverValid = false;    // ¿Hover válido?
static float       gHoverHz = 0.0f;        // Frecuencia en hover
static float       gHoverDb = 0.0f;        // Amplitud dB en hover
static int         gHoverCol = -1, gHoverRow = -1;  // Coordenadas del hover
static int         gMouseX = 0, gMouseY = 0;  // Posición del ratón
static float gHoverTimeSec = 0.0f;  // Tiempo en segundos del punto hover
static int   gHoverDispCol = -1;    // Columna visible (0..Wvis-1)
```
**Propósito**: Almacenar datos del punto donde está el hover (útil para tooltip).

---

## SECCIÓN 5: ESTRUCTURA SETTINGS (líneas 188-218)

```cpp
struct Settings {
    bool paused = false;  // ¿Pausado?
    bool showHUDText = true;  // ¿Mostrar HUD?
    
    bool  autoGain = true;    // ¿Escala automática dB?
    float dbMin = -30.0f;     // Escala min si autoGain OFF
    float dbMax = 30.0f;      // Escala max si autoGain OFF
    
    float gamma = 1.0f;       // Corrección gamma (1.0 = linear)
    int   cmap = 0;           // 0=heat, 1=gray
    
    float jitter = 0.06f;     // Amplitud de movimiento de partículas
    int channel = 1;          // Canal del CSV a visualizar
    
    // ===== filtro previo a FFT =====
    bool  preFilter = true;   // ¿Aplicar filtro?
    bool  showFiltered = true;  // ¿Mostrar filtrada o raw?
    bool  removeMean = true;    // ¿Remover DC?
    float fs = 256.0f;        // Frecuencia de muestreo (Hz)
    float lowHz = 45.0f;      // Filtro paso bajo (Hz)
    bool  useHighpass = true;  // ¿Usar filtro paso alto?
    float highHz = 0.5f;       // Filtro paso alto (Hz)
    
    // ===== 3D look =====
    float zScale = 0.9f;      // Escala de amplitud en Z
    int   timeGridEveryCols = 32;  // Gridlines cada N columnas
    bool  show3DGrid = true;   // ¿Mostrar grid 3D?
};
static Settings g;  // Instancia global
```
**Propósito**: Centralizar todas las opciones del programa.

---

## SECCIÓN 6: FUNCIONES CSV (líneas 220-318)

```cpp
static inline void trim_inplace(std::string& tok) {
    // Elimina espacios en blanco al inicio y final
    tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(),
        [](unsigned char ch) { return !std::isspace(ch); }));
    tok.erase(std::find_if(tok.rbegin(), tok.rend(),
        [](unsigned char ch) { return !std::isspace(ch); }).base(), tok.end());
}
```
**Propósito**: Limpieza de strings.

```cpp
static inline bool looks_like_number(const std::string& s) {
    for (char c : s) if ((c >= '0' && c <= '9') || c == '-' || c == '.' || c == '+') return true;
    return false;
}
```
**Propósito**: Detectar si un token es un número (para saltar headers).

```cpp
static inline char detect_delim(const std::string& line) {
    size_t c1 = std::count(line.begin(), line.end(), ',');
    size_t c2 = std::count(line.begin(), line.end(), ';');
    return (c2 > c1) ? ';' : ',';
}
```
**Propósito**: Detectar si el CSV usa `,` o `;` como delimitador.

```cpp
static bool load_csv_channels(const std::string& path, std::vector<std::vector<float>>& channels_out) {
    // Lee CSV multi-canal en `channels_out`
    // Detecta delimiter, salta headers, valida números
    // Retorna true si leyó al menos 1 columna
}
```
**Propósito**: Cargar datos EEG del CSV.

---

## SECCIÓN 7: FILTRO BIQUAD (líneas 320-400)

```cpp
struct Biquad {
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;  // Coeficientes IIR
    float z1 = 0, z2 = 0;  // Estado del filtro
    inline float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};
```
**Propósito**: Filtro IIR de 2do orden (butterworth).

```cpp
static Biquad make_lowpass(float fs, float fc, float Q = 0.70710678f) {
    // Crea un filtro paso bajo con:
    // fs = frecuencia de muestreo
    // fc = frecuencia de corte
    // Q = factor de calidad (0.707 para butterworth)
}

static Biquad make_highpass(float fs, float fc, float Q = 0.70710678f) {
    // Análogo para paso alto
}

static void preprocess_filter_inplace(std::vector<float>& x, float fs, ...) {
    // Aplica: remover DC -> HP -> 2x LP
}
```
**Propósito**: Preprocesamiento de señal antes del FFT.

---

## SECCIÓN 8: VARIABLES STFT (líneas 402-475)

```cpp
static int   fftSize = 1024;  // Tamaño FFT
static int   hop = 64;        // Desplazamiento entre ventanas
static int   nBinsFull = 0;   // Bins totales FFT (fftSize/2 + 1)
static int   maxBinVis = 0;   // Max bin a visualizar (limitado a ~45 Hz)
static float binHz = 0.0f;    // Hz por bin
static float fTopShownHz = 45.0f;  // Max Hz mostrado

static int Wvis = 256;  // Ancho del espectrograma (# columnas visibles)
static int H = 0;       // Alto del espectrograma (# bins frecuencia)
static int Nvis = 0;    // Total partículas = Wvis * H

static const int stepFrames = 4;  // Procesar 4 frames por tick
static int B = stepFrames;

static int startSample = 0;      // Índice del primer sample a procesar
static int maxStartSample = 0;   // Max válido de startSample
static int nSamples = 0;         // Total de samples del CSV
static int gHead = 0;            // Posición "escribir" del ring buffer
```
**Propósito**: Parámetros STFT.

```cpp
static std::vector<std::vector<float>> h_channels;  // Todos los canales del CSV
static std::vector<float> h_signal_raw;   // Canal seleccionado (sin filtro)
static std::vector<float> h_signal_filt;  // Canal seleccionado (filtrado)
static std::vector<float> h_signal;       // Puntero a raw o filt según g.showFiltered
```
**Propósito**: Datos de señal en CPU.

```cpp
static float* d_signal = nullptr;  // Signal en GPU
static float* d_win = nullptr;     // Ventana Hann en GPU

static float* d_frames[kBuffers] = { nullptr,nullptr,nullptr };  // [B, fftSize]
static cufftComplex* d_fft[kBuffers] = { nullptr,nullptr,nullptr };  // [B, nBinsFull]
static float* d_spec_db[kBuffers] = { nullptr,nullptr,nullptr };  // [B, H] (dB)

static cufftHandle plan = 0;  // Plan cuFFT batched
static float* d_hist_db = nullptr;  // Historial total [Wvis, H] (ring buffer)

static float2* d_off = nullptr;  // Offset de movimiento de partículas
static float2* d_vel = nullptr;  // Velocidad de movimiento

static float* d_minmax[kBuffers] = { ... };  // Min/max locales para cada buffer
static float* d_scale = nullptr;  // Escala global [min, max]
```
**Propósito**: Memoria GPU para STFT y procesamiento.

```cpp
static float gInvWinL2 = 1.0f;  // 1 / L2-norm de la ventana Hann
```
**Propósito**: Normalización de energía (COLA - Constant Overlap-Add).

---

## SECCIÓN 9: MATH - VEC3 Y MAT4 (líneas 477-650)

```cpp
struct Vec3 {
    float x, y, z;
    Vec3() :x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) :x(X), y(Y), z(Z) {}
};
```
**Propósito**: Vector 3D básico.

```cpp
static inline Vec3 operator+(const Vec3& a, const Vec3& b) { ... }  // Suma
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { ... }  // Resta
static inline Vec3 operator*(const Vec3& a, float s) { ... }  // Escala
static inline float dot(const Vec3& a, const Vec3& b) { ... }  // Producto escalar
static inline Vec3 cross(const Vec3& a, const Vec3& b) { ... }  // Producto cruz
static inline float vlen(const Vec3& v) { ... }  // Magnitud
static inline Vec3 normalize(const Vec3& v) { ... }  // Normalización
```
**Propósito**: Operaciones básicas de álgebra linear.

```cpp
struct Mat4 {
    float m[16];  // Column-major (OpenGL)
};
```
**Propósito**: Matriz 4x4 para transformaciones 3D.

```cpp
static Mat4 mat4_identity() { ... }  // Matriz identidad
static Mat4 mat4_mul(const Mat4& A, const Mat4& B) { ... }  // Multiplicación
static Mat4 mat4_perspective(float fovyRad, float aspect, float zNear, float zFar) { ... }
static Mat4 mat4_lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) { ... }
```
**Propósito**: Transformaciones de cámara (proyección + view).

```cpp
static bool mat4_inverse(const Mat4& M, Mat4& invOut) { ... }  // Inversa 4x4
static Vec3 unprojectNDC(const Mat4& invMVP, float ndcX, float ndcY, float ndcZ) { ... }
```
**Propósito**: Para picking (pasar de coordenadas pantalla a mundo).

---

## SECCIÓN 10: CAMERA (líneas 652-707)

```cpp
static int winW = 1100, winH = 700;  // Dimensiones ventana

static float gYaw = 0.7f;     // Rotación horizontal (radianes)
static float gPitch = 0.35f;  // Rotación vertical
static float gDist = 3.0f;    // Distancia a target
static Vec3  gTarget = Vec3(0.0f, 0.0f, 0.0f);  // Punto de enfoque

static bool gMouseL = false;  // ¿Ratón izquierdo presionado?
static bool gMouseR = false;  // ¿Ratón derecho presionado?
static int  gLastMX = 0, gLastMY = 0;  // Última posición ratón
```
**Propósito**: Controles de cámara orbital (esferas).

```cpp
static Mat4 buildMVP() {
    // Construye Projection * View (Model es identidad)
    // Orbita alrededor de gTarget según gYaw, gPitch, gDist
}
```
**Propósito**: MVP actual para renderizado.

---

## SECCIÓN 11: PICKING (líneas 709-820)

```cpp
static bool pickNearestPointID(const Mat4& MVP, unsigned int& outID) {
    // Renderiza a FBO especial con IDs en RGB
    // Lee píxeles alrededor del ratón
    // Busca el ID más cercano al centro
    // Retorna el ID de la partícula más cercana
    
    glBindFramebuffer(GL_FRAMEBUFFER, pickFBO);
    glViewport(0, 0, winW, winH);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glUseProgram(pickProg);
    glUniformMatrix4fv(uPickMVPLoc, 1, GL_FALSE, MVP.m);
    glUniform1f(uPickPointSizeLoc, 8.0f);
    
    glBindVertexArray(vao[gPresent]);  // <<<<< AQUÍ EL ERROR
    glDrawArrays(GL_POINTS, 0, Nvis);
    
    // ReadPixels y búsqueda...
}
```
**Propósito**: Sistema de picking renderizando con IDs.

```cpp
static void updateHoverByNearestPoint(const Mat4& MVP) {
    // Llama a pickNearestPointID()
    // Obtiene columna/fila del ID
    // Lee el valor dB desde d_hist_db asincronamente
}

static void updateHoverSample(const Mat4& MVP) {
    // Alternativa: raycasting en el plano Z=0
    // Convierte coordenadas pantalla -> mundo
    // Intersección con plano espectrograma
}
```
**Propósito**: Dos métodos para saber qué punto es hover (picking O raycasting).

---

## SECCIÓN 12: RECURSOS OPENGL (líneas 822-950)

```cpp
static GLuint vao[kBuffers] = { 0,0,0 };  // Vertex Array Objects (3 buffers)
static GLuint vboPos[kBuffers] = { 0,0,0 };  // Position VBO [Nvis * float4]
static GLuint vboCol[kBuffers] = { 0,0,0 };  // Color VBO [Nvis * float4]
static GLuint program = 0;  // Programa shader de partículas

static GLint uMvpLoc = -1;     // Uniform MVP
static GLint uPointSizeLoc = -1;  // Uniform point size

static cudaGraphicsResource* cudaPosRes[kBuffers] = { nullptr,nullptr,nullptr };
static cudaGraphicsResource* cudaColRes[kBuffers] = { nullptr,nullptr,nullptr };
```
**Propósito**: Recursos de partículas 3D con CUDA-OpenGL interop.

```cpp
// 3D Cube (wireframe)
static GLuint cubeVAO = 0;
static GLuint cubeVBO = 0;
static GLuint cubeEBO = 0;
static GLuint cubeProg = 0;
```
**Propósito**: Caja 3D para enmarcar el espacio.

```cpp
// 3D Axis box + gridlines
static GLuint axisVAO = 0;
static GLuint axisVBO = 0;
static GLuint axisProg = 0;

struct AxisVert { float x, y, z; };
static std::vector<AxisVert> gAxisVerts;
static int gAxisEdgeCount = 0;  // # verts para los 12 bordes
static int gAxisGridCount = 0;  // # verts para las gridlines
```
**Propósito**: Ejes 3D + gridlines.

```cpp
// Overlay (2D)
static GLuint overlayProg = 0;
static GLuint overlayVAO = 0;
static GLuint overlayVBO = 0;

struct OverlayVert { float x, y; float r, g, b, a; };
```
**Propósito**: Leyenda 2D + ejes 2D de referencia.

---

## SECCIÓN 13: KERNELS CUDA (líneas 952-1260)

### Kernel: `build_frames_hann_offset()`
```cuda
__global__ void build_frames_hann_offset(
    const float* signal, int nSamples,
    float* frames, const float* window,
    int fftSize, int hop, int nFrames,
    int startSample)
{
    // Cada thread construye 1 sample windowed
    // frames[frame * fftSize + k] = signal[startSample + frame*hop + k] * window[k]
}
```
**Propósito**: Construir B ventanas de datos (overlap-add).

### Kernel: `mag_db_from_r2c_clip()`
```cuda
__global__ void mag_db_from_r2c_clip(
    const cufftComplex* X, float* spec_db,
    int nFrames, int nBinsFull, int Hvis, ...)
{
    // Convierte complejos -> magnitud -> dB
    // spec_db[frame*Hvis + bin] = 20*log10(mag + eps)
}
```
**Propósito**: FFT a espectrograma dB.

### Kernel: `write_hist_cols()`
```cuda
__global__ void write_hist_cols(
    const float* spec_db_block, float* hist_db,
    int B, int H, int Wvis, int head)
{
    // Escribe B columnas en el ring buffer
    // hist_db[col * H + bin] = spec_db_block[f * H + bin]
    // donde col = (head + f) % Wvis
}
```
**Propósito**: Actualizar historial de espectrograma.

### Kernel: `hist_to_vbos_3d()`
```cuda
__global__ void hist_to_vbos_3d(
    const float* hist_db,
    float2* offState, float2* velState,
    float4* outPosVBO, float4* outColVBO,
    int Wvis, int H, int head,
    const float* scale, ...)
{
    // Para cada partícula:
    // 1) X = tiempo normalizado (scroll)
    // 2) Y = frecuencia normalizada
    // 3) Z = amplitud (dB) mapeada
    // 4) Color = colormap(Z)
    // 5) Movimiento en XY con física de spring
}
```
**Propósito**: Llenar VBOs 3D de posiciones + colores.

---

## SECCIÓN 14: INICIALIZACIÓN (líneas 1262-1600)

```cpp
static void initPipeline() {
    // 1) Set GPU device 0
    // 2) Crear stream y eventos CUDA
    // 3) Allocate pinned memory para escala y hover
    // 4) Cargar CSV
    // 5) Calcular parámetros STFT
    // 6) Allocate GPU memory
    // 7) Crear plan cuFFT batched
}

static void initGLandInterop() {
    // 1) Init GLEW
    // 2) Crear programa shader de partículas
    // 3) Crear VAO/VBO para 3 buffers
    // 4) Registrar VBOs con CUDA
    // 5) Crear ejes, cubo, overlay, picking
}

static void primeFirstFrame() {
    // Procesar primer frame para tener datos iniciales
}
```
**Propósito**: Inicializar todo el pipeline.

---

## SECCIÓN 15: DISPLAY LOOP (líneas 1750-1900)

```cpp
static void display() {
    // 1) Actualizar FPS
    // 2) Triple buffering: cambiar gPresent si está listo
    // 3) Si no pausado: procesar siguiente buffer
    // 4) MVP = buildMVP()
    // 5) Picking (si hay hover)
    // 6) glClear + render:
    //    - Cubo wireframe (no depth write)
    //    - Axis box + gridlines
    //    - Partículas (points)
    //    - Labels 3D (Hz en Y, dB en Z)
    //    - Overlay 2D
    //    - Tooltip
    //    - HUD texto
    // 7) glutSwapBuffers()
}
```
**Propósito**: Loop principal de renderizado.

---

## RESUMEN DE ARQUITECTURA

```
CSV ¬ Load ¬ Señal Raw
            |
         Filtro (CPU)
            |
         Señal Filt ? GPU d_signal
                       |
            Windowed frames (kernel)
                       |
              cuFFT R2C (batched)
                       |
          Magnitud ? dB (kernel)
                       |
         Write hist_db (kernel)
                       |
      hist_to_vbos_3d (kernel)
                       |
         VBOs (pos + col)
                       |
            Renderizado 3D
```

Cada "tick" procesa **B=4 frames** en paralelo (triple buffering).
