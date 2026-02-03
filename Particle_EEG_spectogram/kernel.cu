// ===================================================================================================
// kernel.cu - CUDA/cuFFT + OpenGL (VAO+Shaders) interop (3D PRO)
// ===================================================================================================
// 3D grid of particles: X=time, Y=freq (0..45Hz), Z=amplitude (dB mapped) with colormap
// Keeps your 2D overlay (axis+legend) AND adds:
//  - 3D bounding box (X time, Y Hz, Z amplitude) attached to the world
//  - smooth gridlines: every 5 Hz and every N columns in time
//
// ARQUITECTURA:
// CSV → Cargar → Filtro (HP/LP biquad) → STFT en GPU (cuFFT) → Espectrograma dB → Ring Buffer → 
// → Render 3D particles + Overlay 2D (axis, leyenda, tooltip)
// 
// // PSD density (uV^2/Hz) with unnormalized FFT (cuFFT):
// psd = |X|^2 / (fs * sum(w^2)), one-sided correction applied.

//
// Controls (keyboard):
//  P pausa | T HUD texto ON/OFF (pero overlay eje+leyenda SIEMPRE visibles)
//  A autogain | C cmap | G/H gamma | J/K motion
//  [ ] hop | F fft | 1..9 canal
//  X filtro ON/OFF | V RAW/FILTERED | Y HP ON/OFF
//  - / + : velocidad (gPlaySpeed)
//
// Camera (mouse):
//  Left drag : orbit (yaw/pitch)
//  Right drag: pan
//  Wheel     : zoom
// ===================================================================================================

// ===== Windows macro safety =====
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#define GLEW_NO_GLU
#include <GL/glew.h>
#include <GL/freeglut.h>



#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

// Filesystem: prefer <filesystem>, fallback to experimental for older toolchains
#if defined(__has_include)
#  if __has_include(<filesystem>)
#    include <filesystem>
#    namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
#    namespace fs = std::experimental::filesystem;
#  else
#    error "No <filesystem> or <experimental/filesystem> available"
#  endif
#else
#  include <filesystem>
#  namespace fs = std::filesystem;
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " -> " \
              << cudaGetErrorString(e) << std::endl; \
    std::exit(1); \
  } \
} while(0)

#define CUFFT_CHECK(call) do { \
  cufftResult r = (call); \
  if (r != CUFFT_SUCCESS) { \
    std::cerr << "cuFFT error " << __FILE__ << ":" << __LINE__ << " -> " \
              << (int)r << std::endl; \
    std::exit(1); \
  } \
} while(0)

// ================= PERF: stream + TRIPLE buffer =================
// Sistema de triple buffering para CPU-GPU: mientras se renderiza un buffer, otro se procesa en GPU
// Esto permite máxima eficiencia sin bloqueos.
static constexpr int kBuffers = 3;              // 3 buffers: 1 mostrando, 1 en GPU, 1 libre
static cudaStream_t gStream = nullptr;          // Stream CUDA no-bloqueante para operaciones asincrónicas
static cudaEvent_t  gReadyEvent[kBuffers] = { nullptr, nullptr, nullptr }; // Marcadores de finalización
static bool         gHasData[kBuffers] = { false, false, false }; // ¿Tiene datos válidos cada buffer?

static int gPresent = 0;            // Índice del buffer visible actualmente
static int gQueued = 0;             // Índice del buffer en cola para mostrar
static int gHeadForBuffer[kBuffers] = { 0,0,0 }; // Head del ring buffer (posición de escritura) para cada buffer

// ---- escala dB actual (para leyenda) ----
// Estos valores se sincronizan asincronamente desde GPU para evitar stalls
static float* h_scalePinned = nullptr;  // Memoria pinned: 2 floats (min/max dB) - transferencia GPU rápida
static float  gScaleHost[2] = { -20.0f, 40.0f }; // Caché CPU de la escala dB (mostrada en overlay)
static cudaEvent_t gScaleEvent = nullptr; // Marcador de sincronización para escala



// ===== métricas filtro (HUD) =====
// Estas métricas se muestran en el HUD para feedback del filtro aplicado
static float gRmsRaw = 0.0f;    // RMS de la señal original (sin filtrar)
static float gRmsFilt = 0.0f;   // RMS de la señal después del filtro
static float gRmsRemoved = 0.0f; // RMS del ruido removido por el filtro
static float gRemovedPct = 0.0f; // Porcentaje de energía removida

// Playback speed (sample advance fraction)
// Controla la velocidad de reproducción: cuántos samples avanzan por frame
static float gPlaySpeed = 2.00f; // 1.0=reproduc normal, 0.25=4x más lento


// ================= CLOCK =================
static auto   gLastTick = std::chrono::high_resolution_clock::now();
static double gAccumSamples = 0.0; // samples “pendientes” de reproducir
static inline void resetPlaybackClock()
{
    gLastTick = std::chrono::high_resolution_clock::now();
    gAccumSamples = 0.0;
}



// ===== Picking FBO (nearest point) =====
// Sistema para detectar qué partícula está bajo el ratón (hover detection)
// Renderiza a una textura especial donde cada píxel contiene el ID de la partícula
static GLuint pickFBO = 0;         // Framebuffer Object para picking
static GLuint pickTex = 0;         // Textura RGBA8 donde guardamos los IDs (RGB=ID, A=cobertura)
static GLuint pickDepth = 0;       // Renderbuffer de profundidad para el FBO
static GLuint pickProg = 0;        // Shader programa para renderizar IDs
static GLint  uPickMVPLoc = -1;    // Location del uniform MVP en shader de picking
static GLint  uPickPointSizeLoc = -1; // Location del uniform point size
static std::vector<int> gColSample0; // Para cada columna visible, guarda el índice de sample inicial

// Tamaño del parche a leer alrededor del ratón
// Se lee un cuadrado de (2R+1)x(2R+1) pixels alrededor del cursor
static int gPickRadius = 6; // Radio=6 => cuadrado de 13x13 pixels

// Hover: datos del punto bajo el ratón
// Se actualizan continuamente por picking o raycasting
static float* h_hoverPinned = nullptr;    // Pinned memory: 1 float para valor dB del punto hover
static cudaEvent_t gHoverEvent = nullptr; // Marcador de sync para lectura dB desde GPU
static bool        gHoverPending = false; // ¿Esperando lectura de dB desde GPU?
static bool        gHoverValid = false;   // ¿El hover actual es válido/dentro de bounds?
static float       gHoverHz = 0.0f;       // Frecuencia (Hz) del punto bajo el ratón
static float       gHoverDb = 0.0f;       // Amplitud (dB) del punto bajo el ratón
static int         gHoverCol = -1, gHoverRow = -1; // Índices columna/fila del punto hover
static int         gMouseX = 0, gMouseY = 0;       // Posición actual del ratón (pixels)
static float gHoverTimeSec = 0.0f;  // Tiempo en segundos (desde inicio del CSV)
static int   gHoverDispCol = -1;    // Columna visible (0..Wvis-1, izquierda a derecha)



// ================= Settings =================
// Struct centralizado con todas las opciones del programa
// Se accede globalmente como 'g'
struct Settings {
    // --- Control general ---
    bool paused = false;           // ¿Aplicación pausada?
    bool showHUDText = true;       // ¿Mostrar HUD (FPS, parámetros)? El overlay 2D siempre visible

    // --- Escala de amplitud (dB) ---
    bool  autoGain = false;         // ¿Escala automática basada en min/max dinámicos?
    float dbMin = -110.0f;          // Escala dB mínima (si autoGain=false)
    float dbMax = 20.0f;           // Escala dB máxima (si autoGain=false)

    // --- Visualización ---
    float gamma = 1.0f;            // Corrección gamma (1.0=lineal, <1=más oscuro, >1=más claro)
    int   cmap = 0;                // Mapa de colores: 0=heat, 1=gray

    // --- Movimiento de partículas (efecto visual) ---
    float jitter = 0.06f;          // Amplitud del movimiento vibratorio de partículas
    int channel = 1;               // Canal del CSV a visualizar (0..N-1)

    // ===== Filtro previo a FFT =====
    bool  preFilter = true;        // ¿Aplicar filtro a la señal antes del FFT?
    bool  showFiltered = true;     // ¿Mostrar filtrada? (true) o raw? (false)
    bool  removeMean = true;       // ¿Remover componente DC (media)?
    float fs = 256.0f;            // Frecuencia de muestreo en Hz (AJUSTA A TU CSV)
    float lowHz = 45.0f;          // Cutoff del filtro paso-bajo (Hz)
    bool  useHighpass = true;      // ¿Aplicar filtro paso-alto?
    float highHz = 0.5f;           // Cutoff del filtro paso-alto (Hz)

    // ===== 3D look =====
    float zScale = 0.25f;           // Escala de amplitud en eje Z ([-zScale..+zScale])
    int   timeGridEveryCols = 32;  // Gridlines verticales cada N columnas
    bool  show3DGrid = true;       // ¿Mostrar grid 3D?
};
static Settings g; // Instancia global - acceso como g.paused, g.channel, etc

// ------------------------ Repo path / CLI helpers ------------------------
static fs::path gCsvPath = fs::path("datasets_csv") / "sample.csv"; // default (resolved later)

static fs::path getExeDir() {
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len == 0) return fs::current_path();
    fs::path p(std::string(buf, (len > 0 ? len : 0)));
    return p.parent_path();
}

static fs::path findRepoRoot() {
    fs::path p = getExeDir();
    for (int i = 0; i < 8; ++i) {
        if (fs::exists(p / "datasets_csv")) return p;
        if (!p.has_parent_path()) break;
        p = p.parent_path();
    }
    // fallback: current path
    return fs::current_path();
}

static fs::path resolveRepoPath(const fs::path& in) {
    if (in.is_absolute()) return in.lexically_normal();
    fs::path root = findRepoRoot();
    return (root / in).lexically_normal();
}

static void parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) {
            gCsvPath = fs::path(argv[++i]);
        }
        else if (a == "--fs" && i + 1 < argc) {
            try { g.fs = std::stof(argv[++i]); }
            catch (...) { /* ignore invalid */ }
        }
    }
    gCsvPath = resolveRepoPath(gCsvPath);
}

// ================= CSV multi-canal =================
static inline void trim_inplace(std::string& tok) {
    tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(),
        [](unsigned char ch) { return !std::isspace(ch); }));
    tok.erase(std::find_if(tok.rbegin(), tok.rend(),
        [](unsigned char ch) { return !std::isspace(ch); }).base(), tok.end());
}
static inline bool looks_like_number(const std::string& s) {
    for (char c : s) if ((c >= '0' && c <= '9') || c == '-' || c == '.' || c == '+') return true;
    return false;
}
static inline char detect_delim(const std::string& line) {
    size_t c1 = std::count(line.begin(), line.end(), ',');
    size_t c2 = std::count(line.begin(), line.end(), ';');
    return (c2 > c1) ? ';' : ',';
}
static bool load_csv_channels(const std::string& path, std::vector<std::vector<float>>& channels_out) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "No se pudo abrir: " << path << "\n"; return false; }

    std::string line;
    bool delim_set = false; char delim = ',';
    channels_out.clear();
    int nCols = -1;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (!delim_set) { delim = detect_delim(line); delim_set = true; }

        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, delim)) { trim_inplace(tok); cols.push_back(tok); }
        if (cols.empty()) continue;

        if (!looks_like_number(cols[0])) continue; // header

        if (nCols < 0) {
            nCols = (int)cols.size();
            channels_out.resize(nCols);
            for (auto& c : channels_out) c.reserve(1 << 20);
        }
        if ((int)cols.size() < nCols) continue;

        bool ok = true;
        std::vector<float> v(nCols);
        for (int i = 0; i < nCols; ++i) {
            if (!looks_like_number(cols[i])) { ok = false; break; }
            try { v[i] = std::stof(cols[i]); }
            catch (...) { ok = false; break; }
        }
        if (!ok) continue;

        for (int i = 0; i < nCols; ++i) channels_out[i].push_back(v[i]);
    }

    size_t best = 0;
    for (auto& c : channels_out) best = (std::max)(best, c.size());
    return best > 0;
}

// ================= filtro biquad (CPU) =================
struct Biquad {
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float z1 = 0, z2 = 0;
    inline float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

static Biquad make_lowpass(float fs, float fc, float Q = 0.70710678f) {
    Biquad b;
    fc = (std::max)(0.001f, (std::min)(fc, 0.49f * fs));
    float w0 = 2.0f * (float)M_PI * (fc / fs);
    float cs = std::cos(w0);
    float sn = std::sin(w0);
    float alpha = sn / (2.0f * Q);

    float bb0 = (1.0f - cs) * 0.5f;
    float bb1 = (1.0f - cs);
    float bb2 = (1.0f - cs) * 0.5f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cs;
    float a2 = 1.0f - alpha;

    b.b0 = bb0 / a0; b.b1 = bb1 / a0; b.b2 = bb2 / a0;
    b.a1 = a1 / a0;  b.a2 = a2 / a0;
    b.z1 = b.z2 = 0.0f;
    return b;
}

static Biquad make_highpass(float fs, float fc, float Q = 0.70710678f) {
    Biquad b;
    fc = (std::max)(0.001f, (std::min)(fc, 0.49f * fs));
    float w0 = 2.0f * (float)M_PI * (fc / fs);
    float cs = std::cos(w0);
    float sn = std::sin(w0);
    float alpha = sn / (2.0f * Q);

    float bb0 = (1.0f + cs) * 0.5f;
    float bb1 = -(1.0f + cs);
    float bb2 = (1.0f + cs) * 0.5f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cs;
    float a2 = 1.0f - alpha;

    b.b0 = bb0 / a0; b.b1 = bb1 / a0; b.b2 = bb2 / a0;
    b.a1 = a1 / a0;  b.a2 = a2 / a0;
    b.z1 = b.z2 = 0.0f;
    return b;
}

static void preprocess_filter_inplace(std::vector<float>& x, float fs, bool removeMean,
    bool hpOn, float hpHz, float lpHz)
{
    if (x.empty() || fs <= 1.0f) return;

    if (removeMean) {
        double m = 0.0;
        for (float v : x) m += (double)v;
        m /= (double)x.size();
        for (float& v : x) v = (float)(v - m);
    }

    Biquad hp, lp1, lp2;
    if (hpOn) hp = make_highpass(fs, hpHz, 0.70710678f);
    lp1 = make_lowpass(fs, lpHz, 0.70710678f);
    lp2 = make_lowpass(fs, lpHz, 0.70710678f);

    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        if (hpOn) v = hp.process(v);
        v = lp1.process(v);
        v = lp2.process(v);
        x[i] = v;
    }
}

// ================= STFT =================
// Parámetros de la Transformada de Fourier de Corta Duración (Short-Time Fourier Transform)
// Convierte señal temporal en representación tiempo-frecuencia
static float gWinSumSq = 1.0f; // sum(w^2) para PSD
static int   fftSize = 1024;       // Tamaño de la FFT (samples por ventana)
static int   hop = 4;             // Desplazamiento entre ventanas (samples) - menor = más solapamiento
static int   nBinsFull = 0;        // Bins totales FFT (fftSize/2 + 1) para R2C
static int   maxBinVis = 0;        // Max bin a visualizar (limitado a ~45 Hz)
static float binHz = 0.0f;         // Ancho de cada bin en Hz (fs / fftSize)
static float fTopShownHz = 45.0f;  // Frecuencia máxima mostrada en pantalla (Hz)

// --- Parámetros de visualización ---
static int Wvis = 256;             // Ancho del espectrograma visible (# columnas en pantalla)
static int H = 0;                  // Alto del espectrograma (# bins de frecuencia a mostrar)
static int Nvis = 0;               // Total de partículas visibles = Wvis * H

// --- Procesamiento batch ---
static const int stepFrames = 4;   // Procesar cuántos frames por tick
static int B = stepFrames;         // B = número de frames por buffer (normalmente = stepFrames)

// --- Control de posición en la señal ---
static int startSample = 0;        // Índice del primer sample a procesar
static int maxStartSample = 0;     // Máximo valid startSample (para no salirse del final)
static int nSamples = 0;           // Total de samples en la señal cargada
static int gHead = 0;              // Posición de escritura del ring buffer (0..Wvis-1)

// --- Datos de señal (CPU) ---
static std::vector<std::vector<float>> h_channels; // Todos los canales del CSV cargados
static std::vector<float> h_signal_raw;   // Canal seleccionado sin filtro
static std::vector<float> h_signal_filt;  // Canal seleccionado filtrado
static std::vector<float> h_signal;       // Puntero a raw o filt según g.showFiltered

// --- GPU memory (buffers principales) ---
static float* d_signal = nullptr;  // Señal en GPU (nSamples)
static float* d_win = nullptr;     // Ventana Hann en GPU (fftSize)

// --- Triple buffer para STFT (3 buffers en paralelo) ---
static float* d_frames[kBuffers] = { nullptr,nullptr,nullptr };    // Frames windowed [B, fftSize]
static cufftComplex* d_fft[kBuffers] = { nullptr,nullptr,nullptr };  // Resultado FFT [B, nBinsFull]
static float* d_spec_db[kBuffers] = { nullptr,nullptr,nullptr };   // Espectrograma dB [B, H]

// --- Procesamiento FFT batch ---
static cufftHandle plan = 0;       // Plan de cuFFT batched (reutilizado cada frame)
static float* d_hist_db = nullptr; // Historial total [Wvis, H] - ring buffer con todos los frames



// --- Normalización ---
static float* d_minmax[kBuffers] = { nullptr,nullptr,nullptr }; // Min/max locales por buffer
static float* d_scale = nullptr;   // Escala global [min, max] sincronizada desde GPU
static float gInvWinL2 = 1.0f;     // 1 / L2-norm de ventana Hann (normalización COLA)

// ================= FPS / HUD =================
// Metrics para mostrar en el HUD (performance y debug info)
static double fps = 0.0;            // Frames por segundo calculado
static int fpsFrames = 0;           // Contador de frames para el cálculo de FPS
static auto fpsT0 = std::chrono::high_resolution_clock::now(); // Timestamp anterior para FPS






// ================= Simple math (Vec3/Mat4) =================
// Matemática 3D minimalista para transformaciones de cámara y proyección
// (se evita GLM para mantener dependencias mínimas)

// --- Vector 3D ---
struct Vec3 {
    float x, y, z;
    Vec3() :x(0), y(0), z(0) {}  // Constructor default (origen)
    Vec3(float X, float Y, float Z) :x(X), y(Y), z(Z) {} // Constructor explícito
};

// Operadores Vec3
static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
static inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; } // Producto escalar
static inline Vec3 cross(const Vec3& a, const Vec3& b) { // Producto cruz
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float vlen(const Vec3& v) { return std::sqrt(dot(v, v)); } // Magnitud/longitud
static inline Vec3 normalize(const Vec3& v) { // Normalización a vector unitario
    float L = vlen(v); if (L < 1e-8f) return Vec3(0, 0, 0); return v * (1.0f / L);
}

// --- Matriz 4x4 (OpenGL column-major) ---
struct Mat4 {
    // Column-major: m[col*4 + row] accede a elemento (row, col)
    // Así es como OpenGL lo espera para shaders
    float m[16];
};

// Operaciones Mat4
static Mat4 mat4_identity() {
    // Crea matriz identidad 4x4
    Mat4 r{};
    r.m[0] = 1; r.m[5] = 1; r.m[10] = 1; r.m[15] = 1;
    return r;
}

static Mat4 mat4_mul(const Mat4& A, const Mat4& B) {
    // Multiplicación matricial A * B (column-major)
    Mat4 R{};
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            R.m[c * 4 + r] =
                A.m[0 * 4 + r] * B.m[c * 4 + 0] +
                A.m[1 * 4 + r] * B.m[c * 4 + 1] +
                A.m[2 * 4 + r] * B.m[c * 4 + 2] +
                A.m[3 * 4 + r] * B.m[c * 4 + 3];
        }
    }
    return R;
}

static Mat4 mat4_perspective(float fovyRad, float aspect, float zNear, float zFar) {
    // Matriz de proyección perspectiva (OpenGL style)
    Mat4 r{};
    float f = 1.0f / std::tan(fovyRad * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zFar + zNear) / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zFar * zNear) / (zNear - zFar);
    return r;
}

static Mat4 mat4_lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    // Matriz de vista (cámara apuntando de 'eye' a 'center')
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);

    Mat4 r = mat4_identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;

    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

// ================= Camera (orbit) =================
// Sistema de cámara orbital esférica: el usuario orbita alrededor de un punto central
static int winW = 1100, winH = 700;  // Dimensiones de la ventana

// --- Parámetros de cámara orbital ---
static float gYaw = 0.7f;     // Ángulo horizontal (radianes) - izquierda/derecha
static float gPitch = 0.35f;  // Ángulo vertical (radianes) - arriba/abajo
static float gDist = 3.0f;    // Distancia a gTarget
static Vec3  gTarget = Vec3(0.0f, 0.0f, 0.0f); // Centro de orbita

// --- Control del ratón ---
static bool gMouseL = false;  // ¿Botón izquierdo presionado?
static bool gMouseR = false;  // ¿Botón derecho presionado?
static int  gLastMX = 0;      // Última posición X del ratón
static int  gLastMY = 0;      // Última posición Y del ratón

// Construye matriz MVP (Model-View-Projection) basada en parámetros de cámara
static Mat4 buildMVP() {
    float aspect = (winH > 0) ? (float)winW / (float)winH : 1.0f;
    Mat4 P = mat4_perspective(55.0f * (float)M_PI / 180.0f, aspect, 0.05f, 50.0f);

    // Conversión de coordenadas esféricas (gYaw, gPitch, gDist) a cartesianas
    float cp = std::cos(gPitch), sp = std::sin(gPitch);
    float cy = std::cos(gYaw), sy = std::sin(gYaw);

    Vec3 eye = gTarget + Vec3(
        gDist * cp * sy,
        gDist * sp,
        gDist * cp * cy
    );

    Mat4 V = mat4_lookAt(eye, gTarget, Vec3(0, 1, 0));
    return mat4_mul(P, V);
}
struct Vec4 { float x, y, z, w; }; // Vector homogéneo 4D (para transformaciones 3D)

// Multiplica matriz 4x4 por vector 4D (column-major)
static Vec4 mul(const Mat4& M, const Vec4& v)
{
    Vec4 r;
    r.x = M.m[0] * v.x + M.m[4] * v.y + M.m[8] * v.z + M.m[12] * v.w;
    r.y = M.m[1] * v.x + M.m[5] * v.y + M.m[9] * v.z + M.m[13] * v.w;
    r.z = M.m[2] * v.x + M.m[6] * v.y + M.m[10] * v.z + M.m[14] * v.w;
    r.w = M.m[3] * v.x + M.m[7] * v.y + M.m[11] * v.z + M.m[15] * v.w;
    return r;
}

// Proyecta un punto 3D a coordenadas 2D de pantalla usando MVP
// Retorna true si el punto está visible, false si está fuera o detrás de cámara
static bool projectToScreen(const Vec3& p, const Mat4& MVP, int W, int H, float& sx, float& sy)
{
    Vec4 clip = mul(MVP, Vec4{ p.x, p.y, p.z, 1.0f }); // Transforma a clip space
    if (clip.w <= 1e-6f) return false;           // Detrás de cámara

    // Divide por w para obtener NDC (Normalized Device Coordinates, -1..1)
    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;

    // Clipping suave: considera fuera si está muy lejos en NDC
    if (ndcX < -1.2f || ndcX > 1.2f || ndcY < -1.2f || ndcY > 1.2f) return false;

    // Convierte NDC a coordenadas de pantalla (pixels)
    sx = (ndcX * 0.5f + 0.5f) * (float)W;
    sy = (ndcY * 0.5f + 0.5f) * (float)H;
    return true;
}



static void passiveMotion(int x, int y) { gMouseX = x; gMouseY = y; }

// Forward declaration of VAO array (defined later in OpenGL main resources)
static GLuint vao[kBuffers] = { 0,0,0 };




static bool pickNearestPointID(const Mat4& MVP, unsigned int& outID /*0..Nvis-1*/)
{
    if (!pickFBO || !pickProg) return false;
    if (winW <= 1 || winH <= 1) return false;

    // Si el ratón está fuera, NO intentes leer píxeles
    if (gMouseX < 0 || gMouseX >= winW || gMouseY < 0 || gMouseY >= winH)
        return false;

    // Render pass a FBO
    glBindFramebuffer(GL_FRAMEBUFFER, pickFBO);
    glViewport(0, 0, winW, winH);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(pickProg);
    glUniformMatrix4fv(uPickMVPLoc, 1, GL_FALSE, MVP.m);
    glUniform1f(uPickPointSizeLoc, 8.0f); // más grande para facilitar picking

    if (gPresent >= 0 && gPresent < kBuffers) {
        glBindVertexArray(vao[gPresent]);
        glDrawArrays(GL_POINTS, 0, Nvis);
    }


    // ReadPixels de un parche alrededor del ratón
    int cx = gMouseX;
    int cy = winH - gMouseY - 1; // OpenGL bottom-left

    int R = gPickRadius;
    int x0 = (std::max)(0, cx - R);
    int y0 = (std::max)(0, cy - R);
    int x1 = (std::min)(winW - 1, cx + R);
    int y1 = (std::min)(winH - 1, cy + R);

    // GUARD crítico (evita rw/rh negativos)
    if (x1 < x0 || y1 < y0) return false;

    int rw = x1 - x0 + 1;
    int rh = y1 - y0 + 1;

    std::vector<unsigned char> pix((size_t)rw * (size_t)rh * 4);
    glReadPixels(x0, y0, rw, rh, GL_RGBA, GL_UNSIGNED_BYTE, pix.data());

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Buscar pixel no-fondo más cercano al centro
    int bestDx = 0, bestDy = 0;
    int bestD2 = 1e9;
    unsigned int bestID = 0;
    bool found = false;

    for (int j = 0; j < rh; ++j) {
        for (int i = 0; i < rw; ++i) {
            const unsigned char* p = &pix[(j * rw + i) * 4];
            unsigned char a = p[3];
            if (a == 0) continue; // fondo

            unsigned int enc = (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16);
            if (enc == 0) continue; // fondo por seguridad

            unsigned int id = enc - 1u; // deshacer +1
            if (id >= (unsigned int)Nvis) continue;

            int px = x0 + i;
            int py = y0 + j;
            int dx = px - cx;
            int dy = py - cy;
            int d2 = dx * dx + dy * dy;
            if (d2 < bestD2) {
                bestD2 = d2;
                bestID = id;
                bestDx = dx;
                bestDy = dy;
                found = true;
            }
        }
    }

    if (!found) return false;
    outID = bestID;
    return true;
}



static inline float  clamp01f(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }



// ================= Speed slider (UI) =================
static double gSpeedMin = 0.05;   // 20x más lento
static double gSpeedMax = 100.0;  // 200x más rápido (ajústalo)
static float  gSpeedT = 0.0f;   // slider 0..1
static bool   gDragSpeed = false; // dragging knob  

// Rect del slider en pixels (bottom-left coords)
static int gSpeedX = 80;
static int gSpeedY = 25;
static int gSpeedW = 320;
static int gSpeedH = 14;

static inline double clampd(double x, double a, double b) { return (x < a) ? a : (x > b) ? b : x; }
static inline double lerpd(double a, double b, double t) { return a + (b - a) * t; }

static double speedFromT(double t) {
    t = clampd(t, 0.0, 1.0);
    double lo = std::log10(gSpeedMin);
    double hi = std::log10(gSpeedMax);
    return std::pow(10.0, lerpd(lo, hi, t));
}

static double tFromSpeed(double s) {
    s = clampd(s, gSpeedMin, gSpeedMax);
    double lo = std::log10(gSpeedMin);
    double hi = std::log10(gSpeedMax);
    return (std::log10(s) - lo) / (hi - lo);
}

// ===== Modo B: sliders de tiempo =====
static bool   gDragZoom = false;
static bool   gDragHop = false;

// Zoom temporal (segundos visibles)
static double gWinSecTarget = 4.0;   // valor real aplicado
static double gWinSecPending = 4.0;   // preview mientras arrastras
static float  gWinSecT = 0.0f;   // knob 0..1

// Hop (resolución temporal)
static int    gHopPending = 4;
static float  gHopT = 0.5f;

// Rangos recomendados
static constexpr double kWinSecMin = 2.0;
static constexpr double kWinSecMax = 240.0;

static constexpr int kHopMin = 4;
static constexpr int kHopMax = 512;
static constexpr int kHopStep = 4;

// Clamp Wvis para que no explote VRAM/CPU
static constexpr int kWvisMin = 64;
static constexpr int kWvisMax = 4096;

// Geometría (pon donde te guste)
static int gZoomX = 500, gZoomY = 25, gZoomW = 260, gZoomH = 14;
static int gHopX = 800, gHopY = 25, gHopW = 260, gHopH = 14;
// Rects globales (bottom-left) para sliders extra
static int gFftX = 0, gFftY = 0, gFftW = 0, gFftH = 14;
static int gTimeZoomX = 0, gTimeZoomY = 0, gTimeZoomW = 0, gTimeZoomH = 14;


static constexpr int kKnobW = 10;
static constexpr int kSliderPad = 10;

static int snapHop(int v) {
    v = (std::max)(kHopMin, (std::min)(kHopMax, v));
    v = (int)std::round((float)v / (float)kHopStep) * kHopStep;
    v = (std::max)(kHopMin, (std::min)(kHopMax, v));
    return v;
}

static int hopFromT(float t) {
    t = clamp01f(t);
    double a = std::log((double)kHopMin), b = std::log((double)kHopMax);
    int h = (int)std::round(std::exp(a + (double)t * (b - a)));
    return snapHop(h);
}
static double winSecFromT(float t) {
    t = clamp01f(t);
    double a = std::log(kWinSecMin), b = std::log(kWinSecMax);
    return std::exp(a + (double)t * (b - a));
}




// ================= UI V2: calc vs view =================

// FFT (calc): pending + drag
static bool  gDragFFT = false;
static int   gFftPending = 1024;
static float gFftT = 0.0f;

// Opciones discretas de fftN (puedes añadir 4096 si quieres)
static constexpr int kFftOptions[] = { 256, 512, 1024, 2048, 4096 };
static constexpr int kFftOptCount = (int)(sizeof(kFftOptions) / sizeof(kFftOptions[0]));

// Time zoom (view-only)
static bool   gDragTimeZoom = false;
static double gTimeZoom = 1.0;     // 1x = ver todo Wvis
static float  gTimeZoomT = 0.0f;
static constexpr double kTimeZoomMin = 1.0;
static constexpr double kTimeZoomMax = 16.0;

// Viewport derivado del zoom temporal (en columnas)
static int  gViewStart = 0;       // en dispCol (0..Wvis-1)
static int  gVisibleFrames = 0;   // columnas visibles
static bool gViewDirty = true;    // requiere recalcular VBO/axis por cambio visual

// Caja Calc: botón Apply
static bool gPressApply = false;
static int  gApplyX = 0, gApplyY = 0, gApplyW = 90, gApplyH = 22;

// Layout cajas (bottom-left coords)
static int gCalcBoxX = 20, gCalcBoxY = 20, gCalcBoxW = 230, gCalcBoxH = 200;

static int gViewBoxX = 20, gViewBoxY = 250, gViewBoxW = 320, gViewBoxH = 90;

// Convierte mouse (GLUT top-left) a bottom-left y testea el rect del slider
static bool hitSpeedSlider(int mx, int myTopLeft) {
    int my = winH - myTopLeft; // ahora bottom-left
    // ampliamos un poco para poder clicar el knob fácil
    int pad = 8;
    return (mx >= gSpeedX - pad && mx <= gSpeedX + gSpeedW + pad &&
        my >= gSpeedY - pad && my <= gSpeedY + gSpeedH + pad);
}
static inline float sliderTFromMouseX(int mx, int x, int w)
{
    float t = (float)(mx - x - 0.5f * (float)kKnobW) / (float)(w - kKnobW);
    return clamp01f(t);
}
static bool gCalcDirty = false;


// Calc dirty: si pending != aplicado
static void updateCalcDirty()
{
    gCalcDirty =
        (std::abs(gWinSecPending - gWinSecTarget) > 1e-9) ||
        (gHopPending != hop) ||
        (gFftPending != fftSize);
}



static void setZoomFromMouseX(int mx)
{
    float t = sliderTFromMouseX(mx, gZoomX, gZoomW);
    gWinSecT = t;
    gWinSecPending = winSecFromT(t);
    updateCalcDirty();                // <<< AÑADIR
}

static void setHopFromMouseX(int mx)
{
    float t = sliderTFromMouseX(mx, gHopX, gHopW);
    gHopT = t;
    gHopPending = hopFromT(t);
    updateCalcDirty();                // <<< AÑADIR
}

// Cambia speed a partir de mouse X (drag)
static void setSpeedFromMouseX(int mx) {
    int knobW = 10;
    double t = (double)(mx - gSpeedX) / (double)(gSpeedW - knobW);
    t = clampd(t, 0.0, 1.0);
    gSpeedT = (float)t;
    gPlaySpeed = (float)speedFromT(t);

    // IMPORTANTE para tu playback dt-based: evita "catch-up" al cambiar speed
    resetPlaybackClock();
}

// Helper: cuando cambias gPlaySpeed desde teclado, sincronizas slider
static void syncSliderFromSpeed() {
    gSpeedT = (float)tFromSpeed(gPlaySpeed);
}

// ================= UI V3: Transport =================
static int gTransportBoxX = 560, gTransportBoxY = 120, gTransportBoxW = 320, gTransportBoxH = 90;

static bool gLoopPlayback = true;

// Seek slider (scrub)
static int   gSeekX = 0, gSeekY = 0, gSeekW = 0, gSeekH = 14;
static bool  gDragSeek = false;
static float gSeekT = 0.0f;                 // 0..1
static double gSeekSecPending = 0.0;        // texto mientras arrastras
static bool  gSeekDirty = false;
static bool  gSeekWasPaused = true;         // para restaurar play/pause tras drag

// Botones transporte
static int gBtnPlayX = 0, gBtnPlayY = 0, gBtnPlayW = 70, gBtnPlayH = 20;
static int gBtnRestartX = 0, gBtnRestartY = 0, gBtnRestartW = 70, gBtnRestartH = 20;
static int gBtnStepX = 0, gBtnStepY = 0, gBtnStepW = 70, gBtnStepH = 20;
static int gBtnLoopX = 0, gBtnLoopY = 0, gBtnLoopW = 70, gBtnLoopH = 20;




// ================= UI V4: Tone (Gamma + ZScale) =================
static int gToneBoxW = 320, gToneBoxH = 120;
static int gToneBoxX = 0, gToneBoxY = 0;   // se recalculan en layout (top-right)

// Sliders rect (bottom-left coords)
static int gGammaX = 0, gGammaY = 0, gGammaW = 0, gGammaH = 14;
static int gZsX = 0, gZsY = 0, gZsW = 0, gZsH = 14;

// Drag state
enum ToneActiveSlider { TONE_NONE = 0, TONE_GAMMA = 1, TONE_ZSCALE = 2 };
static ToneActiveSlider gToneActive = TONE_NONE;
static bool gToneDragging = false;

// RANGOS (log para control fino)
static constexpr float kGammaMin = 0.25f;
static constexpr float kGammaMax = 3.00f;

static constexpr float kZScaleMin = 0.05f;
static constexpr float kZScaleMax = 1.50f;

// Helper clamp
static inline float clampf(float v, float a, float b) { return (v < a) ? a : (v > b) ? b : v; }

// Map log slider
static float sliderLogFromT(float t, float vmin, float vmax)
{
    t = clamp01f(t);
    float a = std::log(vmin);
    float b = std::log(vmax);
    return std::exp(a + t * (b - a));
}
static float tFromSliderLog(float v, float vmin, float vmax)
{
    v = clampf(v, vmin, vmax);
    float a = std::log(vmin);
    float b = std::log(vmax);
    return (std::log(v) - a) / (b - a);
}

// (Opcional) útil para depurar: saber si UI consume
static bool gUIDragging = false;

static bool gMouseInside = true;

static void cancelAllUIDrags()
{
    gUIDragging = false;

    gDragSpeed = false;
    gDragZoom = false;
    gDragHop = false;
    gDragFFT = false;
    gDragTimeZoom = false;

    gDragSeek = false;

    gToneDragging = false;
    gToneActive = TONE_NONE;

    // opcional: cancela cámara si la usas
    gMouseL = false;
    gMouseR = false;
}

static inline double fsSafeD() { return (double)(std::max)(1e-6f, g.fs); }

static void updateHoverByNearestPoint(const Mat4& MVP)
{

    if (gMouseX < 0 || gMouseX >= winW || gMouseY < 0 || gMouseY >= winH) {
        gHoverValid = false;
        return;
    }
    unsigned int id = 0;
    if (!pickNearestPointID(MVP, id)) {
        gHoverValid = false;
        return;
    }

    int col = (int)(id % (unsigned int)Wvis);
    int row = (int)(id / (unsigned int)Wvis);
    row = (std::max)(0, (std::min)(H - 1, row));

    gHoverHz = row * binHz;

    // head coherente con el buffer que se está mostrando
    int head = gHeadForBuffer[gPresent];

    // columna "visible" (dispCol) coherente con el kernel
    int dispColFull = (col - head + Wvis) % Wvis;
    int dispColView = dispColFull - gViewStart;
    gHoverDispCol = (dispColView >= 0 && dispColView < gVisibleFrames) ? dispColView : -1;

    int s0 = (col >= 0 && col < Wvis) ? gColSample0[col] : -1;
    if (s0 < 0) {
        // antes tenías "dispCol" (no existe). Usa dispColFull:
        s0 = dispColFull * hop;
    }

    gHoverTimeSec = (float)((double)s0 / fsSafeD());


    if (col != gHoverCol || row != gHoverRow) {
        gHoverCol = col;
        gHoverRow = row;

        const float* dPtr = d_hist_db + (col * H + row);
        CUDA_CHECK(cudaMemcpyAsync(h_hoverPinned, dPtr, sizeof(float),
            cudaMemcpyDeviceToHost, gStream));
        CUDA_CHECK(cudaEventRecord(gHoverEvent, gStream));
        gHoverPending = true;
    }

    if (gHoverPending && cudaEventQuery(gHoverEvent) == cudaSuccess) {
        gHoverDb = h_hoverPinned[0];
        gHoverPending = false;
    }

    gHoverValid = true;
}





static float tFromWinSec(double sec) {
    sec = clampd(sec, kWinSecMin, kWinSecMax);
    double a = std::log(kWinSecMin), b = std::log(kWinSecMax);
    return (float)((std::log(sec) - a) / (b - a));
}

static float tFromFft(int n)
{
    int idx = 0;
    for (int i = 0; i < kFftOptCount; ++i) if (kFftOptions[i] == n) idx = i;
    return (kFftOptCount > 1) ? (float)idx / (float)(kFftOptCount - 1) : 0.0f;
}


static float tFromHop(int h) {
    h = snapHop(h);
    double a = std::log((double)kHopMin), b = std::log((double)kHopMax);
    return (float)((std::log((double)h) - a) / (b - a));
}



// GLUT mouse: (0,0) arriba-izq  -> overlay: (0,0) abajo-izq
static inline int mouseY_toBL(int yTopLeft) { return winH - yTopLeft - 1; }

static inline bool hitRectBL(int mx, int myTopLeft, int x, int y, int w, int h, int pad = kSliderPad)
{
    int my = mouseY_toBL(myTopLeft);
    return (mx >= x - pad && mx <= x + w + pad &&
        my >= y - pad && my <= y + h + pad);
}


static bool hitZoomSlider(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gZoomX, gZoomY, gZoomW, gZoomH); }
static bool hitHopSlider(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gHopX, gHopY, gHopW, gHopH); }



static int computeWvis(double winSec, int hop_)
{
    double fs = (double)(std::max)(1e-6f, g.fs);
    int w = (int)std::round(winSec * fs / (double)hop_);
    w = (std::max)(kWvisMin, (std::min)(kWvisMax, w));
    return w;
}

// Recalcula ventana visible SOLO visual (no toca datos)
static void updateTimeView()
{
    double z = clampd(gTimeZoom, kTimeZoomMin, kTimeZoomMax);
    int framesWin = (std::max)(2, Wvis);
    int vis = (int)std::round((double)framesWin / z);
    vis = (std::max)(2, (std::min)(framesWin, vis));
    gVisibleFrames = vis;
    gViewStart = framesWin - vis;
}

static float tFromTimeZoom(double z)
{
    z = clampd(z, kTimeZoomMin, kTimeZoomMax);
    double a = std::log(kTimeZoomMin), b = std::log(kTimeZoomMax);
    return (float)((std::log(z) - a) / (b - a));
}


static void syncTimeSlidersFromState()
{
    // Calc (aplicado -> pending)
    gWinSecPending = gWinSecTarget;
    gWinSecT = tFromWinSec(gWinSecTarget);

    gHopPending = hop;
    gHopT = tFromHop(hop);

    gFftPending = fftSize;
    gFftT = tFromFft(fftSize);

    updateCalcDirty();

    // View defaults coherentes (no resetea al rebuild)
    gTimeZoomT = tFromTimeZoom(gTimeZoom);
    updateTimeView();
    gViewDirty = true;
}


static int fftFromT(float t)
{
    t = clamp01f(t);
    int idx = (int)std::round(t * (float)(kFftOptCount - 1));
    idx = (std::max)(0, (std::min)(kFftOptCount - 1, idx));
    return kFftOptions[idx];
}



static double timeZoomFromT(float t)
{
    t = clamp01f(t);
    double a = std::log(kTimeZoomMin), b = std::log(kTimeZoomMax);
    return std::exp(a + (double)t * (b - a));
}

// Kernel: Llena el ring buffer con un valor (usado en reset)
__global__ void fill_hist(float* hist_db, int total, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) hist_db[i] = v;
}




static inline double spanSecVisible()
{
    if (Wvis <= 1) return 0.0;
    return (double)(Wvis - 1) * (double)hop / fsSafeD();
}


// Tiempo de columna = INICIO de la ventana (no depende de fftSize)
static inline double secFromS0(int s0)
{
    return (double)s0 / fsSafeD();
}

// máximo seek razonable si NO quieres padding al final (tu pipeline actual no paddea al final)
static inline double maxRightSecNoPad()
{
    int s0_last = maxStartSample + (B - 1) * hop; // start sample del último frame
    return secFromS0(s0_last);
}


static inline double durationSec()
{
    return (double)nSamples / fsSafeD();
}

// Tiempo del borde derecho REAL (última columna escrita del ring)
static inline double rightEdgeSec()
{
    if (Wvis <= 0) return 0.0;
    int colLast = (gHead - 1 + Wvis) % Wvis;

    int s0 = (colLast >= 0 && colLast < (int)gColSample0.size()) ? gColSample0[colLast] : -1;
    if (s0 < 0) {
        // fallback aproximado coherente en centro:
        double approxS0 = (double)startSample + (double)(Wvis - 1) * (double)hop;
        return approxS0 / fsSafeD();
    }
     return secFromS0(s0);
} 


// Convierte slider 0..1 -> segundos 0..duración
static inline double seekMinRightSec()
{
    return spanSecVisible();
}

static inline double seekMaxRightSec()
{
    // si prefieres permitir padding al final, aquí puedes devolver durationSec()
    // pero como tu pipeline NO permite startSample > maxStartSample, esto es lo coherente:
    return maxRightSecNoPad();
}

static inline double seekSecFromT(float t)
{
    t = clamp01f(t);
    double a = seekMinRightSec();
    double b = (std::max)(a, seekMaxRightSec());
    return a + (b - a) * (double)t;
}

static inline float tFromSeekSec(double sec)
{
    double a = seekMinRightSec();
    double b = (std::max)(a, seekMaxRightSec());
    sec = clampd(sec, a, b);
    double den = (b - a);
    if (den < 1e-9) return 0.0f;
    return (float)((sec - a) / den);
}

// Bloques necesarios para “llenar” la ventana tras un seek (Wvis columnas)
static inline int blocksToFillWindow()
{
    return (Wvis + B - 1) / B; // B=stepFrames
}

// Máximo startSample que permite hacer prefill sin wrap durante blocksToFillWindow()
static inline int maxStartForPrefill()
{
    int blocksFill = (std::max)(1, blocksToFillWindow());
    int m = maxStartSample - (blocksFill - 1) * (B * hop);
    return (m < 0) ? 0 : m;
}


static void clearHistAndResetScroll()
{
    gHead = 0;
    for (int i = 0; i < kBuffers; ++i) gHeadForBuffer[i] = 0;

    if (d_hist_db) {
        int total = Wvis * H;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        fill_hist << <blocks, threads, 0, gStream >> > (d_hist_db, total, g.dbMin);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(gStream));
        std::fill(gColSample0.begin(), gColSample0.end(), -1);

    }
}






// ================= OpenGL main resources (3D particles) =================
static GLuint vboPos[kBuffers] = { 0,0,0 };   // float4 position
static GLuint vboCol[kBuffers] = { 0,0,0 };   // float4 color
static GLuint program = 0;

static GLint uMvpLoc = -1;
static GLint uPointSizeLoc = -1;



static cudaGraphicsResource* cudaPosRes[kBuffers] = { nullptr,nullptr,nullptr };
static cudaGraphicsResource* cudaColRes[kBuffers] = { nullptr,nullptr,nullptr };









// ================= 3D axis box + gridlines (world-attached) =================
static GLuint axisVAO = 0;
static GLuint axisVBO = 0;
static GLuint axisProg = 0;
static GLint  uAxisMVPLoc = -1;
static GLint  uAxisColorLoc = -1;

struct AxisVert { float x, y, z; };
static std::vector<AxisVert> gAxisVerts;    // full VBO
static int gAxisEdgeCount = 0;             // number of vertices for edges (GL_LINES)
static int gAxisGridCount = 0;             // number of vertices for gridlines (GL_LINES)

// ================= Overlay (2D) =================
static GLuint overlayProg = 0;
static GLuint overlayVAO = 0;
static GLuint overlayVBO = 0;
static GLint  uViewportLoc = -1;

struct OverlayVert { float x, y; float r, g, b, a; };

static inline float clamp01(float x) { return (std::max)(0.0f, (std::min)(1.0f, x)); }
static void cmap_heat_cpu(float t, float& r, float& gg, float& b) {
    t = clamp01(t);
    if (t < 0.33f) { float u = t / 0.33f; r = 0.0f; gg = u; b = 1.0f; }
    else if (t < 0.66f) { float u = (t - 0.33f) / 0.33f; r = u; gg = 1.0f; b = 1.0f - u; }
    else { float u = (t - 0.66f) / 0.34f; r = 1.0f; gg = 1.0f - u; b = 0.0f; }
}
static void cmap_gray_cpu(float t, float& r, float& gg, float& b) { t = clamp01(t); r = gg = b = t; }

static void overlayAddTri(std::vector<OverlayVert>& v, float x0, float y0, float x1, float y1, float r, float g_, float b, float a) {
    v.push_back({ x0,y0,r,g_,b,a }); v.push_back({ x1,y0,r,g_,b,a }); v.push_back({ x1,y1,r,g_,b,a });
    v.push_back({ x0,y0,r,g_,b,a }); v.push_back({ x1,y1,r,g_,b,a }); v.push_back({ x0,y1,r,g_,b,a });
}
static void overlayAddLine(std::vector<OverlayVert>& v, float x0, float y0, float x1, float y1, float r, float g_, float b, float a) {
    v.push_back({ x0,y0,r,g_,b,a }); v.push_back({ x1,y1,r,g_,b,a });
}

// ================= CUDA helpers (min/max float) =================
__device__ inline float atomicMinFloat(float* addr, float value) {
    int* a = (int*)addr; int old = *a, assumed;
    while (true) {
        assumed = old;
        if (__int_as_float(assumed) <= value) break;
        old = atomicCAS(a, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}
__device__ inline float atomicMaxFloat(float* addr, float value) {
    int* a = (int*)addr; int old = *a, assumed;
    while (true) {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(a, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}

__global__ void init_minmax(float* mm) { mm[0] = 1e30f; mm[1] = -1e30f; }

__global__ void reduce_minmax(const float* spec, int total, float* mm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = spec[i];
    atomicMinFloat(&mm[0], v);
    atomicMaxFloat(&mm[1], v);
}

__global__ void smooth_scale(const float* mm, float* scale, float alpha) {
    float mn = mm[0], mx = mm[1];
    float a = fminf(1.0f, fmaxf(0.0f, alpha));
    scale[0] = scale[0] + a * (mn - scale[0]);
    scale[1] = scale[1] + a * (mx - scale[1]);
    if (scale[1] - scale[0] < 5.0f) {
        float mid = 0.5f * (scale[0] + scale[1]);
        scale[0] = mid - 2.5f;
        scale[1] = mid + 2.5f;
    }
}

// ================= Colormaps (GPU) =================
__device__ inline float3 cmap_heat(float t) {
    t = fminf(1.0f, fmaxf(0.0f, t));
    if (t < 0.33f) { float u = t / 0.33f; return make_float3(0.0f, u, 1.0f); }
    else if (t < 0.66f) { float u = (t - 0.33f) / 0.33f; return make_float3(u, 1.0f, 1.0f - u); }
    else { float u = (t - 0.66f) / 0.34f; return make_float3(1.0f, 1.0f - u, 0.0f); }
}
__device__ inline float3 cmap_gray(float t) { t = fminf(1.0f, fmaxf(0.0f, t)); return make_float3(t, t, t); }
__device__ inline float3 apply_cmap(int cmap, float t) { return (cmap == 0) ? cmap_heat(t) : cmap_gray(t); }

// ================= CUDA kernels =================
// Kernels que se ejecutan en GPU para procesamiento de STFT y rendering de partículas

// Kernel: Construye ventanas (frames) con Hann windowing
// Entrada: signal (temporal), window (Hann)
// Salida: frames windowed listos para FFT
__global__ void build_frames_hann_offset(
    const float* signal,        // Señal de entrada
    int nSamples,               // Longitud de la señal
    float* frames,              // Salida: frames windowed [nFrames, fftSize]
    const float* window,        // Ventana Hann pre-computada
    int fftSize,                // Tamaño de cada frame
    int hop,                    // Desplazamiento entre frames
    int nFrames,                // Número de frames a procesar
    int startSample)            // Índice de inicio en la señal
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nFrames * fftSize;
    if (idx >= total) return;

    // Descompose linear index en (frame, sample dentro del frame)
    int frame = idx / fftSize;
    int k = idx - frame * fftSize;

    // Calcula índice en la señal original
    int s = startSample + frame * hop + k;

    // Carga sample o 0 si está fuera de rango
    float x = (s >= 0 && s < nSamples) ? signal[s] : 0.0f;

    // Aplica ventana Hann
    frames[idx] = x * window[k];
}

// Kernel: Convierte magnitud compleja a dB (Decibeles)
// Entrada: FFT output (complejos)
// Salida: espectrograma en dB
__global__ void psd_db_from_r2c_onesided(
    const cufftComplex* X,  // [nFrames, nBinsFull]
    float* spec_db,         // [nFrames, Hvis]
    int nFrames,
    int nBinsFull,
    int Hvis,
    float fs,
    float winSumSq,         // sum(w^2)
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nFrames * Hvis;
    if (idx >= total) return;

    int frame = idx / Hvis;
    int bin = idx - frame * Hvis;

    cufftComplex c = X[frame * nBinsFull + bin];
    float re = c.x, im = c.y;

    float mag2 = re * re + im * im; // |X|^2

    // PSD density scaling (dB/Hz)
    float psd = mag2 / (fmaxf(1e-12f, fs * winSumSq));

    // one-sided correction (for real signals): *2 except DC (and Nyquist if present)
    if (bin > 0 && bin < nBinsFull - 1) psd *= 2.0f;


    spec_db[idx] = 10.0f * log10f(psd + eps);
}


// Kernel: Escribe columnas del buffer actual en el ring buffer histórico
// Actualiza d_hist_db con nuevos datos del frame procesado
__global__ void write_hist_cols(
    const float* spec_db_block,  // Espectrograma nuevo [B, H]
    float* hist_db,              // Ring buffer histórico [Wvis, H]
    int B,                       // Número de frames en este bloque
    int H,                       // Alto (bins)
    int Wvis,                    // Ancho del ring buffer
    int head)                    // Posición de escribir en el ring
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx >= total) return;

    int f = idx / H;   // Qué frame dentro del bloque
    int bin = idx - f * H; // Qué bin de frecuencia

    // Columna de escribir = (head + f) % Wvis (circular)
    int col = (head + f) % Wvis;

    // Copia desde buffer nuevo al ring buffer
    hist_db[col * H + bin] = spec_db_block[f * H + bin];
}



// Kernel: Inicializa estados de movimiento de partículas


__global__ void hist_to_vbos_3d_nomotion(
    const float* hist_db,
    float4* outPosVBO,
    float4* outColVBO,
    int Wvis, int H,
    int head,
    const float* scale,
    float dbMinManual, float dbMaxManual,
    int autoGain,
    int cmap,
    float gammaColor,
    float binHz,
    float fAxisMaxHz,
    float zScale,
    int viewStart,         // NUEVO (en dispCol)
    int visibleFrames)     // NUEVO
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Wvis * H;
    if (idx >= N) return;

    int col = idx % Wvis;
    int row = idx / Wvis;

    int dispCol = (col - head + Wvis) % Wvis; // 0..Wvis-1

    // ---- CULL: si está fuera del viewport visible, no lo dibujes ----
    if (dispCol < viewStart) {
        outPosVBO[idx] = make_float4(2.0f, 2.0f, 2.0f, 1.0f); // fuera de clip
        outColVBO[idx] = make_float4(0, 0, 0, 0);             // alpha 0
        return;
    }

    int d = dispCol - viewStart; // 0..visibleFrames-1
    float fx = (visibleFrames > 1) ? (float)d / (float)(visibleFrames - 1) : 0.0f;

    float freq = row * binHz;
    float fy = (fAxisMaxHz > 1e-6f) ? (freq / fAxisMaxHz) : 0.0f;
    fy = fminf(1.0f, fmaxf(0.0f, fy));

    float x = fx * 2.0f - 1.0f;
    float y = fy * 2.0f - 1.0f;

    float v = hist_db[col * H + row]; // dB/Hz real

    float dbMin = autoGain ? scale[0] : dbMinManual;
    float dbMax = autoGain ? scale[1] : dbMaxManual;

    float vc = fminf(dbMax, fmaxf(dbMin, v));
    float tLin = (vc - dbMin) / fmaxf(1e-6f, (dbMax - dbMin));
    tLin = fminf(1.0f, fmaxf(0.0f, tLin));

    float z = (tLin - 0.5f) * 2.0f * zScale;

    float tCol = powf(tLin, fmaxf(0.05f, gammaColor));
    float3 rgb = apply_cmap(cmap, tCol);

    outColVBO[idx] = make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
    outPosVBO[idx] = make_float4(x, y, z, 1.0f);
}


// ================= GL shader utils =================
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<GLchar> log((size_t)std::max(1, len));
        GLsizei outLen = 0;
        glGetShaderInfoLog(s, len, &outLen, log.data());
        std::cerr << "Shader compile error:\n" << log.data() << "\n";
        std::exit(1);
    }
    return s;
}

static GLuint createProgramFromSrc(const char* vsSrc, const char* fsSrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);

    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::vector<GLchar> log((size_t)std::max(1, len));
        GLsizei outLen = 0;
        glGetProgramInfoLog(p, len, &outLen, log.data());
        std::cerr << "Program link error:\n" << log.data() << "\n";
        std::exit(1);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

static GLuint createMainProgram3D() {
    const char* vsSrc = R"(
        #version 330 core
        layout(location=0) in vec4 aPos;
        layout(location=1) in vec4 aCol;
        out vec4 vCol;

        uniform mat4 uMVP;
        uniform float uPointSize;

        void main(){
            gl_Position = uMVP * vec4(aPos.xyz, 1.0);
            gl_PointSize = uPointSize;
            vCol = aCol;
        }
    )";

    const char* fsSrc = R"(
        #version 330 core
        in vec4 vCol;
        out vec4 FragColor;
        void main(){ FragColor = vCol; }
    )";

    return createProgramFromSrc(vsSrc, fsSrc);
}
static GLuint createPickProgram()
{
    const char* vsSrc = R"(
        #version 330 core
        layout(location=0) in vec4 aPos;

        uniform mat4 uMVP;
        uniform float uPointSize;

        flat out uint vID;

        void main(){
            gl_Position = uMVP * vec4(aPos.xyz, 1.0);
            gl_PointSize = uPointSize;
            vID = uint(gl_VertexID) + 1u; // +1 para que 0 sea "fondo"
        }
    )";

    const char* fsSrc = R"(
        #version 330 core
        flat in uint vID;
        out vec4 FragColor;

        void main(){
            // empaqueta vID (24 bits) en RGB8
            uint id = vID;
            uint r = (id      ) & 255u;
            uint g = (id >>  8) & 255u;
            uint b = (id >> 16) & 255u;
            FragColor = vec4(float(r)/255.0, float(g)/255.0, float(b)/255.0, 1.0);
        }
    )";

    return createProgramFromSrc(vsSrc, fsSrc);
}

static void destroyPicking()
{
    if (pickDepth) { glDeleteRenderbuffers(1, &pickDepth); pickDepth = 0; }
    if (pickTex) { glDeleteTextures(1, &pickTex); pickTex = 0; }
    if (pickFBO) { glDeleteFramebuffers(1, &pickFBO); pickFBO = 0; }
    if (pickProg) { glDeleteProgram(pickProg); pickProg = 0; }
}

static void createPicking(int W, int H)
{
    if (!pickProg) {
        pickProg = createPickProgram();
        uPickMVPLoc = glGetUniformLocation(pickProg, "uMVP");
        uPickPointSizeLoc = glGetUniformLocation(pickProg, "uPointSize");
    }

    if (!pickFBO) glGenFramebuffers(1, &pickFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, pickFBO);

    // Color texture RGBA8
    if (!pickTex) glGenTextures(1, &pickTex);
    glBindTexture(GL_TEXTURE_2D, pickTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pickTex, 0);

    // Depth (para que el picking respete profundidad si quieres)
    if (!pickDepth) glGenRenderbuffers(1, &pickDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, pickDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, pickDepth);

    GLenum drawBuf = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuf);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Pick FBO incompleto!\n";
        std::exit(1);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}



static GLuint createAxisProgram() {
    const char* vsSrc = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uMVP;
        void main(){
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
    )";
    const char* fsSrc = R"(
        #version 330 core
        uniform vec4 uColor;
        out vec4 FragColor;
        void main(){ FragColor = uColor; }
    )";
    return createProgramFromSrc(vsSrc, fsSrc);
}

static GLuint createOverlayProgram() {
    const char* vsSrc = R"(
        #version 330 core
        layout(location=0) in vec2 aPosPx;
        layout(location=1) in vec4 aCol;
        out vec4 vCol;
        uniform vec2 uViewport;

        void main(){
            vec2 ndc = vec2(
                (aPosPx.x / uViewport.x) * 2.0 - 1.0,
                (aPosPx.y / uViewport.y) * 2.0 - 1.0
            );
            gl_Position = vec4(ndc, 0.0, 1.0);
            vCol = aCol;
        }
    )";

    const char* fsSrc = R"(
        #version 330 core
        in vec4 vCol;
        out vec4 FragColor;
        void main(){ FragColor = vCol; }
    )";

    return createProgramFromSrc(vsSrc, fsSrc);
}

static GLuint createVBO(size_t bytes) {
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

// ================= Hann window =================
static void buildHannWindow(std::vector<float>& h_win, int fftSize_) {
    h_win.resize(fftSize_);
    for (int i = 0; i < fftSize_; ++i)
        h_win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (fftSize_ - 1)));

    double sumsq = 0.0;
    for (int i = 0; i < fftSize_; ++i) sumsq += (double)h_win[i] * (double)h_win[i];
    gWinSumSq = (float)(std::max(1e-12, sumsq));
    double l2 = std::sqrt((std::max)(1e-12, sumsq));
    gInvWinL2 = (float)(1.0 / l2);
}

// ================= 3D axis + grid geometry =================
static void buildAxisBoxAndGrid()
{
    gAxisVerts.clear();
    gAxisEdgeCount = 0;
    gAxisGridCount = 0;

    auto addLine = [&](float x0, float y0, float z0, float x1, float y1, float z1) {
        gAxisVerts.push_back({ x0,y0,z0 });
        gAxisVerts.push_back({ x1,y1,z1 });
        };

    const float zS = g.zScale;
    const float x0 = -1.0f, x1 = 1.0f;
    const float y0 = -1.0f, y1 = 1.0f;
    const float z0 = -zS, z1 = zS;

    // 8 corners
    Vec3 c000(x0, y0, z0), c100(x1, y0, z0), c010(x0, y1, z0), c110(x1, y1, z0);
    Vec3 c001(x0, y0, z1), c101(x1, y0, z1), c011(x0, y1, z1), c111(x1, y1, z1);

    // 12 edges
    addLine(c000.x, c000.y, c000.z, c100.x, c100.y, c100.z);
    addLine(c010.x, c010.y, c010.z, c110.x, c110.y, c110.z);
    addLine(c001.x, c001.y, c001.z, c101.x, c101.y, c101.z);
    addLine(c011.x, c011.y, c011.z, c111.x, c111.y, c111.z);

    addLine(c000.x, c000.y, c000.z, c010.x, c010.y, c010.z);
    addLine(c100.x, c100.y, c100.z, c110.x, c110.y, c110.z);
    addLine(c001.x, c001.y, c001.z, c011.x, c011.y, c011.z);
    addLine(c101.x, c101.y, c101.z, c111.x, c111.y, c111.z);

    addLine(c000.x, c000.y, c000.z, c001.x, c001.y, c001.z);
    addLine(c100.x, c100.y, c100.z, c101.x, c101.y, c101.z);
    addLine(c010.x, c010.y, c010.z, c011.x, c011.y, c011.z);
    addLine(c110.x, c110.y, c110.z, c111.x, c111.y, c111.z);

    gAxisEdgeCount = (int)gAxisVerts.size();

    if (!g.show3DGrid) {
        gAxisGridCount = 0;
        return;
    }

    // ---- Gridlines every 5 Hz (lines across X on back plane z=-zScale) ----
    int fMaxTick = (int)std::round(fTopShownHz);
    if (fMaxTick < 5) fMaxTick = 5;

    for (int f = 0; f <= (int)fTopShownHz; f += 5) {

        float t = (float)f / fTopShownHz;          // 0..1
        float yy = -1.0f + clamp01(t) * 2.0f;         // -1..1

        addLine(x0, yy, z0, x1, yy, z0);
    }

    // ---- Time gridlines every N columns (vertical lines in Y, on back plane z=-zScale) ----
    int every = (std::max)(4, g.timeGridEveryCols);
    int vs = (std::max)(2, gVisibleFrames);
    int vStart = (std::max)(0, (std::min)(Wvis - 1, gViewStart));

    for (int col = vStart; col < Wvis; col += every) {
        int d = col - vStart; // 0..visibleFrames-1
        float fx = (vs > 1) ? (float)d / (float)(vs - 1) : 0.0f;
        float xx = fx * 2.0f - 1.0f;
        addLine(xx, y0, z0, xx, y1, z0);
    }

    gAxisGridCount = (int)gAxisVerts.size() - gAxisEdgeCount;
}

// ================= Init pipeline (CUDA+cuFFT) =================
static void initPipeline() {
    CUDA_CHECK(cudaSetDevice(0));

    if (!gStream) CUDA_CHECK(cudaStreamCreateWithFlags(&gStream, cudaStreamNonBlocking));
    for (int i = 0; i < kBuffers; ++i)
        if (!gReadyEvent[i]) CUDA_CHECK(cudaEventCreateWithFlags(&gReadyEvent[i], cudaEventDisableTiming));

    if (!h_scalePinned) CUDA_CHECK(cudaHostAlloc((void**)&h_scalePinned, 2 * sizeof(float), cudaHostAllocDefault));
    if (!gScaleEvent)   CUDA_CHECK(cudaEventCreateWithFlags(&gScaleEvent, cudaEventDisableTiming));
    if (!h_hoverPinned) CUDA_CHECK(cudaHostAlloc((void**)&h_hoverPinned, sizeof(float), cudaHostAllocDefault));
    if (!gHoverEvent)   CUDA_CHECK(cudaEventCreateWithFlags(&gHoverEvent, cudaEventDisableTiming));

    for (int i = 0; i < kBuffers; ++i) CUDA_CHECK(cudaEventRecord(gReadyEvent[i], gStream));
    CUDA_CHECK(cudaStreamSynchronize(gStream));

    // Resolve CSV path (gCsvPath may be set by CLI). Ensure it's resolved against repo root.
    fs::path csv_resolved = resolveRepoPath(gCsvPath);
    std::cout << "Repo root (detected): " << findRepoRoot().string() << "\n";
    std::cout << "Using CSV: " << csv_resolved.string() << "\n";

    if (h_channels.empty()) {
        if (!load_csv_channels(csv_resolved.string(), h_channels)) {
            std::cerr << "No pude leer CSV: " << csv_resolved.string() << "\n"; std::exit(1);
        }
        std::cout << "CSV cargado. Canales=" << (int)h_channels.size() << "\n";
    }

    g.channel = (std::max)(0, (std::min)(g.channel, (int)h_channels.size() - 1));
    h_signal_raw = h_channels[g.channel];
    nSamples = (int)h_signal_raw.size();

    h_signal_filt = h_signal_raw;
    if (g.preFilter) preprocess_filter_inplace(h_signal_filt, g.fs, g.removeMean, g.useHighpass, g.highHz, g.lowHz);
    else if (g.removeMean) {
        double m = 0.0;
        for (float v : h_signal_filt) m += (double)v;
        m /= (double)h_signal_filt.size();
        for (float& v : h_signal_filt) v = (float)(v - m);
    }

    h_signal = g.showFiltered ? h_signal_filt : h_signal_raw;

    auto rms = [](const std::vector<float>& v) -> float {
        double s2 = 0.0;
        for (float x : v) s2 += (double)x * (double)x;
        return (float)std::sqrt(s2 / (double)((std::max)(size_t(1), v.size())));
        };
    gRmsRaw = rms(h_signal_raw);
    gRmsFilt = rms(h_signal_filt);

    double s2 = 0.0;
    for (size_t i = 0; i < h_signal_raw.size(); ++i) {
        double d = (double)h_signal_raw[i] - (double)h_signal_filt[i];
        s2 += d * d;
    }
    gRmsRemoved = (float)std::sqrt(s2 / (double)((std::max)(size_t(1), h_signal_raw.size())));
    gRemovedPct = (gRmsRaw > 1e-12f) ? (100.0f * gRmsRemoved / gRmsRaw) : 0.0f;

    // ---- FFT bins ----
    nBinsFull = fftSize / 2 + 1;

    float nyq = 0.5f * g.fs;
    float fMaxVis = (std::min)(g.lowHz, nyq);

    binHz = (g.fs > 1.0f) ? (g.fs / (float)fftSize) : 1.0f;

    maxBinVis = (int)std::floor(fMaxVis / binHz + 1e-9f);
    maxBinVis = (std::max)(1, (std::min)(maxBinVis, nBinsFull - 1));

    H = maxBinVis + 1;
    Nvis = Wvis * H;
    fTopShownHz = maxBinVis * binHz;

    B = stepFrames;

    maxStartSample = nSamples - ((B - 1) * hop + fftSize);
    if (maxStartSample < 0) {
        std::cerr << "No caben B columnas. Ajusta hop/fftSize o baja B.\n";
        std::exit(1);
    }
    startSample = 0;
    gHead = 0;
    gColSample0.assign(Wvis, -1);





    std::vector<float> h_win;
    buildHannWindow(h_win, fftSize);

    CUDA_CHECK(cudaMalloc(&d_signal, (size_t)nSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_win, (size_t)fftSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_signal, h_signal.data(), (size_t)nSamples * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win, h_win.data(), (size_t)fftSize * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < kBuffers; ++i) {
        CUDA_CHECK(cudaMalloc(&d_frames[i], (size_t)B * fftSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fft[i], (size_t)B * nBinsFull * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_spec_db[i], (size_t)B * H * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_minmax[i], 2 * sizeof(float)));
    }

    CUDA_CHECK(cudaMalloc(&d_scale, 2 * sizeof(float)));
    float initScale[2] = { g.dbMin, g.dbMax };
    CUDA_CHECK(cudaMemcpy(d_scale, initScale, 2 * sizeof(float), cudaMemcpyHostToDevice));
    gScaleHost[0] = g.dbMin; gScaleHost[1] = g.dbMax;

    CUDA_CHECK(cudaMalloc(&d_hist_db, (size_t)Wvis * H * sizeof(float)));
    {
        int total = Wvis * H;
        int threads = 256, blocks = (total + threads - 1) / threads;
        fill_hist << <blocks, threads, 0, gStream >> > (d_hist_db, total, g.dbMin);
        CUDA_CHECK(cudaGetLastError());
    }



    CUFFT_CHECK(cufftCreate(&plan));
    int rank = 1; int n[1] = { fftSize };
    int inembed[1] = { fftSize };
    int onembed[1] = { nBinsFull };
    int istride = 1, ostride = 1;
    int idist = fftSize, odist = nBinsFull;

    CUFFT_CHECK(cufftPlanMany(&plan, rank, n,
        inembed, istride, idist,
        onembed, ostride, odist,
        CUFFT_R2C, B));
    CUFFT_CHECK(cufftSetStream(plan, gStream));

    for (int i = 0; i < kBuffers; ++i) gHasData[i] = false;
    gPresent = 0; gQueued = 0;
    for (int i = 0; i < kBuffers; ++i) gHeadForBuffer[i] = 0;
    syncSliderFromSpeed();
    syncTimeSlidersFromState();

}

// ================= Init GL + interop =================
static void initGLandInterop() {
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW init failed: " << glewGetErrorString(err) << "\n";
        std::exit(1);
    }
    while (glGetError() != GL_NO_ERROR) {}

    program = createMainProgram3D();
    glUseProgram(program);

    uMvpLoc = glGetUniformLocation(program, "uMVP");
    uPointSizeLoc = glGetUniformLocation(program, "uPointSize");

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);

    for (int i = 0; i < kBuffers; ++i) {
        vboPos[i] = createVBO(sizeof(float4) * (size_t)Nvis);
        vboCol[i] = createVBO(sizeof(float4) * (size_t)Nvis);

        glGenVertexArrays(1, &vao[i]);
        glBindVertexArray(vao[i]);

        glBindBuffer(GL_ARRAY_BUFFER, vboPos[i]);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, vboCol[i]);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    CUDA_CHECK(cudaSetDevice(0));
    for (int i = 0; i < kBuffers; ++i) {
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPosRes[i], vboPos[i], cudaGraphicsRegisterFlagsWriteDiscard));
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaColRes[i], vboCol[i], cudaGraphicsRegisterFlagsWriteDiscard));
    }

    // ---- 3D axis & gridlines ----
    axisProg = createAxisProgram();
    uAxisMVPLoc = glGetUniformLocation(axisProg, "uMVP");
    uAxisColorLoc = glGetUniformLocation(axisProg, "uColor");

    glGenVertexArrays(1, &axisVAO);
    glBindVertexArray(axisVAO);

    glGenBuffers(1, &axisVBO);
    glBindBuffer(GL_ARRAY_BUFFER, axisVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(AxisVert), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    buildAxisBoxAndGrid();
    glBindBuffer(GL_ARRAY_BUFFER, axisVBO);
    glBufferData(GL_ARRAY_BUFFER, gAxisVerts.size() * sizeof(AxisVert), gAxisVerts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ---- Overlay (2D axis+legend, se conserva) ----
    overlayProg = createOverlayProgram();
    uViewportLoc = glGetUniformLocation(overlayProg, "uViewport");

    glGenVertexArrays(1, &overlayVAO);
    glBindVertexArray(overlayVAO);

    glGenBuffers(1, &overlayVBO);
    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVert), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(OverlayVert), (void*)(2 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


    createPicking(winW, winH);

    glBindVertexArray(0);

}

// ================= compute: STFT B columnas =================
static void computeSTFTBlock(int bufIdx) {
    int threads = 256;

    int total = B * fftSize;
    int blocks = (total + threads - 1) / threads;

    build_frames_hann_offset << <blocks, threads, 0, gStream >> > (
        d_signal, nSamples, d_frames[bufIdx], d_win,
        fftSize, hop, B, startSample);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecR2C(plan, (cufftReal*)d_frames[bufIdx], (cufftComplex*)d_fft[bufIdx]));

    int total2 = B * H;
    int blocks2 = (total2 + threads - 1) / threads;

    psd_db_from_r2c_onesided << <blocks2, threads, 0, gStream >> > (
        d_fft[bufIdx], d_spec_db[bufIdx],
        B, nBinsFull, H,
        g.fs,
        gWinSumSq,
        1e-20f);
    CUDA_CHECK(cudaGetLastError());

    if (g.autoGain) {
        init_minmax << <1, 1, 0, gStream >> > (d_minmax[bufIdx]);
        reduce_minmax << <blocks2, threads, 0, gStream >> > (d_spec_db[bufIdx], total2, d_minmax[bufIdx]);
        smooth_scale << <1, 1, 0, gStream >> > (d_minmax[bufIdx], d_scale, 0.08f);
    }
    else {
        float s[2] = { g.dbMin, g.dbMax };
        CUDA_CHECK(cudaMemcpyAsync(d_scale, s, 2 * sizeof(float), cudaMemcpyHostToDevice, gStream));
    }

    CUDA_CHECK(cudaMemcpyAsync(h_scalePinned, d_scale, 2 * sizeof(float), cudaMemcpyDeviceToHost, gStream));
    CUDA_CHECK(cudaEventRecord(gScaleEvent, gStream));
}



// ================= write hist + write VBOs =================
static void produceFrame(int bufIdx) {
    int blockStartSample = startSample;  // el startSample usado para esta tanda
    int headBefore = gHead;              // head donde se van a escribir estas B columnas

    computeSTFTBlock(bufIdx);

    int total = B * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    write_hist_cols << <blocks, threads, 0, gStream >> > (d_spec_db[bufIdx], d_hist_db, B, H, Wvis, gHead);
    CUDA_CHECK(cudaGetLastError());

    for (int f = 0; f < B; ++f) {
        int colw = (headBefore + f) % Wvis;
        gColSample0[colw] = blockStartSample + f * hop;
    }

    gHead = (gHead + B) % Wvis;
    gHeadForBuffer[bufIdx] = gHead;

    cudaGraphicsResource* res[2] = { cudaPosRes[bufIdx], cudaColRes[bufIdx] };
    CUDA_CHECK(cudaGraphicsMapResources(2, res, gStream));

    float4* dPosVBO = nullptr; size_t b0 = 0;
    float4* dColVBO = nullptr; size_t b1 = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dPosVBO, &b0, cudaPosRes[bufIdx]));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dColVBO, &b1, cudaColRes[bufIdx]));

    int threads2 = 256;
    int blocks2 = (Nvis + threads2 - 1) / threads2;

    hist_to_vbos_3d_nomotion << <blocks2, threads2, 0, gStream >> > (
        d_hist_db,
        dPosVBO, dColVBO,
        Wvis, H,
        gHeadForBuffer[bufIdx],
        d_scale,
        g.dbMin, g.dbMax,
        g.autoGain ? 1 : 0,
        g.cmap, g.gamma,
        binHz,
        fTopShownHz,   // <- mejor que 45 fijo
        g.zScale, gViewStart, gVisibleFrames);

    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cudaGraphicsUnmapResources(2, res, gStream));

    CUDA_CHECK(cudaEventRecord(gReadyEvent[bufIdx], gStream));
    gHasData[bufIdx] = true;

    startSample += (B * hop);
    if (startSample > maxStartSample) {
        if (gLoopPlayback) {
            startSample = 0;
        }
        else {
            startSample = maxStartSample;
            g.paused = true;
            gAccumSamples = 0.0;
            resetPlaybackClock();
        }
    }
}


static void refreshVBOForBuffer(int bufIdx)
{
    if (!gHasData[bufIdx] || !d_hist_db) return;

    if (cudaEventQuery(gReadyEvent[bufIdx]) != cudaSuccess) {
        CUDA_CHECK(cudaEventSynchronize(gReadyEvent[bufIdx]));
    }

    cudaGraphicsResource* res[2] = { cudaPosRes[bufIdx], cudaColRes[bufIdx] };
    CUDA_CHECK(cudaGraphicsMapResources(2, res, gStream));

    float4* dPosVBO = nullptr; size_t b0 = 0;
    float4* dColVBO = nullptr; size_t b1 = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dPosVBO, &b0, cudaPosRes[bufIdx]));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dColVBO, &b1, cudaColRes[bufIdx]));

    int threads2 = 256;
    int blocks2 = (Nvis + threads2 - 1) / threads2;

    hist_to_vbos_3d_nomotion << <blocks2, threads2, 0, gStream >> > (
        d_hist_db,
        dPosVBO, dColVBO,
        Wvis, H,
        gHeadForBuffer[bufIdx],
        d_scale,
        g.dbMin, g.dbMax,
        g.autoGain ? 1 : 0,
        g.cmap, g.gamma,
        binHz,
        fTopShownHz,
        g.zScale,
        gViewStart, gVisibleFrames
        );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaGraphicsUnmapResources(2, res, gStream));
    CUDA_CHECK(cudaEventRecord(gReadyEvent[bufIdx], gStream));
}


// ================= triple-buffer picker =================
static int pickComputeBuffer() {
    for (int i = 0; i < kBuffers; ++i) {
        if (i == gPresent || i == gQueued) continue;
        if (cudaEventQuery(gReadyEvent[i]) == cudaSuccess) return i;
    }
    for (int i = 0; i < kBuffers; ++i) if (i != gPresent) return i;
    return 0;
}

// ================= text helper =================
static void drawText(int x, int y, const std::string& s) {
    glRasterPos2i(x, y);
    for (char c : s) glutBitmapCharacter(GLUT_BITMAP_8_BY_13, c);

}


// ================ = ================ =

static void drawHzLabelsOnCubeY(const Mat4& MVP)
{
    const float fAxisMaxHz = fTopShownHz;
    const int step = 5;

    // Elegimos un borde del cubo para poner el texto:
    // x ligeramente a la izquierda para que no se solape con el borde
    float x = -1.08f;
    float z = -g.zScale;       // plano “trasero” del cubo (coherente con tu axis box)


    glDisable(GL_DEPTH_TEST);

    // Pasamos a 2D
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 1, 1);

    for (int f = 0; f <= (int)fAxisMaxHz; f += step)
    {
        float t = (float)f / fAxisMaxHz;     // 0..1
        float y = -1.0f + 2.0f * t;          // -1..1 (eje Y del cubo)

        float sx, sy;
        if (projectToScreen(Vec3(x, y, z), MVP, winW, winH, sx, sy))
        {
            drawText((int)sx, (int)sy - 4, std::to_string(f));
        }
    }
    float t = 1.0f;                // 45/45
    float y = -1.0f + 2.0f * t;     // y del tick superior

    // un pelín más a la izquierda que los números para que no solape
    float xHz = -1.15f;
    float zHz = -g.zScale;

    float sx, sy;
    if (projectToScreen(Vec3(xHz, y, zHz), MVP, winW, winH, sx, sy))
    {
        drawText((int)sx, (int)sy + 12, "Hz"); // +12 px para que quede encima/pegado
    }

    glEnable(GL_DEPTH_TEST);
}


static float dbToZWorld(float db, float dbMin, float dbMax, float zScale)
{
    float denom = (dbMax - dbMin);
    if (denom < 1e-6f) denom = 1.0f;

    float t = (db - dbMin) / denom; // lineal
    t = fminf(1.0f, fmaxf(0.0f, t));
    return (t - 0.5f) * 2.0f * zScale;
}

// ================ = ================ =

static inline void updateScaleHostIfReady()
{
    if (gScaleEvent && h_scalePinned && cudaEventQuery(gScaleEvent) == cudaSuccess) {
        gScaleHost[0] = h_scalePinned[0];
        gScaleHost[1] = h_scalePinned[1];
    }
}

static void drawDbLabelsOnCubeZ(const Mat4& MVP)
{
    // Asegura que gScaleHost esté actualizado si autoGain
    updateScaleHostIfReady();

    float dbMin = g.autoGain ? gScaleHost[0] : g.dbMin;
    float dbMax = g.autoGain ? gScaleHost[1] : g.dbMax;
    if (dbMax < dbMin + 1e-3f) dbMax = dbMin + 1.0f;

    // 3 ticks: arriba/medio/abajo (como tu leyenda 2D)
    float dbTop = dbMax;
    float dbMid = 0.5f * (dbMin + dbMax);
    float dbBot = dbMin;

    // Elegimos un borde del cubo donde variar Z:
    // x a la derecha, y abajo (fuera un pelín para que no se meta en la nube)
    const float x = -1.08f;
    const float y = -1.08f;

    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 1, 1);

    auto drawOne = [&](float dbVal, const char* txt)
        {
            float z = dbToZWorld(dbVal, dbMin, dbMax, g.zScale);
            float sx, sy;
            if (projectToScreen(Vec3(x, y, z), MVP, winW, winH, sx, sy)) {
                drawText((int)sx, (int)sy - 12, txt);
            }
        };

    // Formatea como entero (rápido/limpio)
    std::string sTop = std::to_string((int)std::round(dbTop));
    std::string sMid = std::to_string((int)std::round(dbMid));
    std::string sBot = std::to_string((int)std::round(dbBot));

    drawOne(dbTop, sTop.c_str());
    drawOne(dbMid, sMid.c_str());
    drawOne(dbBot, sBot.c_str());

    // Unidad "dB" una vez, cerca del tick superior, un poco desplazada
    {
        float z = dbToZWorld(dbTop, dbMin, dbMax, g.zScale);
        float sx, sy;
        if (projectToScreen(Vec3(x + 0.04f, y, z), MVP, winW, winH, sx, sy)) {
            drawText((int)sx, (int)sy - 12, "dB");
        }
    }

    glEnable(GL_DEPTH_TEST);
}

// Mantén el slider sincronizado si NO estás arrastrándolo
static void syncSeekSliderFromState()
{
    if (gDragSeek) return;
    gSeekT = tFromSeekSec(rightEdgeSec());
    gSeekSecPending = seekSecFromT(gSeekT);
    gSeekDirty = false;
}

// Prefill: vacía ring y computa suficientes frames para que la ventana salga "llena"
static inline int minStartForPrefill()
{
    // tiempo = inicio de ventana => arrancamos en 0
    return 0;
}


static void prefillWindowAtStartSample(int start0)
{
    int lo = minStartForPrefill();
    int hi = maxStartForPrefill();
    start0 = (std::max)(lo, (std::min)(start0, hi));

    gAccumSamples = 0.0;
    resetPlaybackClock();

    updateTimeView();

    startSample = start0;
    clearHistAndResetScroll();

    for (int i = 0; i < kBuffers; ++i) {
        gHasData[i] = false;
        gHeadForBuffer[i] = 0;
    }

    int buf = 0;
    gPresent = buf;
    gQueued = buf;

    int fills = blocksToFillWindow();
    for (int i = 0; i < fills; ++i) {
        if (cudaEventQuery(gReadyEvent[buf]) != cudaSuccess)
            CUDA_CHECK(cudaEventSynchronize(gReadyEvent[buf]));

        produceFrame(buf);
        CUDA_CHECK(cudaEventSynchronize(gReadyEvent[buf]));
    }

    gHoverValid = false;
    gViewDirty = true;

    // MUY importante: sincroniza el slider al estado real tras prefill
    syncSeekSliderFromState();
}


// Aplica seek interpretando el slider como “borde derecho” de la ventana
// NOTA: corregido para que el rightEdge quede alineado incluso si Wvis no es múltiplo de B
static void transportSeekToRightEdgeSec(double rightSec, bool resumeAfter)
{
    rightSec = clampd(rightSec, seekMinRightSec(), seekMaxRightSec());

    int rightSample = (int)std::llround(rightSec * fsSafeD()); // rightSec = s0/fs
    int lastFrameS0 = rightSample;                              // NO -fftSize/2


    int fills = blocksToFillWindow();
    int producedCols = fills * B;

    int start0 = lastFrameS0 - (producedCols - 1) * hop;

    prefillWindowAtStartSample(start0);

    g.paused = !resumeAfter;
    gAccumSamples = 0.0;
    resetPlaybackClock();
}




// Drag handler
static void setSeekFromMouseX(int mx)
{
    gSeekT = sliderTFromMouseX(mx, gSeekX, gSeekW);
    gSeekSecPending = seekSecFromT(gSeekT);
    gSeekDirty = true;
}


// ================= Overlay render (axis + legend) (SE CONSERVA 2D) =================
static void renderAxisAndLegendOverlay() {
    updateScaleHostIfReady();

    float dbMin = g.autoGain ? gScaleHost[0] : g.dbMin;
    float dbMax = g.autoGain ? gScaleHost[1] : g.dbMax;
    if (dbMax < dbMin + 1e-3f) dbMax = dbMin + 1.0f;

    // Layout overlay (px)
    int axisBgW = 70;

    int y0 = 60;
    int y1 = winH - 60;

    int barW = 18;
    int barH = 220;
    int xBar = winW - 70;
    int yBar = 70;
    int legendBgW = 95;

    const int seg = 80;
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(4096);
    lineV.reserve(512);

    overlayAddTri(triV, 0.0f, (float)(y0 - 15), (float)axisBgW, (float)(y1 + 15), 0, 0, 0, 0.65f);
    overlayAddTri(triV, (float)(winW - legendBgW), (float)(yBar - 20),
        (float)(winW), (float)(yBar + barH + 55), 0, 0, 0, 0.65f);

    // Gradiente leyenda
    for (int i = 0; i < seg; ++i) {
        float t0 = (float)i / (float)seg;
        float t1 = (float)(i + 1) / (float)seg;
        float tm = 0.5f * (t0 + t1);
        float tg = std::pow(clamp01(tm), (std::max)(0.05f, g.gamma));

        float r, gg, bb;
        if (g.cmap == 0) cmap_heat_cpu(tg, r, gg, bb);
        else            cmap_gray_cpu(tg, r, gg, bb);

        float yy0 = (float)yBar + t0 * (float)barH;
        float yy1 = (float)yBar + t1 * (float)barH;
        overlayAddTri(triV, (float)xBar, yy0, (float)(xBar + barW), yy1, r, gg, bb, 1.0f);
    }



    // Borde leyenda
    overlayAddLine(lineV, (float)(xBar - 1), (float)(yBar - 1), (float)(xBar + barW + 1), (float)(yBar - 1), 1, 1, 1, 1);
    overlayAddLine(lineV, (float)(xBar + barW + 1), (float)(yBar - 1), (float)(xBar + barW + 1), (float)(yBar + barH + 1), 1, 1, 1, 1);
    overlayAddLine(lineV, (float)(xBar + barW + 1), (float)(yBar + barH + 1), (float)(xBar - 1), (float)(yBar + barH + 1), 1, 1, 1, 1);
    overlayAddLine(lineV, (float)(xBar - 1), (float)(yBar + barH + 1), (float)(xBar - 1), (float)(yBar - 1), 1, 1, 1, 1);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // TRI
    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    // LINES
    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // Text labels
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 1, 1);



    // Labels leyenda
    std::ostringstream s0, s1, s2;
    s0 << (int)std::round(dbMax) << " dB/Hz";
    s1 << (int)std::round(0.5f * (dbMin + dbMax)) << " dB/Hz";
    s2 << (int)std::round(dbMin) << " dB/Hz";

    drawText(xBar - 85, yBar + barH - 5, s0.str());
    drawText(xBar - 85, yBar + barH / 2 - 5, s1.str());
    drawText(xBar - 85, yBar - 5, s2.str());

    drawText(xBar - 55, yBar + barH + 15, (g.cmap == 0 ? "cmap: heat" : "cmap: gray"));
    drawText(xBar - 55, yBar + barH + 30, (g.autoGain ? "autoGain: ON" : "autoGain: OFF"));
}

static void updateUILayout()
{
    // ---- tamaños compactos ----
    const int padX = 12;
    const int padY = 140;
    const int sliderHCalc = 12;
    const int sliderHView = 14;
    const int rowGap = -50;

    // ===== Calc box =====
    gZoomX = gCalcBoxX + padX;  gZoomW = gCalcBoxW - 2 * padX;  gZoomH = sliderHCalc;
    gHopX = gZoomX;            gHopW = gZoomW;                gHopH = sliderHCalc;
    gFftX = gZoomX;            gFftW = gZoomW;                gFftH = sliderHCalc;

    int top = gCalcBoxY + gCalcBoxH - padY;
    gZoomY = top - sliderHCalc;         // fila 1 (winSec)
    gHopY = gZoomY - rowGap;           // fila 2 (hop)
    gFftY = gHopY - rowGap;           // fila 3 (fft)

    // Apply abajo-dcha (más natural + no tapa)
    gApplyW = 80; gApplyH = 20;
    gApplyX = gCalcBoxX + gCalcBoxW - gApplyW - padX;
    gApplyY = gCalcBoxY + 10;

    // ===== View box (igual que antes, pero consistente) =====
    gSpeedX = gViewBoxX + 14; gSpeedW = gViewBoxW - 28; gSpeedH = sliderHView;
    gSpeedY = gViewBoxY + 48;

    gTimeZoomX = gViewBoxX + 14; gTimeZoomW = gViewBoxW - 28; gTimeZoomH = sliderHView;
    gTimeZoomY = gViewBoxY + 10;

    // ===== Transport box (encima de View, mismo ancho) =====
    gTransportBoxX = gViewBoxX;
    gTransportBoxW = gViewBoxW;
    gTransportBoxH = 90;
    gTransportBoxY = gViewBoxY + gViewBoxH + 25;

    // Slider seek dentro de transport
    gSeekX = gTransportBoxX + 14;
    gSeekW = gTransportBoxW - 28;
    gSeekH = 14;
    gSeekY = gTransportBoxY + 10;

    // Botones
    int by = gTransportBoxY + 50;
    int bx = gTransportBoxX + 14;
    int gap = 8;

    gBtnPlayX = bx; gBtnPlayY = by;
    gBtnRestartX = gBtnPlayX + gBtnPlayW + gap; gBtnRestartY = by;
    gBtnStepX = gBtnRestartX + gBtnRestartW + gap; gBtnStepY = by;
    gBtnLoopX = gBtnStepX + gBtnStepW + gap; gBtnLoopY = by;
    // ===== Tone box (TOP-RIGHT) =====
// Queremos coords bottom-left, pero "arriba a la derecha"
    const int margin = 14;
    gToneBoxW = 320;
    gToneBoxH = 120;

    gToneBoxX = winW - gToneBoxW - margin;
    gToneBoxY = 420;

    // sliders dentro
    const int padX2 = 14;
    const int padTop = 34;   // espacio para título
    const int rowGap2 = 34;  // separación vertical

    gGammaX = gToneBoxX + padX2;
    gGammaW = gToneBoxW - 2 * padX2;
    gGammaH = 14;

    gZsX = gGammaX;
    gZsW = gGammaW;
    gZsH = 14;

    // filas (gamma arriba, zscale abajo)
    gGammaY = gToneBoxY + gToneBoxH - padTop - gGammaH;  // fila 1
    gZsY = gGammaY - rowGap2;                          // fila 2


}

// ================= renderBOXSliders =================
static void renderBox(float x, float y, float w, float h, float a = 0.55f)
{


    std::vector<OverlayVert> triV, lineV;
    triV.reserve(6);
    lineV.reserve(8);

    overlayAddTri(triV, x, y, x + w, y + h, 0, 0, 0, a);

    overlayAddLine(lineV, x, y, x + w, y, 1, 1, 1, 0.25f);
    overlayAddLine(lineV, x + w, y, x + w, y + h, 1, 1, 1, 0.25f);
    overlayAddLine(lineV, x + w, y + h, x, y + h, 1, 1, 1, 0.25f);
    overlayAddLine(lineV, x, y + h, x, y, 1, 1, 1, 0.25f);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);
}

static void renderButton(int x, int y, int w, int h, bool enabled, const char* label)
{
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(6); lineV.reserve(8);

    float a = enabled ? 0.85f : 0.35f;

    overlayAddTri(triV, (float)x, (float)y, (float)(x + w), (float)(y + h), 1, 1, 1, a);

    overlayAddLine(lineV, (float)x, (float)y, (float)(x + w), (float)y, 0, 0, 0, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)y, (float)(x + w), (float)(y + h), 0, 0, 0, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)(y + h), (float)x, (float)(y + h), 0, 0, 0, 0.35f);
    overlayAddLine(lineV, (float)x, (float)(y + h), (float)x, (float)y, 0, 0, 0, 0.35f);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // Texto
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glColor3f(0, 0, 0);
    drawText(x + 10, y + 6, label);
}
static void renderSeekSliderOverlay()
{
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(64); lineV.reserve(32);

    int x = gSeekX, y = gSeekY, w = gSeekW, h = gSeekH;
    int knobW = 10;
    int kx = x + (int)std::round(gSeekT * (float)(w - knobW));

    overlayAddTri(triV, (float)x, (float)y, (float)(x + w), (float)(y + h), 0, 0, 0, 0.55f);
    overlayAddTri(triV, (float)x, (float)y, (float)(kx + knobW * 0.5f), (float)(y + h), 1, 1, 1, 0.18f);
    overlayAddTri(triV, (float)kx, (float)(y - 4), (float)(kx + knobW), (float)(y + h + 4), 1, 1, 1, 0.85f);

    overlayAddLine(lineV, (float)x, (float)y, (float)(x + w), (float)y, 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)y, (float)(x + w), (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)(y + h), (float)x, (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)x, (float)(y + h), (float)x, (float)y, 1, 1, 1, 0.35f);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // Texto
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glColor3f(1, 1, 1);

    double D = durationSec();
    double span = (double)(Wvis - 1) * (double)hop / fsSafeD();

    std::ostringstream ss;
    ss.setf(std::ios::fixed); ss.precision(2);
    ss << "Seek: " << gSeekSecPending << " s / " << D << " s  (win~" << span << " s)";
    if (gSeekDirty) ss << "  (release to apply)";
    drawText(x, y + 18, ss.str());
}
static void renderTransportOverlay()
{
    renderBox((float)gTransportBoxX, (float)gTransportBoxY, (float)gTransportBoxW, (float)gTransportBoxH);

    renderButton(gBtnPlayX, gBtnPlayY, gBtnPlayW, gBtnPlayH, true, g.paused ? "PLAY" : "PAUSE");
    renderButton(gBtnRestartX, gBtnRestartY, gBtnRestartW, gBtnRestartH, true, "RESTART");
    renderButton(gBtnStepX, gBtnStepY, gBtnStepW, gBtnStepH, g.paused, "STEP");
    renderButton(gBtnLoopX, gBtnLoopY, gBtnLoopW, gBtnLoopH, true, gLoopPlayback ? "LOOP:ON" : "LOOP:OFF");

    renderSeekSliderOverlay();
}

// ================= renderSpeedSlider =================
static void renderSpeedSliderOverlay()
{
    // Fondo+barra+knob con tu overlay shader
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(64);
    lineV.reserve(32);

    int x = gSpeedX, y = gSpeedY, w = gSpeedW, h = gSpeedH;
    int knobW = 10;
    int kx = x + (int)std::round(gSpeedT * (float)(w - knobW));



    // fondo
    overlayAddTri(triV, (float)x, (float)y, (float)(x + w), (float)(y + h), 0, 0, 0, 0.55f);

    // relleno (progreso)
    overlayAddTri(triV, (float)x, (float)y, (float)(x + gSpeedT * (float)w), (float)(y + h), 1, 1, 1, 0.18f);

    // knob
    overlayAddTri(triV, (float)kx, (float)(y - 4), (float)(kx + knobW), (float)(y + h + 4), 1, 1, 1, 0.85f);

    // borde
    overlayAddLine(lineV, (float)x, (float)y, (float)(x + w), (float)y, 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)y, (float)(x + w), (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)(y + h), (float)x, (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)x, (float)(y + h), (float)x, (float)y, 1, 1, 1, 0.35f);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // Texto (fixed pipeline)
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 1, 1);

    std::ostringstream ss;
    ss.setf(std::ios::fixed); ss.precision(2);
    double secPerCol = (double)hop / (double)g.fs;
    double winSec = (double)Wvis * secPerCol;

    ss << "speed: " << gPlaySpeed << "x"
        << " | " << secPerCol << " s/col";

    drawText(x, y + 18, ss.str());
}
static bool hitSeekSlider(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gSeekX, gSeekY, gSeekW, gSeekH); }

static bool hitBtnPlay(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gBtnPlayX, gBtnPlayY, gBtnPlayW, gBtnPlayH, 2); }
static bool hitBtnRestart(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gBtnRestartX, gBtnRestartY, gBtnRestartW, gBtnRestartH, 2); }
static bool hitBtnStep(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gBtnStepX, gBtnStepY, gBtnStepW, gBtnStepH, 2); }
static bool hitBtnLoop(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gBtnLoopX, gBtnLoopY, gBtnLoopW, gBtnLoopH, 2); }
static bool hitGammaSlider(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gGammaX, gGammaY, gGammaW, gGammaH); }
static bool hitZScaleSlider(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gZsX, gZsY, gZsW, gZsH); }

// Funnción para (re)construir y subir el VBO de ejes + grid 3D para keyboard
static void uploadAxisVBO()
{
    buildAxisBoxAndGrid();
    glBindBuffer(GL_ARRAY_BUFFER, axisVBO);
    glBufferData(GL_ARRAY_BUFFER,
        gAxisVerts.size() * sizeof(AxisVert),
        gAxisVerts.data(),
        GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
// ---------- Render fft-timezoom- apply slider ----------
static bool hitFftSlider(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gFftX, gFftY, gFftW, gFftH); }
static bool hitTimeZoomSlider(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gTimeZoomX, gTimeZoomY, gTimeZoomW, gTimeZoomH); }
static bool hitApplyButton(int mx, int myTopLeft) { return hitRectBL(mx, myTopLeft, gApplyX, gApplyY, gApplyW, gApplyH, 2); }

static void setFftFromMouseX(int mx)
{
    gFftT = sliderTFromMouseX(mx, gFftX, gFftW);
    gFftPending = fftFromT(gFftT);
    updateCalcDirty();
}

static void setTimeZoomFromMouseX(int mx)
{
    gTimeZoomT = sliderTFromMouseX(mx, gTimeZoomX, gTimeZoomW);
    gTimeZoom = timeZoomFromT(gTimeZoomT);
    updateTimeView();
    gViewDirty = true;
}
static void setGammaFromMouseX(int mx)
{
    float t = sliderTFromMouseX(mx, gGammaX, gGammaW);
    float v = sliderLogFromT(t, kGammaMin, kGammaMax);
    g.gamma = v;
    refreshVBOForBuffer(gPresent);

    // si quieres que el overlay (leyenda) se vea “coherente” al vuelo, no hace falta más
    // resetPlaybackClock();  // NO: gamma es visual, no afecta tiempo
}

static void setZScaleFromMouseX(int mx)
{
    float t = sliderTFromMouseX(mx, gZsX, gZsW);
    float v = sliderLogFromT(t, kZScaleMin, kZScaleMax);
    g.zScale = v;
    refreshVBOForBuffer(gPresent);

    // IMPORTANTÍSIMO: si cambias zScale, el cubo/grid 3D cambia => hay que re-subir axis VBO
    uploadAxisVBO();

    // Y además la nube de partículas usa zScale: para reflejarlo al instante,
    // forzamos refresh del buffer actual (solo VBOs, no STFT)
    gViewDirty = true; // (lo usas para recomputar visual)
}


static void renderToneSlider(int x, int y, int w, int h, float t, const char* label, const char* valueText)
{
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(64); lineV.reserve(32);

    int knobW = 10;
    t = clamp01f(t);
    int kx = x + (int)std::round(t * (float)(w - knobW));

    // fondo
    overlayAddTri(triV, (float)x, (float)y, (float)(x + w), (float)(y + h), 0, 0, 0, 0.55f);
    // relleno
    overlayAddTri(triV, (float)x, (float)y, (float)(kx + knobW * 0.5f), (float)(y + h), 1, 1, 1, 0.18f);
    // knob
    overlayAddTri(triV, (float)kx, (float)(y - 4), (float)(kx + knobW), (float)(y + h + 4), 1, 1, 1, 0.85f);

    // borde
    overlayAddLine(lineV, (float)x, (float)y, (float)(x + w), (float)y, 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)y, (float)(x + w), (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)(y + h), (float)x, (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)x, (float)(y + h), (float)x, (float)y, 1, 1, 1, 0.35f);

    // draw
    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // texto

    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glColor3f(1, 1, 1);

    // label arriba
    drawText(x, y + 18, std::string(label) + ": " + valueText);


}

static void renderToneOverlay()
{
    // Caja
    renderBox((float)gToneBoxX, (float)gToneBoxY, (float)gToneBoxW, (float)gToneBoxH, 0.55f);

    // Título
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glColor3f(1, 1, 1);
    drawText(gToneBoxX + 12, gToneBoxY + gToneBoxH - 18, "Tone");

    // Gamma slider
    float tGamma = tFromSliderLog(g.gamma, kGammaMin, kGammaMax);
    std::ostringstream sG; sG.setf(std::ios::fixed); sG.precision(2);
    sG << g.gamma;
    renderToneSlider(gGammaX, gGammaY, gGammaW, gGammaH, tGamma, "Gamma", sG.str().c_str());

    // ZScale slider
    float tZ = tFromSliderLog(g.zScale, kZScaleMin, kZScaleMax);
    std::ostringstream sZ; sZ.setf(std::ios::fixed); sZ.precision(3);
    sZ << g.zScale;
    renderToneSlider(gZsX, gZsY, gZsW, gZsH, tZ, "Z scale", sZ.str().c_str());
}


static void renderFftSliderOverlay()
{
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(64); lineV.reserve(32);

    int x = gFftX, y = gFftY, w = gFftW, h = gFftH;
    int knobW = 10;
    int kx = x + (int)std::round(gFftT * (float)(w - knobW));

    overlayAddTri(triV, (float)x, (float)y, (float)(x + w), (float)(y + h), 0, 0, 0, 0.55f);
    overlayAddTri(triV, (float)x, (float)y, (float)(kx + knobW * 0.5f), (float)(y + h), 1, 1, 1, 0.18f);
    overlayAddTri(triV, (float)kx, (float)(y - 4), (float)(kx + knobW), (float)(y + h + 4), 1, 1, 1, 0.85f);

    overlayAddLine(lineV, (float)x, (float)y, (float)(x + w), (float)y, 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)y, (float)(x + w), (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)(y + h), (float)x, (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)x, (float)(y + h), (float)x, (float)y, 1, 1, 1, 0.35f);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glColor3f(1, 1, 1);

    std::ostringstream ss;
    ss << "fftN: " << (gFftPending) << (gFftPending != fftSize ? " (pending)" : "");
    drawText(x, y + 18, ss.str());
    glEnable(GL_DEPTH_TEST);
}

static void renderTimeZoomSliderOverlay()
{
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(64); lineV.reserve(32);

    int x = gTimeZoomX, y = gTimeZoomY, w = gTimeZoomW, h = gTimeZoomH;
    int knobW = 10;
    int kx = x + (int)std::round(gTimeZoomT * (float)(w - knobW));

    overlayAddTri(triV, (float)x, (float)y, (float)(x + w), (float)(y + h), 0, 0, 0, 0.55f);
    overlayAddTri(triV, (float)x, (float)y, (float)(kx + knobW * 0.5f), (float)(y + h), 1, 1, 1, 0.18f);
    overlayAddTri(triV, (float)kx, (float)(y - 4), (float)(kx + knobW), (float)(y + h + 4), 1, 1, 1, 0.85f);

    overlayAddLine(lineV, (float)x, (float)y, (float)(x + w), (float)y, 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)y, (float)(x + w), (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)(y + h), (float)x, (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)x, (float)(y + h), (float)x, (float)y, 1, 1, 1, 0.35f);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // texto
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glColor3f(1, 1, 1);

    double dtCol = (double)hop / (double)(std::max)(1e-6f, g.fs);
    double spanSec = (double)gVisibleFrames * dtCol;

    std::ostringstream ss;
    ss.setf(std::ios::fixed); ss.precision(2);
    ss << "Zoom temporal: " << gTimeZoom << "x  (span=" << spanSec << " s)";
    drawText(x, y + 18, ss.str());
    glEnable(GL_DEPTH_TEST);
}

static void renderApplyButtonOverlay()
{
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(6); lineV.reserve(8);

    float a = gCalcDirty ? 0.85f : 0.35f;
    overlayAddTri(triV, (float)gApplyX, (float)gApplyY, (float)(gApplyX + gApplyW), (float)(gApplyY + gApplyH), 1, 1, 1, a);

    overlayAddLine(lineV, (float)gApplyX, (float)gApplyY, (float)(gApplyX + gApplyW), (float)gApplyY, 0, 0, 0, 0.35f);
    overlayAddLine(lineV, (float)(gApplyX + gApplyW), (float)gApplyY, (float)(gApplyX + gApplyW), (float)(gApplyY + gApplyH), 0, 0, 0, 0.35f);
    overlayAddLine(lineV, (float)(gApplyX + gApplyW), (float)(gApplyY + gApplyH), (float)gApplyX, (float)(gApplyY + gApplyH), 0, 0, 0, 0.35f);
    overlayAddLine(lineV, (float)gApplyX, (float)(gApplyY + gApplyH), (float)gApplyX, (float)gApplyY, 0, 0, 0, 0.35f);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glColor3f(0, 0, 0);

    drawText(gApplyX + 18, gApplyY + 6, "APPLY");
    glEnable(GL_DEPTH_TEST);
}


// ---------- Render Zoom slider ----------
static void renderZoomSliderOverlay()
{
    // Valor mostrado (modo B)
    // El texto de arriba debe reflejar SIEMPRE el valor actual del slider (pending)
    const bool dirty = std::fabs(gWinSecPending - gWinSecTarget) > 1e-6;
    const double winSecShown = gWinSecPending; // <- clave

    std::vector<OverlayVert> triV, lineV;
    triV.reserve(64);
    lineV.reserve(32);

    const int x = gZoomX, y = gZoomY, w = gZoomW, h = gZoomH;
    const int knobW = 10;
    const float t = (std::max)(0.0f, (std::min)(1.0f, gWinSecT));
    const int kx = x + (int)std::round(t * (float)(w - knobW));



    // fondo
    overlayAddTri(triV, (float)x, (float)y, (float)(x + w), (float)(y + h), 0, 0, 0, 0.55f);

    // progreso (hasta el centro del knob)
    overlayAddTri(triV, (float)x, (float)y, (float)(kx + knobW * 0.5f), (float)(y + h), 1, 1, 1, 0.18f);

    // knob
    overlayAddTri(triV, (float)kx, (float)(y - 4), (float)(kx + knobW), (float)(y + h + 4), 1, 1, 1, 0.85f);

    // borde
    overlayAddLine(lineV, (float)x, (float)y, (float)(x + w), (float)y, 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)y, (float)(x + w), (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)(y + h), (float)x, (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)x, (float)(y + h), (float)x, (float)y, 1, 1, 1, 0.35f);

    // draw (tri + lines)
    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // texto
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 1, 1);

    std::ostringstream ss;
    ss.setf(std::ios::fixed); ss.precision(2);
    ss << "Window: " << winSecShown << " s";
    if (dirty) ss << "  (pending)";

    drawText(x, y + 18, ss.str());

    if (dirty) {
    const int lineH = 30; // o 16 si tu fuente es más grande

    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(2);

    // Línea 1: Applied + valor
    s.str(""); s.clear();
    s << "Applied: " << gWinSecTarget << " s";
    drawText(x, y-15, s.str());

    // Línea 2: debajo
    // Si tu sistema de coordenadas Y crece hacia ARRIBA, usa (y - lineH).
    // Si tu Y crece hacia ABAJO, usa (y + lineH).
    drawText(x, y - lineH, "Press ENTER");
    drawText(x, y - lineH-12, "to Apply");
}

    glEnable(GL_DEPTH_TEST);
}

// ---------- Render Hop slider ----------
static void renderHopSliderOverlay()
{
    // Valor mostrado (modo B): si arrastras, preview; si no, hop real
    const int hopShown = gHopPending;
    const double fsSafe = (double)(std::max)(1e-6f, g.fs);
    const double dtCol = (double)hopShown / fsSafe;


    std::vector<OverlayVert> triV, lineV;
    triV.reserve(64);
    lineV.reserve(32);

    const int x = gHopX, y = gHopY, w = gHopW, h = gHopH;
    const int knobW = 10;
    const float t = (std::max)(0.0f, (std::min)(1.0f, gHopT));
    const int kx = x + (int)std::round(t * (float)(w - knobW));



    // fondo
    overlayAddTri(triV, (float)x, (float)y, (float)(x + w), (float)(y + h), 0, 0, 0, 0.55f);

    // progreso
    overlayAddTri(triV, (float)x, (float)y, (float)(kx + knobW * 0.5f), (float)(y + h), 1, 1, 1, 0.18f);

    // knob
    overlayAddTri(triV, (float)kx, (float)(y - 4), (float)(kx + knobW), (float)(y + h + 4), 1, 1, 1, 0.85f);

    // borde
    overlayAddLine(lineV, (float)x, (float)y, (float)(x + w), (float)y, 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)y, (float)(x + w), (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)(x + w), (float)(y + h), (float)x, (float)(y + h), 1, 1, 1, 0.35f);
    overlayAddLine(lineV, (float)x, (float)(y + h), (float)x, (float)y, 1, 1, 1, 0.35f);

    // draw
    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // texto
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 1, 1);

    std::ostringstream ss;
    ss.setf(std::ios::fixed); ss.precision(4);
    ss << "Hop: " << hopShown << "samples (dt=" << dtCol << "s)";
    drawText(x, y + 18, ss.str());

    glEnable(GL_DEPTH_TEST);
}




// ================= prime =================
static void primeFirstFrame() {
    gPresent = 0;
    gQueued = 0;

    int idx = 0;
    produceFrame(idx);
    CUDA_CHECK(cudaEventSynchronize(gReadyEvent[idx]));

    gPresent = idx;
    gQueued = idx;
    syncSliderFromSpeed();
    resetPlaybackClock();

}





static void renderHoverTooltip()
{
    if (!gHoverValid) return;

    int px = gMouseX + 18;
    int py = (winH - gMouseY) - 18; // a coords bottom-left

    int boxW = 190;
    int boxH = 60;

    // clamp dentro de ventana
    if (px + boxW > winW - 5) px = winW - boxW - 5;
    if (py + boxH > winH - 5) py = winH - boxH - 5;
    if (px < 5) px = 5;
    if (py < 5) py = 5;

    // Fondo + borde usando overlay shader (mismo estilo que ya tienes)
    std::vector<OverlayVert> triV, lineV;
    triV.reserve(6);
    lineV.reserve(8);

    overlayAddTri(triV, (float)px, (float)py, (float)(px + boxW), (float)(py + boxH),
        0, 0, 0, 0.75f);

    // borde
    overlayAddLine(lineV, (float)px, (float)py, (float)(px + boxW), (float)py, 1, 1, 1, 0.8f);
    overlayAddLine(lineV, (float)(px + boxW), (float)py, (float)(px + boxW), (float)(py + boxH), 1, 1, 1, 0.8f);
    overlayAddLine(lineV, (float)(px + boxW), (float)(py + boxH), (float)px, (float)(py + boxH), 1, 1, 1, 0.8f);
    overlayAddLine(lineV, (float)px, (float)(py + boxH), (float)px, (float)py, 1, 1, 1, 0.8f);

    glUseProgram(overlayProg);
    glUniform2f(uViewportLoc, (float)winW, (float)winH);
    glBindVertexArray(overlayVAO);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, triV.size() * sizeof(OverlayVert), triV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)triV.size());

    glBufferData(GL_ARRAY_BUFFER, lineV.size() * sizeof(OverlayVert), lineV.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, (GLsizei)lineV.size());

    glBindVertexArray(0);
    glUseProgram(0);

    // Texto encima
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 1, 1);

    std::ostringstream a, b;
    a.setf(std::ios::fixed); a.precision(2);
    b.setf(std::ios::fixed); b.precision(1);

    std::ostringstream tline;
    tline.setf(std::ios::fixed); tline.precision(2);


    a << "f: " << gHoverHz << " Hz";
    b << "PSD: " << gHoverDb << " dB/Hz";
    tline << "t: " << gHoverTimeSec << " s  (col " << gHoverDispCol << ")";

    drawText(px + 10, py + boxH - 18, a.str());
    drawText(px + 10, py + boxH - 34, tline.str());  // t: ... s (col ..)
    drawText(px + 10, py + 10, b.str());

    glEnable(GL_DEPTH_TEST);
}


// ================= display =================
static void display() {
    fpsFrames++;
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - fpsT0).count();
    if (dt >= 0.5) { fps = fpsFrames / dt; fpsFrames = 0; fpsT0 = t1; }

    if (gHasData[gQueued] && cudaEventQuery(gReadyEvent[gQueued]) == cudaSuccess) {
        gPresent = gQueued;
    }


    // 1) medir dt real
    auto now = std::chrono::high_resolution_clock::now();
    double dtSec = std::chrono::duration<double>(now - gLastTick).count();
    gLastTick = now;

    // (opcional) clamp dt para evitar “catch-up” enorme al arrastrar ventana, breakpoint, etc.
    dtSec = std::min(dtSec, 0.1);

    // 2) acumular samples deseados (speed=1 => tiempo real)
    if (!g.paused) {
        gAccumSamples += dtSec * (double)g.fs * (double)gPlaySpeed;

        const double blockSamples = (double)(B * hop);

        // produce como mucho 1 bloque por display (suficiente con tus límites de speed/hop)
        if (gAccumSamples >= blockSamples) {
            gAccumSamples -= blockSamples;

            int computeIdx = pickComputeBuffer();
            if (cudaEventQuery(gReadyEvent[computeIdx]) != cudaSuccess && computeIdx != gPresent) {
                CUDA_CHECK(cudaEventSynchronize(gReadyEvent[computeIdx]));
            }
            produceFrame(computeIdx);
            gQueued = computeIdx;
        }
    }
    else {
        // si está pausado, evita que al reanudar haga “catch-up”
        gAccumSamples = 0.0;
    }
    syncSeekSliderFromState();



    Mat4 MVP = buildMVP();
    updateTimeView();
    if (gViewDirty) {
        refreshVBOForBuffer(gPresent);
        uploadAxisVBO(); // opcional pero recomendado para que el grid temporal coincida
        gViewDirty = false;
    }

    updateHoverByNearestPoint(MVP);



    glViewport(0, 0, winW, winH);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 1) Cubo wireframe (no depth write) -> NO tapa el espectrograma


    // 2) Axis box + gridlines (como lo tienes)
    glUseProgram(axisProg);
    glUniformMatrix4fv(uAxisMVPLoc, 1, GL_FALSE, MVP.m);

    glBindVertexArray(axisVAO);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUniform4f(uAxisColorLoc, 0.95f, 0.95f, 0.95f, 0.95f);
    glDrawArrays(GL_LINES, 0, gAxisEdgeCount);

    if (gAxisGridCount > 0) {
        glUniform4f(uAxisColorLoc, 0.9f, 0.9f, 0.9f, 0.20f);
        glDrawArrays(GL_LINES, gAxisEdgeCount, gAxisGridCount);
    }
    glBindVertexArray(0);
    glUseProgram(0);

    // 3) Partículas (para evitar auto-occlusion rara: depth test sí, depth write NO)
    glUseProgram(program);
    if (uMvpLoc >= 0) glUniformMatrix4fv(uMvpLoc, 1, GL_FALSE, MVP.m);
    if (uPointSizeLoc >= 0) glUniform1f(uPointSizeLoc, 2.5f);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE); // <<< CLAVE: no se tapan entre sí ni las tapa el cubo

    // (opcional) un pelín de blend para “look” más bonito:
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(vao[gPresent]);
    glDrawArrays(GL_POINTS, 0, Nvis);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glUseProgram(0);

    drawHzLabelsOnCubeY(MVP);
    drawDbLabelsOnCubeZ(MVP);

    // 4) overlay 2D (como lo tienes)
    glDisable(GL_DEPTH_TEST);
    renderAxisAndLegendOverlay();
    updateUILayout();

    renderBox((float)gCalcBoxX, (float)gCalcBoxY, (float)gCalcBoxW, (float)gCalcBoxH);
    renderBox((float)gViewBoxX, (float)gViewBoxY, (float)gViewBoxW, (float)gViewBoxH);

    renderTransportOverlay();

    renderZoomSliderOverlay();      // ahora es WINSEC (calc)
    renderHopSliderOverlay();       // calc
    renderFftSliderOverlay();       // calc
    renderApplyButtonOverlay();
    renderToneOverlay();// calc

    renderSpeedSliderOverlay();     // view
    renderTimeZoomSliderOverlay();  // view
    renderHoverTooltip();
    glEnable(GL_DEPTH_TEST);


    // ---- HUD texto opcional ----
    if (g.showHUDText) {
        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, winW, 0, winH, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glColor3f(1, 1, 1);

        std::ostringstream ss;
        ss << "FPS:" << (int)fps
            << " ch=" << g.channel
            << " show=" << (g.showFiltered ? "FILTERED" : "RAW")
            << " preFilter=" << (g.preFilter ? "ON" : "OFF")
            << " HP=" << (g.useHighpass ? "ON" : "OFF")
            << " fs=" << g.fs
            << " fTop=" << fTopShownHz
            << " fft=" << fftSize
            << " hop=" << hop
            << " W=" << Wvis << " H=" << H
            << " autoGain=" << (g.autoGain ? "ON" : "OFF")
            << " gamma=" << g.gamma
            << " zScale=" << g.zScale
            << " speed=" << gPlaySpeed
            << (g.paused ? " PAUSED" : "");
        drawText(10, winH - 20, ss.str());

        std::ostringstream ssF;
        ssF << "Filter RMS raw=" << gRmsRaw
            << " filt=" << gRmsFilt
            << " removed=" << gRmsRemoved
            << " (" << (int)std::round(gRemovedPct) << "%)";
        drawText(10, winH - 40, ssF.str());

        drawText(10, winH - 60,
            "P pausa | A autogain | C cmap | G/H gamma | [ ] hop | F fft | 1..9 canal | T HUD | X filtro | V raw/filt | Y HP | -/+ speed");
        glEnable(GL_DEPTH_TEST);
    }

    glutSwapBuffers();
}


static void reshape(int w, int h)
{
    winW = w; winH = h;
    updateUILayout();                 // <<< CLAVE
    if (winW > 0 && winH > 0) createPicking(winW, winH);
}


static void idle() { glutPostRedisplay(); }

// ================= cleanup / rebuild =================
static void cleanupAll() {
    // CUDA interop unregister
    for (int i = 0; i < kBuffers; ++i) {
        if (cudaPosRes[i]) CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPosRes[i]));
        if (cudaColRes[i]) CUDA_CHECK(cudaGraphicsUnregisterResource(cudaColRes[i]));
        cudaPosRes[i] = cudaColRes[i] = nullptr;
    }

    // GL particles
    for (int i = 0; i < kBuffers; ++i) {
        if (vao[i]) glDeleteVertexArrays(1, &vao[i]);
        vao[i] = 0;
        if (vboPos[i]) glDeleteBuffers(1, &vboPos[i]);
        if (vboCol[i]) glDeleteBuffers(1, &vboCol[i]);
        vboPos[i] = vboCol[i] = 0;
    }
    if (program) glDeleteProgram(program);
    program = 0;




    // Axis
    if (axisVBO) glDeleteBuffers(1, &axisVBO);
    axisVBO = 0;
    if (axisVAO) glDeleteVertexArrays(1, &axisVAO);
    axisVAO = 0;
    if (axisProg) glDeleteProgram(axisProg);
    axisProg = 0;

    // Overlay
    if (overlayVBO) glDeleteBuffers(1, &overlayVBO);
    overlayVBO = 0;
    if (overlayVAO) glDeleteVertexArrays(1, &overlayVAO);
    overlayVAO = 0;
    if (overlayProg) glDeleteProgram(overlayProg);
    overlayProg = 0;

    // cuFFT
    if (plan) { CUFFT_CHECK(cufftDestroy(plan)); plan = 0; }

    // CUDA mem
    if (d_signal) CUDA_CHECK(cudaFree(d_signal));
    if (d_win)    CUDA_CHECK(cudaFree(d_win));
    d_signal = d_win = nullptr;

    for (int i = 0; i < kBuffers; ++i) {
        if (d_frames[i])  CUDA_CHECK(cudaFree(d_frames[i]));
        if (d_fft[i])     CUDA_CHECK(cudaFree(d_fft[i]));
        if (d_spec_db[i]) CUDA_CHECK(cudaFree(d_spec_db[i]));
        if (d_minmax[i])  CUDA_CHECK(cudaFree(d_minmax[i]));
        d_frames[i] = nullptr; d_fft[i] = nullptr; d_spec_db[i] = nullptr; d_minmax[i] = nullptr;
    }

    if (d_scale) CUDA_CHECK(cudaFree(d_scale));
    d_scale = nullptr;

    if (d_hist_db) CUDA_CHECK(cudaFree(d_hist_db));
    d_hist_db = nullptr;



    // events/stream
    for (int i = 0; i < kBuffers; ++i) {
        if (gReadyEvent[i]) { CUDA_CHECK(cudaEventDestroy(gReadyEvent[i])); gReadyEvent[i] = nullptr; }
    }
    if (gStream) { CUDA_CHECK(cudaStreamDestroy(gStream)); gStream = nullptr; }

    if (gScaleEvent) { CUDA_CHECK(cudaEventDestroy(gScaleEvent)); gScaleEvent = nullptr; }
    if (h_scalePinned) { CUDA_CHECK(cudaFreeHost(h_scalePinned)); h_scalePinned = nullptr; }
    if (gHoverEvent) { CUDA_CHECK(cudaEventDestroy(gHoverEvent)); gHoverEvent = nullptr; }
    if (h_hoverPinned) { CUDA_CHECK(cudaFreeHost(h_hoverPinned)); h_hoverPinned = nullptr; }

    //destroyPicking
    destroyPicking();
}

static void rebuildAll() {
    cleanupAll();
    initPipeline();
    initGLandInterop();
    syncTimeSlidersFromState();
    resetPlaybackClock();
    updateUILayout();
    primeFirstFrame();
}

static void applyCalcAndRebuild()
{
    // Aplica pending -> aplicado
    gWinSecTarget = clampd(gWinSecPending, kWinSecMin, kWinSecMax);
    hop = snapHop(gHopPending);
    fftSize = gFftPending;

    // Wvis depende científicamente de winSec/hop/fs
    Wvis = computeWvis(gWinSecTarget, hop);

    {
        double fsSafe = (double)(std::max)(1e-6f, g.fs);
        gWinSecTarget = (double)Wvis * (double)hop / fsSafe;
    }


    // Rebuild total (robusto): cambia buffers, bins, cuFFT plan, etc.
    rebuildAll(); // esto resetea espectrograma (correcto)

    // Después del rebuild, deja UI sincronizada y sin “dirty”
    syncTimeSlidersFromState();
    resetPlaybackClock();
}


static void transportTogglePlay()
{
    g.paused = !g.paused;
    gAccumSamples = 0.0;
    resetPlaybackClock();
}

static void transportRestart(bool resume)
{
    // ir al inicio con ventana "llena"
    prefillWindowAtStartSample(0);
    g.paused = !resume;
    gAccumSamples = 0.0;
    resetPlaybackClock();
}

static void transportStepOnce()
{
    if (!g.paused) return;

    int computeIdx = pickComputeBuffer();
    if (cudaEventQuery(gReadyEvent[computeIdx]) != cudaSuccess && computeIdx != gPresent)
        CUDA_CHECK(cudaEventSynchronize(gReadyEvent[computeIdx]));

    produceFrame(computeIdx);
    gQueued = computeIdx;

    gAccumSamples = 0.0;
    resetPlaybackClock();
}




static void recomputeMaxStartSample()
{
    maxStartSample = nSamples - ((B - 1) * hop + fftSize);
    if (maxStartSample < 0) maxStartSample = 0;
    if (startSample > maxStartSample) startSample = 0;
}

static void uploadCurrentSignalToGPU(bool resetScroll = true)
{
    h_signal = g.showFiltered ? h_signal_filt : h_signal_raw;

    CUDA_CHECK(cudaMemcpyAsync(d_signal,
        h_signal.data(),
        (size_t)nSamples * sizeof(float),
        cudaMemcpyHostToDevice,
        gStream));
    CUDA_CHECK(cudaStreamSynchronize(gStream));

    if (resetScroll) clearHistAndResetScroll();
}

static void rebuildSignalOnly(bool resetScroll = true)
{
    g.channel = (std::max)(0, (std::min)(g.channel, (int)h_channels.size() - 1));
    h_signal_raw = h_channels[g.channel];
    nSamples = (int)h_signal_raw.size();

    h_signal_filt = h_signal_raw;

    if (g.preFilter) preprocess_filter_inplace(h_signal_filt, g.fs, g.removeMean, g.useHighpass, g.highHz, g.lowHz);
    else if (g.removeMean) {
        double m = 0.0;
        for (float v : h_signal_filt) m += (double)v;
        m /= (double)h_signal_filt.size();
        for (float& v : h_signal_filt) v = (float)(v - m);
    }

    auto rms = [](const std::vector<float>& v) -> float {
        double s2 = 0.0;
        for (float x : v) s2 += (double)x * (double)x;
        return (float)std::sqrt(s2 / (double)((std::max)(size_t(1), v.size())));
        };
    gRmsRaw = rms(h_signal_raw);
    gRmsFilt = rms(h_signal_filt);

    double s2 = 0.0;
    for (size_t i = 0; i < h_signal_raw.size(); ++i) {
        double d = (double)h_signal_raw[i] - (double)h_signal_filt[i];
        s2 += d * d;
    }
    gRmsRemoved = (float)std::sqrt(s2 / (double)((std::max)(size_t(1), h_signal_raw.size())));
    gRemovedPct = (gRmsRaw > 1e-12f) ? (100.0f * gRmsRemoved / gRmsRaw) : 0.0f;

    static int lastNSamples = -1;
    if (lastNSamples < 0) lastNSamples = nSamples;
    if (nSamples != lastNSamples) {
        rebuildAll();
        return;
    }

    recomputeMaxStartSample();
    uploadCurrentSignalToGPU(resetScroll);
}



static void applyZoomSec(double newWinSec)
{
    gWinSecTarget = clampd(newWinSec, kWinSecMin, kWinSecMax);

    int newW = computeWvis(gWinSecTarget, hop);
    if (newW != Wvis) {
        Wvis = newW;
        rebuildAll();                  // porque Wvis cambia buffers
    }
    else {
        recomputeMaxStartSample();
        clearHistAndResetScroll();
    }

    resetPlaybackClock();
    syncTimeSlidersFromState();
    syncTimeSlidersFromState();

}


static void applyHop_ModeB(int newHop)
{
    newHop = snapHop(newHop);
    if (newHop == hop) return;

    hop = newHop;

    int newW = computeWvis(gWinSecTarget, hop);
    if (newW != Wvis) {
        Wvis = newW;
        rebuildAll();
    }
    else {
        recomputeMaxStartSample();
        clearHistAndResetScroll();
    }

    resetPlaybackClock();
    syncTimeSlidersFromState();
}

// ================= mouse camera =================
// ================= mouse camera =================
static void motion(int x, int y)
{
    gMouseX = x; gMouseY = y;
    updateUILayout(); // rects siempre correctos

    // si arrastras Tone, consume y no muevas cámara
    if (gToneDragging) {
        if (gToneActive == TONE_GAMMA)  setGammaFromMouseX(x);
        if (gToneActive == TONE_ZSCALE) setZScaleFromMouseX(x);
        return;
    }
    if (gDragSeek) { setSeekFromMouseX(x); return; }


    // --- UI drags (consumen el motion) ---
    if (gDragSpeed) { setSpeedFromMouseX(x); return; }
    if (gDragZoom) { setZoomFromMouseX(x);  return; }
    if (gDragHop) { setHopFromMouseX(x);   return; }
    if (gDragFFT) { setFftFromMouseX(x);   return; }
    if (gDragTimeZoom) { setTimeZoomFromMouseX(x); return; }

    // --- CAMERA drags ---
    int dx = x - gLastMX;
    int dy = y - gLastMY;
    gLastMX = x;
    gLastMY = y;

    // Sensibilidades
    const float orbitSpeed = 0.005f;   // rad/pixel aprox
    const float fovyRad = 55.0f * (float)M_PI / 180.0f;

    if (gMouseL) {
        // Orbit (yaw/pitch)
        gYaw += (float)dx * orbitSpeed;
        gPitch += (float)(-dy) * orbitSpeed;

        // Clamp pitch para que no “flippee”
        const float pitchMax = 1.55f; // ~ 89 deg
        gPitch = (std::max)(-pitchMax, (std::min)(pitchMax, gPitch));
    }
    else if (gMouseR) {
        // Pan (mueve el target en el plano de cámara)
        float cp = std::cos(gPitch), sp = std::sin(gPitch);
        float cy = std::cos(gYaw), sy = std::sin(gYaw);

        Vec3 eye = gTarget + Vec3(
            gDist * cp * sy,
            gDist * sp,
            gDist * cp * cy
        );

        Vec3 fwd = normalize(gTarget - eye);
        Vec3 right = normalize(cross(fwd, Vec3(0, 1, 0)));
        Vec3 up = normalize(cross(right, fwd));

        // Escala de pan: “world units por pixel” a la distancia gDist
        float panScale = (2.0f * gDist * std::tan(0.5f * fovyRad)) / (float)(std::max)(1, winH);

        // Signos: ajusta si lo prefieres invertido
        gTarget = gTarget + right * (-(float)dx * panScale) + up * ((float)dy * panScale);
    }
}

static void onMouseEntry(int state)
{
    // state: GLUT_ENTERED o GLUT_LEFT
    gMouseInside = (state == GLUT_ENTERED);

    if (!gMouseInside) {
        cancelAllUIDrags();
        gHoverValid = false;
    }
}

static void mouseButton(int button, int state, int x, int y)
{
    gMouseX = x; gMouseY = y;
    updateUILayout(); // <<< CLAVE

    // ----- rueda: zoom cámara -----
    if (button == 3 && state == GLUT_DOWN) { gDist *= 0.92f; gDist = (std::max)(0.15f, (std::min)(50.0f, gDist)); return; }
    if (button == 4 && state == GLUT_DOWN) { gDist *= 1.08f; gDist = (std::max)(0.15f, (std::min)(50.0f, gDist)); return; }    // wheel up/down


    if (button != GLUT_LEFT_BUTTON && button != GLUT_RIGHT_BUTTON) return;

    // ================== UI FIRST ==================
    if (button == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_DOWN)


        {

            // --- Tone sliders ---
            if (hitGammaSlider(x, y)) {
                gToneActive = TONE_GAMMA;
                gToneDragging = true;
                gUIDragging = true;
                setGammaFromMouseX(x);
                return;
            }
            if (hitZScaleSlider(x, y)) {
                gToneActive = TONE_ZSCALE;
                gToneDragging = true;
                gUIDragging = true;
                setZScaleFromMouseX(x);
                return;
            }
            // --- Transport UI primero ---
            if (hitSeekSlider(x, y)) {
                gUIDragging = true;
                gDragSeek = true;
                gSeekWasPaused = g.paused;
                g.paused = true;           // pausa mientras scrubeas
                setSeekFromMouseX(x);
                return;
            }
            if (hitBtnPlay(x, y)) { transportTogglePlay(); return; }
            if (hitBtnRestart(x, y)) { transportRestart(true); return; }
            if (hitBtnStep(x, y)) { transportStepOnce(); return; }
            if (hitBtnLoop(x, y)) { gLoopPlayback = !gLoopPlayback; return; }


            // Click APPLY
            if (hitApplyButton(x, y)) {
                gPressApply = true;
                return;
            }

            // Calc sliders
            if (hitZoomSlider(x, y)) {
                gDragZoom = true;
                setZoomFromMouseX(x);
                return;
            }
            if (hitHopSlider(x, y)) {
                gDragHop = true;
                setHopFromMouseX(x);
                return;
            }
            if (hitFftSlider(x, y)) {
                gDragFFT = true;
                setFftFromMouseX(x);
                return;
            }

            // View sliders
            if (hitRectBL(x, y, gSpeedX, gSpeedY, gSpeedW, gSpeedH)) {
                gDragSpeed = true;
                setSpeedFromMouseX(x);
                return;
            }
            if (hitTimeZoomSlider(x, y)) {
                gDragTimeZoom = true;
                setTimeZoomFromMouseX(x);
                return;
            }
        }
        else // GLUT_UP
        {

            // suelta Tone
            if (gToneDragging) {
                gToneDragging = false;
                gToneActive = TONE_NONE;
                gUIDragging = false;
                return;
            } gMouseL = false;

            // --- soltar seek drag ---
            if (gDragSeek) {
                gDragSeek = false;
                gUIDragging = false;

                if (gSeekDirty) {
                    bool resume = !gSeekWasPaused;
                    transportSeekToRightEdgeSec(gSeekSecPending, resume);
                }
                else {
                    g.paused = gSeekWasPaused;
                }
                gSeekDirty = false;
                return;
            }

            // Si soltaste encima del botón y estaba presionado -> APPLY
            if (gPressApply && hitApplyButton(x, y)) {
                if (gCalcDirty) applyCalcAndRebuild();
            }
            gPressApply = false;

            gDragZoom = gDragHop = gDragFFT = false;
            gDragSpeed = gDragTimeZoom = false;
        }
    }

    // ================== CAMERA (si no consumió UI) ==================
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) { gMouseL = true; gLastMX = x; gLastMY = y; }
        else { gMouseL = false; }
    }
    if (button == GLUT_RIGHT_BUTTON) {
        if (state == GLUT_DOWN) { gMouseR = true; gLastMX = x; gLastMY = y; }
        else { gMouseR = false; }
    }
}





// ================= keyboard =================
static void keyboard(unsigned char key, int, int) {
    if (key == 27) std::exit(0);

    if (key == 'p' || key == 'P') {
        g.paused = !g.paused;
        resetPlaybackClock(); // evita dt grande al volver
    }
    if (key == 't' || key == 'T') g.showHUDText = !g.showHUDText;

    if (key == 'a' || key == 'A') g.autoGain = !g.autoGain;
    if (key == 'c' || key == 'C') g.cmap = (g.cmap + 1) % 2;

    if (key == 'g' || key == 'G') g.gamma = (std::max)(0.1f, g.gamma - 0.1f);
    if (key == 'h' || key == 'H') g.gamma = (std::min)(5.0f, g.gamma + 0.1f);

    if (key == 'j' || key == 'J') g.jitter = (std::max)(0.0f, g.jitter - 0.01f);
    if (key == 'k' || key == 'K') g.jitter = (std::min)(0.25f, g.jitter + 0.01f);

    // speed
    if (key == '-') {
        gPlaySpeed = (float)std::max((float)gSpeedMin, gPlaySpeed * 0.8f);
        syncSliderFromSpeed();   // <-- mueve el knob
        resetPlaybackClock();    // <-- evita catch-up
    }
    if (key == '+') {
        gPlaySpeed = (float)std::min((float)gSpeedMax, gPlaySpeed * 1.25f);
        syncSliderFromSpeed();   // <-- mueve el knob
        resetPlaybackClock();
    }


    // hop (no rebuild)
    if (key == '[') {
        gHopPending = snapHop(gHopPending - kHopStep);
        gHopT = tFromHop(gHopPending);
        updateCalcDirty();
    }
    if (key == ']') {
        gHopPending = snapHop(gHopPending + kHopStep);
        gHopT = tFromHop(gHopPending);
        updateCalcDirty();
    }


    // fftSize requires rebuild
    if (key == 'f' || key == 'F') {
        // Cicla pending FFT (sin rebuild)
        int idx = 0;
        for (int i = 0; i < kFftOptCount; ++i) if (kFftOptions[i] == gFftPending) idx = i;
        gFftPending = kFftOptions[(idx + 1) % kFftOptCount];
        gFftT = tFromFft(gFftPending);
        updateCalcDirty();
    }

    if (key == 13 /*Enter*/ && gCalcDirty) {
        applyCalcAndRebuild();
        return;
    }



    // filter / view / HP (no rebuild, signal-only)
    if (key == 'x' || key == 'X') { g.preFilter = !g.preFilter; rebuildSignalOnly(true); }
    if (key == 'v' || key == 'V') { g.showFiltered = !g.showFiltered; uploadCurrentSignalToGPU(true); }
    if (key == 'y' || key == 'Y') { g.useHighpass = !g.useHighpass; rebuildSignalOnly(true); }

    // 3D grid toggle
    if (key == 'm' || key == 'M') {
        g.show3DGrid = !g.show3DGrid;
        uploadAxisVBO();
    }

    // zScale tweak (rebuild axis VBO only)
    if (key == 'z') {
        g.zScale = (std::max)(0.2f, g.zScale - 0.05f);
        uploadAxisVBO();
    }
    if (key == 'Z') {
        g.zScale = (std::min)(2.5f, g.zScale + 0.05f);
        uploadAxisVBO();
    }

    // time grid spacing tweak
    if (key == 'n') {
        g.timeGridEveryCols = (std::max)(4, g.timeGridEveryCols - 4);
        uploadAxisVBO();
    }
    if (key == 'N') {
        g.timeGridEveryCols = (std::min)(128, g.timeGridEveryCols + 4);
        uploadAxisVBO();
    }

    // channel
    if (key >= '1' && key <= '9') {
        int ch = (key - '0');
        if (ch >= 0 && ch < (int)h_channels.size()) {
            g.channel = ch;
            rebuildSignalOnly(true);
        }
        else {
            std::cout << "Canal " << ch << " no existe.\n";
        }
    }
}

// ================= main =================
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitContextVersion(3, 3);
    glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(winW, winH);

    glutCreateWindow("EEG Spectrogram Particles 3D PRO (CUDA+cuFFT+OpenGL) - Overlay kept + 3D box/grid");

    // Parse CLI args (--csv, --fs) to override defaults. Must be done before initPipeline().
    parseArgs(argc, argv);

    initPipeline();
    initGLandInterop();
    primeFirstFrame();

    updateUILayout();
    syncSeekSliderFromState();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouseButton);
    glutEntryFunc(onMouseEntry);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(passiveMotion);

    glutMainLoop();
    return 0;
}
