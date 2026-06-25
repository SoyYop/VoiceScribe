# VoiceScribe

VoiceScribe es un prototipo de transcripción de voz en tiempo real para Windows, escrito en C# y basado en el modelo **NVIDIA Nemotron 3.5 ASR Streaming 0.6B** exportado a ONNX.

La aplicación captura audio desde un micrófono, genera características log-mel localmente y ejecuta un pipeline RNN-T compuesto por encoder, decoder y joint. La inferencia se realiza en la máquina local mediante ONNX Runtime; la conexión a Internet solo es necesaria para descargar los archivos del modelo si todavía no existen.

## Estado actual

- Compila correctamente con .NET 10.
- Captura audio PCM mono a 16 kHz y 16 bits.
- Procesa bloques de 560 ms, equivalentes a 8960 muestras.
- Mantiene cachés acústicas y de características entre bloques.
- Decodifica tokens de forma greedy mediante el pipeline RNN-T.
- Escribe la transcripción en consola y, opcionalmente, en un archivo.
- Descarga automáticamente los archivos faltantes desde Hugging Face, previa confirmación.
- No incluye actualmente pruebas automatizadas.

## Requisitos

- Windows.
- [.NET 10 SDK](https://dotnet.microsoft.com/download).
- Un micrófono reconocido por Windows.
- Conexión a Internet durante la primera descarga del modelo.
- Espacio en disco suficiente para los archivos ONNX y sus datos externos.

La implementación actual utiliza el proveedor de ejecución CPU de `Microsoft.ML.OnnxRuntime`. No se configura CUDA, DirectML ni otro acelerador.

## Inicio rápido

Clonar el repositorio y restaurar las dependencias:

```powershell
git clone <URL_DEL_REPOSITORIO>
cd VoiceScribe
dotnet restore
```

Revisar `src/VoiceScribe.Console/VoiceAppConfig.json` y cambiar `ModelDownloadsPath`. El valor incluido en el repositorio es una ruta absoluta de la máquina de desarrollo y no será válido en otras instalaciones.

Ejemplo:

```json
{
  "ModelDownloadsPath": "C:/Modelos/nemotron-3.5-asr",
  "RepoUrl": "https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/main",
  "Audio": {
    "SampleRate": 16000,
    "BitsPerSample": 16,
    "Channels": 1,
    "BufferMilliseconds": 560,
    "SilenceThreshold": 0.003,
    "ShutdownDrainMilliseconds": 300
  },
  "Nemotron": {
    "LanguageId": 101,
    "MaxSymbolsPerStep": 10,
    "BlankId": null
  }
}
```

`ModelFiles` es opcional. Si se omite, se utiliza la lista estándar centralizada en `NemotronModelFiles`. Puede declararse explícitamente para trabajar con una distribución de archivos diferente.

Ejecutar la aplicación:

```powershell
dotnet run --project src/VoiceScribe.Console
```

Si faltan archivos del modelo, la aplicación solicitará autorización para descargarlos:

```text
[Missing Assets] Nemotron model layers missing. Download? (y/n):
```

Una vez cargado el modelo:

1. Selecciona el micrófono si se detecta más de uno.
2. Habla con claridad.
3. Presiona `Enter` para detener la captura.

## Guardar la transcripción

El primer argumento de la aplicación se interpreta como el archivo de salida. El contenido se agrega al final si el archivo ya existe.

```powershell
dotnet run --project src/VoiceScribe.Console -- transcripcion.txt
```

La salida se escribe progresivamente y se vacía después de procesar cada bloque de audio.

## Compilación

```powershell
dotnet build VoiceScribe.sln
```

La solución contiene dos proyectos:

| Proyecto | Responsabilidad |
| --- | --- |
| `VoiceScribe.Console` | Inicio de la aplicación, logging, selección de micrófono, captura de audio, configuración y descarga del modelo. |
| `VoiceScribe.Core` | Extracción de características, administración de modelos ONNX, cachés de streaming y decodificación RNN-T. |

## Arquitectura

```text
Micrófono
   |
   | PCM 16 kHz / 16 bits / mono
   v
NAudio WaveInEvent
   |
   | bloques de 560 ms
   v
AudioFeatureExtractor
   |
   | log-mel [1, 65, 128]
   v
Encoder ONNX + cachés acústicas
   |
   | representaciones [1, T, D]
   v
Decoder ONNX + estados LSTM
   |
   v
Joint ONNX -> ArgMax
   |
   v
Tokenizer -> consola / archivo
```

### Captura y preprocesamiento

`Program.cs` configura `WaveInEvent` con:

- 16 000 muestras por segundo.
- 16 bits por muestra.
- Un canal.
- Búferes de 560 ms.

`AudioFeatureExtractor` aplica:

- Normalización de la longitud del bloque.
- Preénfasis y dithering opcional.
- Ventana Hann.
- FFT de 512 puntos implementada en el proyecto.
- Banco de 128 filtros mel con normalización de estilo Slaney.
- Conversión a energía log-mel.
- Caché de nueve frames previos.

Con la configuración estándar, cada bloque produce 56 frames actuales más nueve frames almacenados: un tensor de entrada de forma `[1, 65, 128]`.

### Inferencia ONNX

`NemotronEngine` crea tres sesiones de ONNX Runtime:

- `encoder.onnx`: procesa las características acústicas y actualiza las cachés de streaming.
- `decoder.onnx`: mantiene el estado lingüístico recurrente.
- `joint.onnx`: combina las salidas del encoder y decoder para obtener los logits de tokens.

La decodificación es greedy: se selecciona el token con mayor logit y el límite de símbolos por frame es configurable. Si `Nemotron.BlankId` es `null`, el motor utiliza la última clase de la salida del joint como token blank.

Las dimensiones de las cachés acústicas y de los estados recurrentes se obtienen desde `InputMetadata` de las sesiones ONNX. Las dimensiones de los embeddings del encoder y decoder se obtienen de los tensores producidos durante la inferencia, evitando fijar valores como `1024` o `640` en el código.

### Tokenización

El vocabulario se lee desde `tokenizer.json`, bajo `model.vocab`. El cargador admite elementos representados como strings, arreglos u objetos con una propiedad `piece`.

Antes de emitir cada token:

- Se descartan tokens especiales que comienzan con `<` o `[`.
- `▁` se convierte en un espacio.
- Se elimina el prefijo `##`.

## Configuración

El archivo `VoiceAppConfig.json` se copia al directorio de salida al compilar.

| Propiedad | Descripción |
| --- | --- |
| `ModelDownloadsPath` | Directorio local que contiene o recibirá los archivos del modelo. |
| `RepoUrl` | URL base utilizada para descargar cada archivo. |
| `ModelFiles` | Lista de archivos obligatorios. Si se omite, se usa la lista estándar de Nemotron. |
| `Audio.SampleRate` | Frecuencia de captura; valor predeterminado: 16 000 Hz. |
| `Audio.BitsPerSample` | Profundidad PCM: 8, 16, 24 o 32 bits. |
| `Audio.Channels` | Número de canales capturados. El extractor procesa el primer canal. |
| `Audio.BufferMilliseconds` | Duración de cada bloque de captura. |
| `Audio.SilenceThreshold` | Pico mínimo normalizado para procesar un bloque. |
| `Audio.ShutdownDrainMilliseconds` | Espera final para drenar el procesamiento pendiente. |
| `Nemotron.LanguageId` | Identificador de idioma enviado al encoder; valor predeterminado: `101`. |
| `Nemotron.MaxSymbolsPerStep` | Máximo de tokens no blank permitidos por frame acústico. |
| `Nemotron.BlankId` | Sobrescritura opcional del token blank. Con `null`, se usa la última clase del joint. |

Si el archivo no existe o no puede deserializarse, se usa la configuración predeterminada definida en `Program.cs`. La verificación de los modelos comprueba únicamente que cada archivo exista; no valida tamaño, hash ni integridad.

## Estructura del repositorio

```text
VoiceScribe/
├── VoiceScribe.sln
├── src/
│   ├── VoiceScribe.Console/
│   │   ├── Program.cs
│   │   ├── VoiceAppConfig.json
│   │   └── VoiceScribe.Console.csproj
│   └── VoiceScribe.Core/
│       ├── Audio/
│       │   └── AudioFeatureExtractor.cs
│       ├── Configuration/
│       │   └── VoiceAppConfig.cs
│       ├── Engine/
│       │   └── NemotronEngine.cs
│       ├── ModelAssets/
│       │   └── ModelDownloader.cs
│       └── VoiceScribe.Core.csproj
└── artifacts/
    └── models/                 # Ignorado por Git
```

## Dependencias principales

| Paquete | Uso |
| --- | --- |
| `Microsoft.ML.OnnxRuntime` | Ejecución de los grafos ONNX. |
| `NAudio` | Enumeración de dispositivos y captura de audio en Windows. |
| `Microsoft.Extensions.Logging` | Registro de eventos y errores. |

Las versiones exactas se encuentran en los archivos `.csproj`.

## Códigos de salida

| Código | Motivo |
| ---: | --- |
| `1` | No fue posible cargar una configuración válida. |
| `2` | La lista de archivos del modelo está vacía. |
| `3` | La ruta de descarga del modelo no está configurada. |
| `4` | Faltan archivos del modelo y no pudieron descargarse. |
| `10` | No se detectó ningún dispositivo de entrada. |

## Limitaciones conocidas

- Solo funciona en Windows porque los proyectos apuntan a `net10.0-windows` y la captura usa NAudio.
- El procesamiento de inferencia ocurre directamente en el callback de audio; una inferencia más lenta que la captura puede provocar latencia o pérdida de bloques.
- La captura calcula las muestras por bloque desde `SampleRate` y `BufferMilliseconds`, pero la exportación estándar del modelo espera bloques de 8960 muestras; otras combinaciones son normalizadas por el extractor y requieren validación específica.
- El modelo sigue dependiendo de los nombres de nodos ONNX de esta exportación concreta, aunque las dimensiones de cachés, estados y embeddings se derivan en tiempo de ejecución.
- Se omiten bloques cuyo pico de amplitud sea inferior a `0.003`.
- No hay beam search, timestamps, separación por hablante, puntuación de confianza ni segmentación formal de frases.
- El nivel de logging predeterminado es `Warning`; los diagnósticos `Information` y `Debug` no aparecen sin modificar la configuración del logger.
- No existe todavía una suite de pruebas automatizadas ni validación comparativa del preprocesamiento frente a la implementación original del modelo.

## Solución de problemas

### No se encuentra el modelo

Comprueba que `ModelDownloadsPath` apunte al directorio correcto y que contenga todos los archivos declarados en `ModelFiles`.

### La descarga falla

Verifica la conexión, el acceso a Hugging Face y la URL configurada. Las descargas parciales no se validan automáticamente; elimina manualmente un archivo incompleto antes de reintentar.

### No aparece ningún micrófono

Comprueba que Windows reconozca el dispositivo y que la aplicación tenga permiso para acceder al micrófono.

### No se muestra transcripción

Comprueba:

- Que el micrófono seleccionado sea el correcto.
- Que la señal supere el umbral de silencio.
- Que los archivos correspondan exactamente a la exportación esperada.
- Que las formas y nombres de entrada/salida ONNX coincidan con los usados por `NemotronEngine`.

Para inspeccionar más información, cambia temporalmente el nivel mínimo de logging de `Warning` a `Information` o `Debug` en `Program.cs`.

## Licencia y modelo

Este repositorio no contiene actualmente un archivo de licencia. Antes de redistribuir el código o los pesos, añade una licencia explícita y revisa por separado los términos del modelo y de sus archivos publicados en Hugging Face.
