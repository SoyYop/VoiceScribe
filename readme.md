# VoiceScribe

VoiceScribe es un prototipo de transcripción de voz en tiempo real para Windows, escrito en C# y basado en el modelo **NVIDIA Nemotron 3.5 ASR Streaming 0.6B** exportado a ONNX.

La aplicación captura audio desde un micrófono, genera características log-mel localmente y ejecuta un pipeline RNN-T compuesto por encoder, decoder y joint. La inferencia se realiza en la máquina local mediante ONNX Runtime; la conexión a Internet solo es necesaria para descargar los archivos del modelo si todavía no existen.

Las invariantes que deben preservarse al modificar audio, tensores ONNX, concurrencia o decodificación están documentadas en [constraints.md](constraints.md). Debe revisarse antes de realizar cambios en esas áreas.

## Estado actual

- Compila correctamente con .NET 10.
- Captura audio PCM mono a 16 kHz y 16 bits.
- Procesa bloques de 560 ms, equivalentes a 8960 muestras.
- Mantiene cachés acústicas y de características entre bloques.
- Decodifica tokens de forma greedy mediante el pipeline RNN-T.
- Escribe la transcripción en consola y, opcionalmente, en un archivo.
- Descarga automáticamente los archivos faltantes desde Hugging Face, previa confirmación.
- Incluye pruebas automatizadas para contrato del modelo, configuración, formas de estado ONNX y extracción de características.

## Requisitos

- Windows.
- [.NET 10 SDK](https://dotnet.microsoft.com/download).
- Un micrófono reconocido por Windows.
- Conexión a Internet durante la primera descarga del modelo.
- Espacio en disco suficiente para los archivos ONNX y sus datos externos.

La implementación actual usa WindowsML como único runtime ONNX. Dentro de ese runtime se puede seleccionar `Cpu` o `DirectMl` como proveedor de ejecución mediante `VoiceAppConfig.json`. WindowsML incluye el proveedor CPU, por lo que el fallback no requiere un paquete CPU separado.

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
    "QueueCapacity": 8
  },
  "Inference": {
    "ExecutionProvider": "DirectMl",
    "EncoderProvider": null,
    "DecoderProvider": null,
    "JoinerProvider": null,
    "DeviceId": 0,
    "GpuMemoryLimitMiB": null,
    "AllowCpuFallback": true,
    "EnableProfiling": false,
    "LogSeverityLevel": "Verbose",
    "LogVerbosityLevel": 5
  },
  "Nemotron": {
    "LanguageId": 101,
    "MaxSymbolsPerStep": null,
    "BlankId": null
  }
}
```

`ModelFiles` es opcional. Si se omite, se utiliza la lista estándar centralizada en `NemotronModelFiles`. Puede declararse explícitamente para trabajar con una distribución de archivos diferente.

Ejecutar la aplicación:

```powershell
dotnet run --project src/VoiceScribe.Console
```

Para usar aceleración por GPU se debe establecer `"ExecutionProvider": "DirectMl"` en `VoiceAppConfig.json`. Para forzar CPU, usar `"ExecutionProvider": "Cpu"`. El runtime subyacente en ambos casos es `Microsoft.Windows.AI.MachineLearning` 2.1.71 y el proyecto compila con `net10.0-windows10.0.18362.0`, mínimo requerido por ese paquete.

Con la exportación INT4 actual, decoder y joint crean sesiones DirectML, pero el encoder falla al inicializar el proveedor. Con `AllowCpuFallback: true`, el encoder se recrea explícitamente en CPU y se registra una advertencia. Por ello, esta modalidad es actualmente híbrida.

Si hay más de un adaptador compatible con DirectML, la consola muestra una selección antes de cargar los modelos. El número elegido se guarda en memoria como `Inference.DeviceId` para esa ejecución.

CUDA y el paquete DirectML clásico no se incluyen como variantes soportadas. Si se necesita reintroducir un runtime alternativo, debe hacerse como rama experimental con referencias NuGet aisladas, validación de paquetes nativos y mediciones contra el benchmark sintético descrito más abajo.

Si faltan archivos del modelo, la aplicación solicitará autorización para descargarlos:

```text
[Missing Assets] Nemotron model layers missing. Download? (y/n):
```

Una vez cargado el modelo:

1. Selecciona el micrófono si se detecta más de uno.
2. Habla con claridad.
3. Presiona `Enter` para detener la captura.

Las sesiones ONNX se cargan en segundo plano mientras se muestra y procesa la selección del micrófono. La captura comienza únicamente cuando ambas operaciones han finalizado.

## Guardar la transcripción

El primer argumento de la aplicación se interpreta como el archivo de salida. El contenido se agrega al final si el archivo ya existe.

```powershell
dotnet run --project src/VoiceScribe.Console -- transcripcion.txt
```

La salida se escribe progresivamente y se vacía después de procesar cada bloque de audio.

`Ctrl+C` solicita cancelación durante la descarga de modelos, la espera de entrada y el procesamiento de la cola. El apagado mediante `Enter` conserva el drenaje normal de los bloques pendientes.

## Compilación

```powershell
dotnet build VoiceScribe.sln
```

Ejecutar las pruebas:

```powershell
dotnet test VoiceScribe.sln
```

## Comparación de rendimiento

Para comparar `Cpu` contra `DirectMl`, usa el mismo `VoiceAppConfig.json` y cambia solo `Inference.ExecutionProvider` o los overrides por submodelo. La consola imprime el runtime activo y reporta métricas periódicas del pipeline: extracción de características, encoder, decoder, joint, bucle de decodificación y tiempo total por bloque.

El modo benchmark evita depender del micrófono. Genera audio PCM determinístico y lo procesa por el mismo `NemotronEngine`:

Ejecuta el benchmark con la configuración actual:

```powershell
dotnet run --project src/VoiceScribe.Console -- --benchmark 20
```

Para comparar CPU puro, cambia temporalmente `ExecutionProvider` y los overrides de submodelo a `Cpu`:

```json
"Inference": {
  "ExecutionProvider": "Cpu",
  "EncoderProvider": null,
  "DecoderProvider": null,
  "JoinerProvider": null
}
```

Para comparar captura real, omite `--benchmark 20` y usa el mismo micrófono y frase en ambas ejecuciones. En ambos casos confirma en los logs si `encoder.onnx`, `decoder.onnx` o `joint.onnx` caen a CPU. Si la política de fallback difiere, compara los tiempos por sesión, no solo el tiempo total por bloque.

En una primera medición local con WindowsML, 20 bloques sintéticos y configuración `encoder=Cpu, decoder=DirectMl, joiner=DirectMl`, el promedio fue `295.0 ms` por bloque.

## Runtimes Alternativos

VoiceScribe no mantiene builds separados para `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.DirectML` ni `Microsoft.ML.OnnxRuntime.Gpu`. La decisión actual es usar WindowsML como runtime único para reducir combinaciones de paquetes nativos, salida de build, pruebas y soporte.

Si en el futuro se necesita reintroducir otro runtime, hacerlo en una rama experimental y mantener estas reglas:

- No mezclar paquetes ONNX Runtime alternativos en el mismo build.
- Reintroducir una propiedad de build explícita solo si el paquete requiere assets nativos incompatibles con WindowsML.
- Mantener `Cpu` y `DirectMl` como valores de `Inference.ExecutionProvider`; no usar nombres de paquetes como providers de configuración.
- Ejecutar el benchmark sintético y una prueba con micrófono antes de promover el cambio.
- Actualizar `constraints.md`, `readme.md` y las pruebas de `OnnxSessionFactoryResolver`.

La solución contiene tres proyectos:

| Proyecto | Responsabilidad |
| --- | --- |
| `VoiceScribe.Console` | Inicio de la aplicación, logging, selección de micrófono, captura de audio, configuración y descarga del modelo. |
| `VoiceScribe.Core` | Extracción de características, administración de modelos ONNX, cachés de streaming y decodificación RNN-T. |
| `VoiceScribe.Core.Tests` | Pruebas del contrato Nemotron, validación, estados dinámicos y extractor acústico. |

## Arquitectura

```text
Micrófono
   |
   | PCM 16 kHz / 16 bits / mono
   v
NAudio WaveInEvent
   |
   | copia rápida de bytes
   v
Cola acotada + acumulador PCM
   |
   | bloques completos de 560 ms
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

La enumeración, selección y creación del dispositivo de entrada están aisladas en `ConsoleAudioInput`, fuera del flujo principal de `Program`.

El callback `DataAvailable` no ejecuta inferencia. Copia los bytes recibidos a una cola acotada y retorna. Un único worker acumula los fragmentos hasta completar exactamente las muestras requeridas por el modelo y luego ejecuta el pipeline ONNX. Si la cola alcanza su capacidad, el bloque nuevo se descarta y se registra una advertencia.

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

Antes de crear las sesiones, la aplicación carga `genai_config.json` como contrato del modelo. De allí obtiene tamaños, parámetros RNN-T, nombres de archivos y nombres de nodos ONNX.

`NemotronEngine` crea tres sesiones de ONNX Runtime:

- `encoder.onnx`: procesa las características acústicas y actualiza las cachés de streaming.
- `decoder.onnx`: mantiene el estado lingüístico recurrente.
- `joint.onnx`: combina las salidas del encoder y decoder para obtener los logits de tokens.

La decodificación es greedy. Si `Nemotron.MaxSymbolsPerStep` o `Nemotron.BlankId` son `null`, se utilizan `max_symbols_per_step` y `blank_id` declarados por el modelo.

Las dimensiones de cachés y estados recurrentes se obtienen desde `InputMetadata` y se validan contra `genai_config.json`. Solo los ejes dinámicos de los tensores de estado inicial se resuelven con tamaño uno.

Los buffers de caché del encoder y los estados LSTM del decoder se conservan durante toda la sesión. Cada inferencia copia los valores nuevos sobre esos buffers sin recrear sus tensores.

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
| `Audio.QueueCapacity` | Número máximo de fragmentos pendientes de inferencia. |
| `Inference.ExecutionProvider` | Proveedor ONNX solicitado dentro de WindowsML: `Cpu` o `DirectMl`. |
| `Inference.EncoderProvider` | Sobrescritura opcional para `encoder.onnx`. Con `null`, usa `ExecutionProvider`. |
| `Inference.DecoderProvider` | Sobrescritura opcional para `decoder.onnx`. Con `null`, usa `ExecutionProvider`. |
| `Inference.JoinerProvider` | Sobrescritura opcional para `joint.onnx`. Con `null`, usa `ExecutionProvider`. |
| `Inference.DeviceId` | Identificador del dispositivo para proveedores GPU. |
| `Inference.GpuMemoryLimitMiB` | Límite opcional de VRAM. Se valida, pero actualmente no se aplica en DirectML. |
| `Inference.AllowCpuFallback` | Permite recrear en CPU una sesión GPU que no pueda inicializarse. |
| `Inference.EnableProfiling` | Activa el perfil de ONNX Runtime. |
| `Inference.LogSeverityLevel` | Nivel opcional de logs de ONNX Runtime: `Verbose`, `Info`, `Warning`, `Error` o `Fatal`. Con `null`, usa el valor predeterminado de ONNX Runtime. |
| `Inference.LogVerbosityLevel` | Verbosidad opcional de ONNX Runtime. Es útil principalmente con `LogSeverityLevel: "Verbose"`. |
| `Nemotron.LanguageId` | Identificador de idioma enviado al encoder; valor predeterminado: `101`. |
| `Nemotron.MaxSymbolsPerStep` | Sobrescritura opcional del máximo de tokens por frame. Con `null`, se usa el contrato del modelo. |
| `Nemotron.BlankId` | Sobrescritura opcional del token blank. Con `null`, se usa el contrato del modelo. |

Si el archivo no existe o no puede deserializarse, se usa la configuración predeterminada definida en `Program.cs`. Antes de iniciar la captura se valida que la frecuencia y las muestras por bloque coincidan con el contrato del modelo. Una configuración incompatible detiene el arranque.

## Logging

El nivel de logging de la aplicación se configura con el archivo estándar `appsettings.json`, que también se copia al directorio de salida. La sección usada por `Microsoft.Extensions.Logging` es `Logging`.

Ejemplo:

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Trace",
      "Microsoft": "Warning",
      "System": "Warning"
    },
    "Console": {
      "LogLevel": {
        "Default": "Trace"
      }
    }
  }
}
```

Esto controla el logging administrado de la aplicación. Los logs nativos de ONNX Runtime se controlan por separado desde `VoiceAppConfig.json`, mediante `Inference.LogSeverityLevel` e `Inference.LogVerbosityLevel`.

## Estructura del repositorio

```text
VoiceScribe/
├── VoiceScribe.sln
├── src/
│   ├── VoiceScribe.Console/
│   │   ├── Audio/
│   │   ├── Benchmark/
│   │   ├── CommandLine/
│   │   ├── Program.cs
│   │   ├── appsettings.json
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
| `Microsoft.Windows.AI.MachineLearning` | Runtime ONNX/WindowsML único. Incluye CPU y DirectML. |
| `NAudio` | Enumeración de dispositivos y captura de audio en Windows. |
| `Microsoft.Extensions.Logging` | Registro de eventos y errores. |
| `Microsoft.Extensions.Configuration` | Lectura de `appsettings.json` para configuración estándar de logging. |

Las versiones exactas se encuentran en los archivos `.csproj`.

## Códigos de salida

| Código | Motivo |
| ---: | --- |
| `1` | No fue posible cargar una configuración válida. |
| `2` | La lista de archivos del modelo está vacía. |
| `3` | La ruta de descarga del modelo no está configurada. |
| `4` | Faltan archivos del modelo y no pudieron descargarse. |
| `5` | No fue posible cargar el contrato desde `genai_config.json`. |
| `6` | La configuración no es compatible con el modelo. |
| `10` | No se detectó ningún dispositivo de entrada. |

## Limitaciones conocidas

- Solo funciona en Windows porque los proyectos apuntan a `net10.0-windows10.0.18362.0`, usan WindowsML y la captura usa NAudio.
- Si la inferencia no mantiene el ritmo y la cola se llena, se descartan fragmentos nuevos para no bloquear el callback de captura.
- La aplicación exige que la frecuencia y duración configuradas produzcan exactamente las muestras requeridas por el modelo; todavía no implementa resampling.
- El contrato depende de la estructura de `genai_config.json` incluida con esta familia de exportaciones.
- Se omiten bloques cuyo pico de amplitud sea inferior a `0.003`.
- No hay beam search, timestamps, separación por hablante, puntuación de confianza ni segmentación formal de frases.
- El nivel de logging administrado se configura en `appsettings.json`; los logs nativos de ONNX Runtime se configuran en `VoiceAppConfig.json`.
- No existe todavía una validación comparativa del preprocesamiento frente a la implementación original del modelo ni una prueba automatizada integral con los tres grafos ONNX.

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

Para inspeccionar más información, cambia temporalmente el nivel mínimo de logging en `appsettings.json`. Para logs nativos de ONNX Runtime, ajusta `Inference.LogSeverityLevel` e `Inference.LogVerbosityLevel` en `VoiceAppConfig.json`.

## Licencia y modelo

Este repositorio no contiene actualmente un archivo de licencia. Antes de redistribuir el código o los pesos, añade una licencia explícita y revisa por separado los términos del modelo y de sus archivos publicados en Hugging Face.
