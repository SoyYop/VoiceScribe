# VoiceScribe: restricciones e invariantes técnicas

Este documento registra las condiciones que deben preservarse al modificar VoiceScribe. Su objetivo es evitar regresiones causadas por asumir formas de tensores, parámetros de audio o comportamientos de concurrencia que no corresponden al modelo real.

Antes de modificar captura de audio, preprocesamiento, sesiones ONNX, decodificación RNN-T o apagado de la aplicación, se debe revisar este archivo.

## Fuentes de verdad

Los valores relacionados con el modelo deben obtenerse en este orden:

1. `genai_config.json`, cargado mediante `NemotronModelDefinition`.
2. Metadata de las sesiones ONNX, para verificar formas y tipos reales.
3. `VoiceAppConfig.json`, únicamente para opciones operativas y sobrescrituras explícitas.
4. Valores predeterminados del código, solo como fallback documentado.

No se deben volver a fijar en `NemotronEngine` nombres de nodos, nombres de archivos, dimensiones, `blank_id` o `max_symbols_per_step`.

## Contrato del modelo

La exportación utilizada está compuesta por:

- Encoder ONNX.
- Decoder ONNX.
- Joint o joiner ONNX.
- `tokenizer.json`.
- `genai_config.json`.
- `audio_processor_config.json`.

`NemotronModelDefinition` debe seguir cargando desde `genai_config.json`:

- Tamaño del vocabulario.
- `blank_id`.
- `max_symbols_per_step`.
- Frecuencia de muestreo.
- Muestras por bloque.
- Tamaños ocultos del encoder y decoder.
- Número de capas del decoder.
- Nombres de archivos ONNX.
- Nombres de entradas y salidas de cada grafo.

Si cambia la estructura de `genai_config.json`, debe adaptarse el parser y fallar durante el arranque con un mensaje claro. No se debe continuar con un contrato parcial.

## Runtime ONNX

VoiceScribe usa WindowsML como único runtime ONNX mediante `Microsoft.Windows.AI.MachineLearning`. No deben reintroducirse builds separados para `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.DirectML` o `Microsoft.ML.OnnxRuntime.Gpu` sin una rama experimental y mediciones comparativas.

`Inference.ExecutionProvider` no representa paquetes NuGet ni flavors de compilación. Solo puede seleccionar proveedores disponibles dentro de WindowsML:

- `Cpu`.
- `DirectMl`.

El proveedor CPU está incluido en WindowsML. El fallback de DirectML a CPU debe seguir creando una sesión sin registrar explícitamente DirectML, no agregando un paquete CPU separado.

Si se retoma otro runtime en el futuro:

- No se deben mezclar paquetes ONNX Runtime alternativos en el mismo output.
- Deben actualizarse `readme.md`, `vram.md`, pruebas de factories y validación de configuración.
- Debe medirse con `--benchmark` y con micrófono real antes de promover el cambio.

## Audio

La captura debe cumplir:

```text
SamplesPerBuffer = SampleRate × BufferMilliseconds / 1000
SamplesPerBuffer = model.ChunkSamples
SampleRate = model.SampleRate
```

Para la exportación actual:

- Frecuencia: 16 000 Hz.
- Bloque: 8960 muestras.
- Duración: 560 ms.

Actualmente no existe resampling. No se debe aceptar una frecuencia diferente a la requerida por el modelo.

Los formatos PCM admitidos son 8, 16, 24 y 32 bits enteros. En entrada multicanal se procesa únicamente el primer canal. Cambiar este comportamiento requiere implementar y documentar una política de mezcla o selección de canal.

Los callbacks de NAudio pueden entregar:

- Menos de un bloque.
- Exactamente un bloque.
- Más de un bloque.

Por ello, no se deben descartar bytes sobrantes ni asumir que cada evento equivale a una inferencia. El acumulador debe conservar los bytes restantes hasta completar el siguiente bloque.

`AudioFeatureExtractor.NormalizeChunkLength` no sustituye la validación de captura. El arranque debe rechazar configuraciones incompatibles antes de iniciar el micrófono.

## Concurrencia y tiempo real

`WaveInEvent.DataAvailable` debe permanecer corto y no bloqueante:

- Copiar únicamente los bytes válidos.
- Intentar encolarlos.
- Retornar sin ejecutar extracción de características ni inferencia ONNX.

La inferencia debe ejecutarse en un único worker para preservar el orden del audio y el estado de streaming.

La cola debe ser acotada. Si se llena, la política actual es descartar el fragmento nuevo y registrar una advertencia. No debe cambiarse a una espera bloqueante dentro del callback de NAudio.

Las sesiones ONNX y los estados del decoder/encoder no deben utilizarse simultáneamente desde varios threads sin introducir sincronización y pruebas específicas.

El worker acepta cancelación. El apagado normal completa y drena la cola; una cancelación explícita puede interrumpir el trabajo pendiente después de finalizar la inferencia síncrona que ya esté en curso.

## Formas de tensores ONNX

Las dimensiones dinámicas de ONNX suelen aparecer como `-1`. No todas significan lo mismo y no deben convertirse globalmente a `1`.

Solo se resuelven con tamaño inicial `1` los ejes dinámicos de los tensores de estado inicial usados por:

- Batch de streaming.
- Secuencia inicial de `targets`.
- Estados recurrentes iniciales.

Después de resolverlos, las formas deben validarse contra `NemotronModelDefinition`.

### Decoder targets

`targets` tiene una dimensión de secuencia dinámica. El estado inicial debe ser un tensor de rango 2 equivalente a `[1, 1]`.

El error que originó esta regla fue:

```text
Input 'targets' has an unresolved dimension at index 1.
```

No se debe volver a rechazar automáticamente esta dimensión dinámica ni aplicar la misma solución indiscriminadamente a cualquier tensor.

### Salida del decoder

La salida de un paso del decoder contiene 640 valores para la exportación actual, pero el eje final del tensor puede medir `1`. Por ello:

- El tamaño del embedding de un paso se valida con `Tensor.Length`.
- No se valida usando `Dimensions[^1]`.
- Antes de enviarlo al joint se reinterpreta como `[1, 1, DecoderHiddenSize]`.

El error que originó esta regla fue:

```text
Decoder produced embedding size 1; expected 640.
```

### Salida del encoder

La salida acústica se interpreta como `[1, T, EncoderHiddenSize]`. Al extraer un frame:

- Debe usarse el tamaño oculto declarado por el modelo.
- El resultado enviado al joint debe ser `[1, 1, EncoderHiddenSize]`.

### Estados recurrentes

Los estados `h_in` y `c_in` deben ser de rango 3 y coincidir con:

```text
[DecoderLayerCount, 1, DecoderHiddenSize]
```

Sus siguientes valores se obtienen de `h_out` y `c_out` después de emitir un token no blank.

Los tensores persistentes `h_in` y `c_in` deben reutilizarse. Los valores de salida se copian sobre sus buffers; no se deben recrear ambos tensores por cada token.

### Cachés del encoder

Las cachés iniciales se crean desde la metadata ONNX y se validan contra el tamaño oculto del encoder. Después de cada ejecución deben reemplazarse usando las salidas `cache_*_next` declaradas en el contrato.

Los objetos tensor de caché deben conservarse durante la sesión. Se actualizan sus buffers para evitar asignaciones de gran tamaño por cada bloque.

## Decodificación RNN-T

- `blank_id` se obtiene del modelo, salvo sobrescritura explícita.
- `max_symbols_per_step` se obtiene del modelo, salvo sobrescritura explícita.
- `BlankId` debe estar dentro del rango del vocabulario.
- Un token blank termina la emisión para el frame acústico actual.
- Los estados del decoder solo avanzan después de un token no blank.

No se debe asumir que blank es siempre la última clase, aunque esto sea común en algunos modelos.

## Tokenizador

El tokenizador actual espera `model.vocab` dentro de `tokenizer.json`.

Se admiten vocabularios cuyos elementos sean:

- Strings.
- Arreglos donde el primer elemento es el token.
- Objetos con propiedad `piece`.

Si se incorpora otra familia de tokenizadores, debe aislarse en una implementación específica en lugar de añadir condiciones indefinidamente dentro del motor.

## Ciclo de vida y apagado

El orden de apagado debe ser:

1. Desuscribir `DataAvailable`.
2. Detener la captura.
3. Completar la cola.
4. Esperar a que el worker procese los bloques pendientes.
5. Liberar sesiones ONNX y recursos de salida.

No se debe reintroducir `Thread.Sleep` como mecanismo de drenaje.

`StopAsync` y `DisposeAsync` deben permanecer idempotentes. No deben aceptar audio después de completar la cola.

La descarga de modelos también debe propagar `CancellationToken`. Una descarga cancelada o fallida no debe dejar un archivo parcial que pueda superar posteriormente una verificación basada solo en existencia.

## Inicio concurrente

La carga de sesiones ONNX puede ejecutarse en segundo plano mientras el usuario selecciona el micrófono. Deben cumplirse estas condiciones:

- La definición y configuración del modelo ya fueron validadas.
- La captura no comienza hasta que la selección y la carga hayan terminado.
- Si se cancela la selección, cualquier motor que termine de cargarse debe ser dispuesto.
- Los errores de carga deben observarse y propagarse antes de iniciar la captura.

## Propiedad de recursos

Debe existir un único propietario claro para:

- Sesiones ONNX.
- Worker y cola.
- Escritor de transcripción.
- Fuente de audio.

El programa que crea el escritor de transcripción es su único propietario y debe disponerlo. El motor puede escribir y hacer `Flush`, pero no debe cerrar ni disponer el writer recibido.

## Validación obligatoria

Antes de abrir el micrófono se debe validar:

- Configuración de audio positiva y soportada.
- Umbral de silencio entre 0 y 1.
- Capacidad de cola mayor que cero.
- Número entero de muestras por búfer.
- Frecuencia coincidente con el modelo.
- Muestras por bloque coincidentes con el modelo.
- Vocabulario y `blank_id` válidos.
- `max_symbols_per_step` positivo.
- Existencia y lectura de `genai_config.json`.

La validación debe presentar todos los errores de configuración detectados, no únicamente el primero.

## Checklist para cambios

Antes de considerar completo un cambio relacionado con audio o inferencia:

- [ ] Se revisó `genai_config.json` y no se duplicó su información en código.
- [ ] No se añadieron nombres de nodos ONNX como strings dispersos.
- [ ] No se asumió que el último eje representa siempre el embedding.
- [ ] Las dimensiones dinámicas se resolvieron según el significado del tensor.
- [ ] El callback de audio sigue sin ejecutar inferencia ni esperar.
- [ ] Los bytes parciales y sobrantes se conservan correctamente.
- [ ] El orden del audio y el estado de streaming siguen siendo secuenciales.
- [ ] El apagado completa y drena la cola.
- [ ] `dotnet build VoiceScribe.sln` finaliza sin errores ni advertencias.
- [ ] Se realizó una prueba manual con micrófono y se confirmó que no aparecen errores ONNX.
- [ ] Si cambió una restricción, se actualizó este archivo y `readme.md`.

## Cobertura automatizada

La suite actual cubre:

- Parseo de `genai_config.json`.
- Validación compatible e incompatible de captura contra el modelo.
- Resolución de `targets` dinámico a `[1, 1]`.
- Preservación de dimensiones estáticas de estados recurrentes.
- Forma de salida del extractor acústico.
- Reinicio del estado de streaming del extractor.

Siguen pendientes:

- Validación automatizada del layout real de la salida del decoder.
- Contrato completo de entradas y salidas usando los tres grafos ONNX.
- Acumulación de fragmentos PCM parciales y múltiples.
- Cola llena y política de descarte.
- Idempotencia de `StopAsync` y `DisposeAsync`.
