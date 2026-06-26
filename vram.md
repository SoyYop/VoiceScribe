# Ejecución de modelos ONNX en VRAM

## Situación actual de VoiceScribe

VoiceScribe ejecuta tres grafos ONNX independientes: encoder, decoder y joint. `NemotronEngine.Initialize` crea las tres sesiones con `SessionOptions`, optimización completa del grafo y ejecución secuencial, pero no registra ningún proveedor de GPU.

El proyecto referencia `Microsoft.ML.OnnxRuntime` 1.27.0, que es la distribución para CPU. Por lo tanto, actualmente:

- Los pesos y las activaciones se mantienen en RAM.
- Encoder, decoder y joint se ejecutan en CPU.
- Los `DenseTensor<T>` de entrada, las cachés acústicas y los estados del decoder también viven en RAM.
- La carga en segundo plano no implica uso de VRAM; solo evita bloquear la selección del micrófono.

La exportación instalada es INT4 y sus datos externos ocupan aproximadamente 751 MiB:

| Grafo | Datos externos aproximados |
|---|---:|
| Encoder | 658 MiB |
| Decoder | 57 MiB |
| Joint | 36 MiB |

El encoder contiene el operador contribuyente `com.microsoft::MatMulNBits`. Esto importa porque la GPU solo ejecutará los operadores para los que el proveedor CUDA tenga kernel compatible; ONNX Runtime puede dejar otros operadores en CPU.

## Enfoque genérico en C#/.NET

ONNX Runtime separa el modelo de los *Execution Providers*. En C# se crea una `InferenceSession` con proveedores ordenados por preferencia:

1. CUDA para una GPU NVIDIA.
2. CPU como respaldo para nodos no compatibles.

Al crear la sesión, ONNX Runtime particiona el grafo. Los pesos y activaciones de los nodos asignados a CUDA se llevan a VRAM. Que la sesión use CUDA no significa necesariamente que el modelo completo esté en VRAM ni que todos sus operadores se ejecuten en GPU.

La implementación habitual requiere:

- Usar `Microsoft.ML.OnnxRuntime.Gpu` en lugar de `Microsoft.ML.OnnxRuntime`, con la misma versión.
- Instalar las versiones de CUDA y cuDNN compatibles con esa versión de ONNX Runtime.
- Agregar CUDA a `SessionOptions` antes de crear cada sesión.
- Mantener CPU disponible como fallback.
- Registrar y medir qué nodos fueron asignados a cada proveedor.
- Opcionalmente usar `OrtValue` e I/O Binding para conservar entradas, salidas y estados persistentes en memoria de dispositivo. Esto es una optimización posterior, no un requisito para comenzar a usar CUDA.

También existen DirectML y otros proveedores. Para esta NVIDIA GTX 1050 conviene probar CUDA primero, porque ofrece configuración y diagnóstico más específicos para hardware NVIDIA.

## Viabilidad en la GTX 1050 de 4 GB

La GPU detectada es una NVIDIA GeForce GTX 1050 con 4096 MiB y compute capability 6.1. El tamaño de los pesos sugiere que hay espacio para una primera prueba, pero los 751 MiB de archivos no equivalen al consumo final:

- ONNX Runtime puede crear copias transformadas o descomprimidas de algunos pesos.
- Las activaciones, espacios de trabajo y arenas de memoria consumen VRAM adicional.
- Hay tres sesiones y cada una administra sus propios recursos.
- Windows y otras aplicaciones también consumen VRAM.

La GTX 1050 es Pascal y no posee Tensor Cores. CUDA puede reducir el tiempo del encoder, pero no debe asumirse que INT4 producirá una gran aceleración. `MatMulNBits`, las transferencias CPU/GPU y el decoder ejecutado muchas veces por frame pueden limitar el beneficio. La decisión debe basarse en mediciones.

Objetivo inicial razonable: mantener el consumo de VoiceScribe por debajo de aproximadamente 3 GiB, dejando margen al sistema. El límite no garantiza que el modelo quepa: si una asignación individual falla, ONNX Runtime puede producir un error de memoria.

## Cambios necesarios en el código actual

### Estado de implementación

La base común para los tres modos ya está implementada:

- `OnnxRuntimeOptions` separa la configuración del runtime de las opciones de Nemotron.
- `OnnxExecutionProvider` declara `Cpu`, `DirectMl` y `Cuda`.
- `IOnnxSessionFactory` se inyecta en `NemotronEngine`.
- `CpuOnnxSessionFactory` conserva el comportamiento actual.
- `DirectMlOnnxSessionFactory` está disponible al compilar con `OnnxRuntimeFlavor=DirectMl`.
- La variante CPU rechaza DirectML y CUDA; la variante DirectML rechaza CUDA.

La prueba con los modelos reales detectó que decoder y joint cargan en DirectML, pero el encoder INT4 falla durante la inicialización con `E_INVALIDARG`. Si `AllowCpuFallback` está activo, solo esa sesión se recrea en CPU y el modo queda híbrido. Queda pendiente la variante CUDA y medir si acelerar decoder/joint compensa las transferencias.

### 1. Dependencia nativa

En `VoiceScribe.Core.csproj`:

- La variante CPU selecciona `Microsoft.ML.OnnxRuntime` 1.27.0.
- La variante DirectML selecciona `Microsoft.ML.OnnxRuntime.DirectML` 1.24.4, última versión publicada de ese paquete.
- La futura variante CUDA deberá seleccionar `Microsoft.ML.OnnxRuntime.Gpu` sin combinar distribuciones nativas.
- Se debe documentar CUDA, cuDNN, el runtime de Visual C++ y las rutas de DLL requeridas por la versión CUDA elegida.

Antes de implementarlo se debe confirmar la matriz exacta de compatibilidad de la versión restaurada. No conviene actualizar ONNX Runtime al mismo tiempo que se introduce CUDA.

### 2. Configuración

Las opciones específicas del runtime son:

- `ExecutionProvider`: `Cpu`, `DirectMl` o `Cuda`.
- `DeviceId`: inicialmente `0`.
- `GpuMemoryLimitMiB`: opcional; valor inicial sugerido para pruebas, `3072`.
- `EnableProfiling`: activa el perfil de ONNX Runtime.

La selección es explícita. Si el proveedor solicitado no está incluido en la variante instalada, el arranque falla con un mensaje claro. ONNX Runtime puede seguir asignando a CPU los nodos que el proveedor GPU no soporte.

La validación debe rechazar proveedores desconocidos, identificadores negativos y límites de memoria inválidos.

### 3. Creación centralizada de sesiones

Extraer de `NemotronEngine.Initialize` una fábrica de `SessionOptions` o de sesiones. Esa pieza debe:

- Conservar `ORT_ENABLE_ALL` y `ORT_SEQUENTIAL` inicialmente.
- Registrar CUDA con prioridad sobre CPU.
- Aplicar `device_id`, límite de memoria y estrategia de arena cuando estén configurados.
- Crear encoder, decoder y joint con la misma política.
- Liberar opciones y sesiones parciales si una creación falla.
- Informar el proveedor solicitado, el proveedor efectivo y el fallback.

No deben cambiarse nombres de nodos, formas, contratos del modelo, flujo RNN-T ni concurrencia del worker.

### 4. Primera etapa: CUDA con tensores en RAM

La primera implementación debería conservar los `DenseTensor<T>` y las llamadas actuales a `Run`. ONNX Runtime hará las transferencias necesarias para los nodos CUDA.

Ventajas:

- Cambio pequeño y reversible.
- Permite validar compatibilidad de los tres grafos.
- Entrega una medición base antes de introducir administración manual de memoria.

Costo:

- Las cachés del encoder y los estados del decoder vuelven a RAM después de cada ejecución y se copian otra vez a GPU en la siguiente.
- El decoder y el joint realizan muchas ejecuciones pequeñas, donde el costo de transferencia puede superar el ahorro de cómputo.

### 5. Segunda etapa opcional: estados residentes en VRAM

Solo si el perfil muestra transferencias relevantes, migrar gradualmente a `OrtValue` e I/O Binding:

- Mantener `cache_last_channel`, `cache_last_time`, `h_in` y `c_in` en el dispositivo.
- Enlazar salidas siguientes a buffers reutilizables.
- Evitar copiar logits completos a CPU si es posible; como mínimo copiar únicamente lo necesario para `ArgMax`.
- Preservar el único worker de inferencia y la propiedad clara de todos los recursos nativos.

Esta etapa es más compleja porque el código actual inspecciona, remodela y copia tensores administrados. Debe implementarse después de probar correctamente la etapa inicial.

## Pruebas y criterios de aceptación

1. Arranque en `Cpu`: comportamiento idéntico al actual.
2. Arranque en `Cuda`: las tres sesiones se crean o se informa exactamente cuál grafo u operador falla.
3. Arranque en `Auto` sin DLL de CUDA/cuDNN: fallback controlado a CPU.
4. Confirmar mediante logs/perfil de ONNX Runtime qué nodos usa CUDA y cuáles CPU.
5. Medir con el mismo audio:
   - tiempo de carga;
   - latencia del encoder;
   - latencia total por bloque de 560 ms;
   - uso máximo de VRAM;
   - uso de CPU;
   - bloques descartados por cola llena.
6. Ejecutar varios minutos para detectar crecimiento de VRAM o errores de memoria.
7. Verificar que la transcripción y el avance de cachés/estados coincidan con CPU.

La GPU se considerará útil si reduce de forma estable la latencia total o los bloques descartados sin agotar VRAM. Que `nvidia-smi` muestre memoria ocupada no basta para demostrar que la parte costosa del grafo está acelerada.

## Orden recomendado de implementación

1. Añadir opciones y validación del proveedor.
2. Cambiar al paquete GPU e instalar dependencias CUDA/cuDNN compatibles.
3. Centralizar la creación de sesiones.
4. Implementar `Cpu`, `Cuda` y `Auto`.
5. Agregar diagnóstico y medición.
6. Probar los tres grafos en la GTX 1050.
7. Evaluar I/O Binding únicamente si las transferencias son el cuello de botella.

## Referencias

- ONNX Runtime, CUDA Execution Provider: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- ONNX Runtime, proveedores disponibles: <https://onnxruntime.ai/docs/execution-providers/>
- NVIDIA, GPUs CUDA y compute capability: <https://developer.nvidia.com/cuda-gpus>
