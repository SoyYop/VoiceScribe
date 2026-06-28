# Ejecución de modelos ONNX en VRAM

## Decisión actual

VoiceScribe usa WindowsML como único runtime ONNX mediante `Microsoft.Windows.AI.MachineLearning` 2.1.71. No hay builds separados para `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.DirectML` ni `Microsoft.ML.OnnxRuntime.Gpu`.

Dentro de WindowsML se selecciona el proveedor de ejecución con `VoiceAppConfig.json`:

- `Cpu`: crea sesiones ONNX sin registrar un proveedor GPU explícito.
- `DirectMl`: registra DirectML con `AppendExecutionProvider_DML`.

WindowsML incluye el proveedor CPU. Por eso el fallback de DirectML a CPU no requiere una referencia NuGet adicional.

## Comportamiento observado

VoiceScribe ejecuta tres grafos ONNX independientes:

- `encoder.onnx`
- `decoder.onnx`
- `joint.onnx`

Con la exportación INT4 actual, `decoder.onnx` y `joint.onnx` crean sesiones DirectML correctamente. `encoder.onnx` falla al inicializar DirectML con `E_INVALIDARG`; si `AllowCpuFallback` está activo, la sesión se recrea en CPU.

El modo efectivo habitual en esta máquina es:

```text
encoder = Cpu
decoder = DirectMl
joiner  = DirectMl
```

En una medición local con WindowsML, 20 bloques sintéticos y esa configuración híbrida, el promedio fue aproximadamente `295.0 ms` por bloque.

## VRAM

La exportación instalada es INT4 y sus datos externos ocupan aproximadamente 751 MiB:

| Grafo | Datos externos aproximados |
|---|---:|
| Encoder | 658 MiB |
| Decoder | 57 MiB |
| Joint | 36 MiB |

Ese tamaño no equivale al consumo final de VRAM:

- ONNX Runtime puede crear copias transformadas o descomprimidas de pesos.
- Las activaciones, espacios de trabajo y arenas de memoria consumen VRAM adicional.
- Hay tres sesiones y cada una administra sus propios recursos.
- Windows y otras aplicaciones también consumen VRAM.

El límite `Inference.GpuMemoryLimitMiB` se valida, pero DirectML no lo aplica actualmente en este código.

## Runtimes retirados

### DirectML clásico

La variante basada en `Microsoft.ML.OnnxRuntime.DirectML` fue retirada como build separado. WindowsML encapsula DirectML y CPU en un solo runtime, con menos combinaciones de paquetes nativos que probar.

### CPU puro

La variante basada en `Microsoft.ML.OnnxRuntime` fue retirada como build separado. CPU sigue disponible como execution provider dentro de WindowsML.

### CUDA

CUDA fue probado como experimento separado con `Microsoft.ML.OnnxRuntime.Gpu` 1.27.0, CUDA 13.3 y cuDNN 9.23. Después de resolver DLL faltantes, falló en `cublasCreate` con la GTX 1050/Pascal por requerir una característica arquitectónica ausente. No se mantiene soporte CUDA en esta rama.

## Cómo reintroducir un runtime alternativo

Si en el futuro se necesita probar otro runtime, hacerlo en una rama experimental:

1. Agregar una propiedad de build explícita solo para ese experimento.
2. Referenciar exactamente un paquete ONNX Runtime por build.
3. No mezclar `Microsoft.Windows.AI.MachineLearning`, `Microsoft.ML.OnnxRuntime.DirectML`, `Microsoft.ML.OnnxRuntime.Gpu` y `Microsoft.ML.OnnxRuntime` en el mismo output.
4. Mantener `Cpu` y `DirectMl` como valores de `Inference.ExecutionProvider`; no usar nombres de paquetes como providers.
5. Restaurar pruebas de resolver/factory para la nueva matriz de runtime.
6. Medir con `--benchmark` y con micrófono real antes de adoptar el cambio.

## Criterios de aceptación para nuevos experimentos

- Compilación limpia.
- Pruebas unitarias limpias.
- Carga de `encoder`, `decoder` y `joint` con fallback explícito y logs claros.
- Medición de latencia por etapa: features, encoder, decoder, joint y total por bloque.
- Revisión de uso de VRAM durante varios minutos.
- Sin cambios en nombres de nodos, formas de tensores, cachés de streaming ni decodificación RNN-T.
