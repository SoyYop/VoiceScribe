# VoiceScribe Notepad

Cuaderno de dictado local para Windows, con WinUI 3, Windows App SDK y .NET 10. Este proyecto contiene su propia copia adaptada del motor Nemotron: no tiene referencias a VoiceScribe.Core ni a VoiceScribe.Console.

## Ejecutar

```powershell
dotnet run --project src/VoiceScribe.Notepad
```

Requiere Windows 10 1903 o superior, .NET 10 y un equipo x64. La compilación incluye el runtime de Windows App SDK. El runtime .NET sigue siendo necesario.

1. Abre **Herramientas → Configuración**.
2. Selecciona el micrófono y la carpeta del modelo Nemotron ONNX. Puedes usar `artifacts/models/nemotron-3.5-asr` de este repositorio: se reutilizan los archivos del modelo, no los proyectos de código.
3. Si no tienes el modelo, marca **Descargar los archivos que faltan al guardar**. La descarga usa Hugging Face y puede cancelarse. Un archivo solo se publica con su nombre definitivo cuando termina la descarga.
4. Crea una sesión y pulsa **Iniciar dictado**. El texto reconocido se añade al final de la nota y puedes editarlo.
5. Pulsa **Detener dictado** para terminar de procesar las últimas palabras. Cerrar la ventana también detiene, drena y guarda antes de salir.

El ID de idioma inicial es `101`, igual que en la configuración del modelo usada por este repositorio. Puedes cambiarlo en Configuración. DirectML está activado con respaldo CPU por sesión; también puedes seleccionar CPU.

## Archivos y sesiones

- Notas: la carpeta Documentos del usuario, subcarpeta `VoiceScribe`. Respeta la ubicación de Documentos configurada en Windows, incluida su redirección a OneDrive.
- Cada sesión es un archivo `.md`, con el título como primera cabecera y el contenido debajo. El nombre incluye fecha y un identificador único. Cambiar el título no cambia el nombre del archivo.
- Guardado automático un segundo después, como máximo en condiciones normales, mediante reemplazo de un archivo temporal. Los fallos se muestran en la barra inferior y se reintenta mientras haya cambios pendientes.
- Ajustes: `%LOCALAPPDATA%\VoiceScribe\settings.json`.
- Modelos descargados por defecto: `%LOCALAPPDATA%\VoiceScribe\Models\Nemotron`.
- Las notas existentes se cargan al abrir la aplicación. Para abrir su carpeta usa **Archivo → Abrir carpeta de notas**.

Evita editar simultáneamente el mismo archivo en otra aplicación: esta primera versión no fusiona cambios externos ni observa la carpeta en tiempo real. No incluye previsualización Markdown, comandos de voz ni borrado de sesiones.

## Responsabilidades

| Carpeta | Responsabilidad |
| --- | --- |
| `Views` | Ventana WinUI, interacción y despacho de tokens al hilo de UI |
| `ViewModels` | Colección de sesiones y selección del cuaderno |
| `Models` | Datos de una sesión |
| `Services` | Persistencia Markdown, ajustes, descarga y ciclo de dictado |
| `Audio` | Conversión PCM a características log-mel |
| `Engine` | Sesiones ONNX, cachés y decodificación RNN-T |
| `Configuration` | Opciones y validación del contrato |
| `ModelAssets` | Nombres de archivos y contrato del modelo |

La captura solo encola audio; un trabajador secuencial hace la inferencia. Los tokens llegan a la UI mediante un evento. Durante el dictado no se puede cambiar de sesión. El cierre espera el evento final del micrófono, drena la cola, procesa el fragmento parcial y el silencio final, espera las notificaciones pendientes de UI y guarda. Cuatro bloques de silencio posteriores al habla permiten terminar las frases durante las pausas.

## Validación

```powershell
dotnet build VoiceScribe.sln
dotnet test VoiceScribe.sln --no-build
```

La prueba ONNX opcional requiere modelos locales:

```powershell
$env:VOICESCRIBE_TEST_MODEL = (Resolve-Path artifacts/models/nemotron-3.5-asr).Path
dotnet test tests/VoiceScribe.Notepad.Tests
```

Esta prueba verifica los tres grafos con CPU, el vaciado de audio parcial y la detención idempotente. No mide precisión de reconocimiento. La validación final de dictado requiere probar un micrófono real: frase, pausa, reanudación, detención, edición y reapertura de la sesión.
