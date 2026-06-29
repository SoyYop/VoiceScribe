# VoiceScribe Technical Constraints

This file captures invariants that should be preserved when changing VoiceScribe internals. Keep it short and update it when audio capture, preprocessing, ONNX execution, streaming state, decoding, or shutdown behavior changes.

## Sources of Truth

Use model data in this order:

1. `genai_config.json`, loaded through `NemotronModelDefinition`.
2. ONNX session metadata, used to verify real tensor names, shapes, and types.
3. `audio_processor_config.json`, used by `AudioFeatureExtractor` for preprocessing parameters.
4. `VoiceAppConfig.json`, used only for operational settings and explicit overrides.
5. Code defaults, only as documented fallbacks.

Do not hard-code ONNX node names, file names, tensor dimensions, `blank_id`, or `max_symbols_per_step` in `NemotronEngine`.

## Audio Contract

The live capture settings must match the model:

```text
SamplesPerBuffer = SampleRate * BufferMilliseconds / 1000
SamplesPerBuffer = model.ChunkSamples
SampleRate = model.SampleRate
```

For the current export:

- `SampleRate = 16000`
- `ChunkSamples = 8960`
- `BufferMilliseconds = 560`

There is no resampling. Startup must reject incompatible sample rates or chunk sizes before opening the microphone.

Supported PCM input depths are 8, 16, 24, and 32-bit integer. For multichannel input, only the first channel is read. A different channel policy must be implemented and documented explicitly.

NAudio callbacks may deliver less than, exactly, or more than one model chunk. The accumulator must preserve partial and leftover bytes.

On normal end-of-stream, any final partial chunk must be padded with silence and processed. Configured final-silence padding chunks must also bypass the silence filter so the streaming model receives trailing context before shutdown. Optional trailing silence during live capture is allowed, but the default behavior should keep live silence filtering unchanged.

## Audio Preprocessor

`AudioFeatureExtractor` converts PCM float samples in `[-1, 1]` to frame-major log-mel features for the encoder.

The current standard tensor shape is:

```text
[1, 65, 128]
```

That is 56 current frames from an 8,960-sample chunk plus nine cached pre-encoder frames. Preserve the feature cache across chunks unless the streaming state is intentionally reset.

The extractor is responsible for chunk normalization, optional dithering, preemphasis, Hann windowing, 512-point FFT, Slaney-style mel filters, log energy conversion, and pre-encoder feature caching.

## Runtime and Providers

VoiceScribe uses `Microsoft.Windows.AI.MachineLearning` as the only ONNX runtime package.

Allowed configured providers are:

- `Cpu`
- `DirectMl`

Do not mix WindowsML with `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.DirectML`, or `Microsoft.ML.OnnxRuntime.Gpu` in the same build output. Any alternate runtime should be isolated in an experiment branch, benchmarked, tested with a microphone, and documented before adoption.

CPU fallback after a DirectML session failure must create a CPU session through WindowsML. It must not add a separate CPU ONNX Runtime package.

## Tensor Shapes

Dynamic ONNX dimensions such as `-1` must be resolved according to tensor meaning, not blindly converted to `1`.

Rules that must remain true:

- Initial decoder `targets` resolves to a rank-2 tensor equivalent to `[1, 1]`.
- Decoder output embedding size is validated with `Tensor.Length`, not only the last dimension, then reshaped for the joint graph as `[1, 1, DecoderHiddenSize]`.
- Encoder output frames are interpreted as `[1, T, EncoderHiddenSize]`.
- Decoder `h_in` and `c_in` are rank-3 tensors shaped `[DecoderLayerCount, 1, DecoderHiddenSize]`.
- Encoder cache tensors are created from ONNX metadata, validated against the model contract, and updated from declared `cache_*_next` outputs.

## Streaming State and Decoding

Inference must remain sequential for a single stream. ONNX sessions, encoder caches, and decoder states are not safe to use concurrently without new synchronization and tests.

RNN-T decoding rules:

- `blank_id` comes from the model unless explicitly overridden.
- `max_symbols_per_step` comes from the model unless explicitly overridden.
- A blank token ends emission for the current acoustic frame.
- Decoder recurrent state advances only after a non-blank token.
- Do not assume blank is the final vocabulary id.

Tokenizer behavior:

- Read vocabulary from `model.vocab` in `tokenizer.json`.
- Accept string entries, array entries where the first item is the token, and object entries with `piece`.
- Strip special tokens that start with `<` or `[`.
- Convert `▁` to a space and remove the `##` prefix.

## Concurrency

`WaveInEvent.DataAvailable` must stay short:

- Copy only valid bytes.
- Try to enqueue the fragment.
- Return without feature extraction, ONNX inference, or blocking waits.

The bounded queue protects capture latency. If full, the current policy is to drop the newest fragment and log a warning.

A single worker must drain the queue, accumulate complete chunks, and run inference in audio order.

## Shutdown

Normal shutdown order:

1. Stop capture.
2. Unsubscribe `DataAvailable`.
3. Complete the queue.
4. Wait for the worker to drain pending chunks.
5. Dispose ONNX sessions and output resources.

Do not use `Thread.Sleep` as a drain mechanism.

`WaveInEvent` should be stopped before unsubscribing `DataAvailable`, so final capture events emitted during stop can still enter the queue.

`StopAsync` and `DisposeAsync` must remain idempotent. After queue completion, no new audio should be accepted.

Model downloads must honor cancellation. A failed or canceled download must not leave a partial file that later passes validation based only on file existence.

## Required Validation

Before opening the microphone, validation must report all detected configuration errors for:

- Positive and supported audio settings.
- Silence threshold between `0` and `1`.
- Queue capacity greater than zero.
- Trailing silence padding greater than or equal to zero.
- Final silence padding greater than or equal to zero.
- Whole-number samples per buffer.
- Capture sample rate matching the model.
- Capture chunk samples matching the model.
- Readable `genai_config.json`.
- Valid vocabulary and `blank_id`.
- Positive `max_symbols_per_step`.

## Change Checklist

For audio, ONNX, streaming, or decoding changes:

- Review `genai_config.json` and avoid duplicating its contract in code.
- Keep ONNX names centralized in the model definition.
- Preserve partial PCM accumulation.
- Preserve sequential streaming state.
- Resolve dynamic dimensions by tensor meaning.
- Keep capture callbacks non-blocking.
- Keep shutdown drain behavior deterministic.
- Run `dotnet build VoiceScribe.sln`.
- Run `dotnet test VoiceScribe.sln` when tests are expected to work in the current environment.
- Manually test with a microphone when changing live capture or ONNX inference.

## Current Test Coverage

Covered:

- `genai_config.json` parsing.
- Compatible and incompatible audio configuration validation.
- Dynamic `targets` initialization.
- Static recurrent state dimensions.
- Audio feature extractor output shape.
- Audio feature extractor streaming reset.

Not covered yet:

- Full three-graph ONNX execution with real model files.
- Decoder output layout against real ONNX metadata.
- PCM accumulator edge cases for partial and multiple chunks.
- Full queue behavior under sustained overload.
- `StopAsync` and `DisposeAsync` idempotency.
