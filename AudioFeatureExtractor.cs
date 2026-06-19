using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;

namespace speech
{
    public sealed record AudioFeatures(float[] Data, int Frames, int MelBins);

    /// <summary>
    /// Local streaming feature extractor for NVIDIA Nemotron 3.5 ASR ONNX.
    ///
    /// Converts mono 16 kHz PCM float samples in [-1, 1] into log-mel features.
    /// Output layout is frame-major: Data[frame * MelBins + mel].
    /// For Nemotron ONNX encoder, the intended tensor shape is [1, 65, 128].
    ///
    /// The encoder expects 65 feature frames per 560 ms chunk:
    ///   56 current frames from 8960 samples at hop_length=160
    /// +  9 pre-encoder cached frames from the previous chunk.
    /// </summary>
    public sealed class AudioFeatureExtractor
    {
        public int SampleRate { get; }
        public int NFft { get; }
        public int HopLength { get; }
        public int NMels { get; }
        public int WindowLength { get; }
        public float FMin { get; }
        public float FMax { get; }
        public float Dither { get; }
        public float Preemphasis { get; }
        public bool Center { get; }
        public float LogZeroGuardValue { get; }
        public float MagPower { get; }
        public bool EnableDither { get; }
        public int ChunkSamples { get; }
        public int PreEncodeCacheFrames { get; }
        public int CurrentFramesPerChunk { get; }
        public int TotalFramesPerChunk => PreEncodeCacheFrames + CurrentFramesPerChunk;

        private readonly float[] _hannWindow;
        private readonly float[] _paddedWindow;
        private readonly float[][] _melFilters;
        private readonly float[] _featureCache;
        private readonly Random _rng = new(1234);

        private bool _hasFeatureCache;
        private bool _hasLastRawSample;
        private float _lastRawSample;

        public AudioFeatureExtractor()
            : this(
                sampleRate: 16000,
                nFft: 512,
                hopLength: 160,
                nMels: 128,
                windowLength: 400,
                fMin: 0f,
                fMax: 8000f,
                dither: 1e-5f,
                preemphasis: 0.97f,
                center: true,
                logZeroGuardValue: 1e-10f,
                magPower: 2.0f,
                enableDither: false,
                chunkSamples: 8960,
                preEncodeCacheFrames: 9)
        {
        }

        private AudioFeatureExtractor(
            int sampleRate,
            int nFft,
            int hopLength,
            int nMels,
            int windowLength,
            float fMin,
            float fMax,
            float dither,
            float preemphasis,
            bool center,
            float logZeroGuardValue,
            float magPower,
            bool enableDither,
            int chunkSamples,
            int preEncodeCacheFrames)
        {
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
            if (nFft <= 0 || (nFft & (nFft - 1)) != 0) throw new ArgumentException("nFft must be a power of two.", nameof(nFft));
            if (hopLength <= 0) throw new ArgumentOutOfRangeException(nameof(hopLength));
            if (nMels <= 0) throw new ArgumentOutOfRangeException(nameof(nMels));
            if (windowLength <= 0 || windowLength > nFft) throw new ArgumentOutOfRangeException(nameof(windowLength));
            if (chunkSamples <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSamples));
            if (preEncodeCacheFrames < 0) throw new ArgumentOutOfRangeException(nameof(preEncodeCacheFrames));

            SampleRate = sampleRate;
            NFft = nFft;
            HopLength = hopLength;
            NMels = nMels;
            WindowLength = windowLength;
            FMin = fMin;
            FMax = fMax > 0 ? fMax : sampleRate / 2.0f;
            Dither = dither;
            Preemphasis = preemphasis;
            Center = center;
            LogZeroGuardValue = logZeroGuardValue > 0 ? logZeroGuardValue : 1e-10f;
            MagPower = magPower;
            EnableDither = enableDither;
            ChunkSamples = chunkSamples;
            PreEncodeCacheFrames = preEncodeCacheFrames;

            // For the exported Nemotron 560 ms model: 8960 / 160 = 56 current frames.
            CurrentFramesPerChunk = Math.Max(1, ChunkSamples / HopLength);

            _hannWindow = BuildHann(WindowLength);
            _paddedWindow = PadWindowToFft(_hannWindow, NFft);
            _melFilters = BuildMelFilterBank(SampleRate, NFft, NMels, FMin, FMax);
            _featureCache = new float[PreEncodeCacheFrames * NMels];
        }

        public static AudioFeatureExtractor FromConfig(
            string audioProcessorConfigPath,
            string? genaiConfigPath = null,
            bool enableDither = false)
        {
            if (!File.Exists(audioProcessorConfigPath))
                throw new FileNotFoundException("No se encontró audio_processor_config.json", audioProcessorConfigPath);

            using var audioDoc = JsonDocument.Parse(File.ReadAllText(audioProcessorConfigPath));
            JsonElement audioRoot = audioDoc.RootElement;
            JsonElement p = audioRoot.TryGetProperty("audio_params", out var audioParams)
                ? audioParams
                : audioRoot;

            int chunkSamples = 8960;
            int preEncodeCacheFrames = 9;
            float logGuard = GetFloat(p, "log_zero_guard_value", 1e-10f);

            if (!string.IsNullOrWhiteSpace(genaiConfigPath) && File.Exists(genaiConfigPath))
            {
                using var genaiDoc = JsonDocument.Parse(File.ReadAllText(genaiConfigPath));
                JsonElement genaiRoot = genaiDoc.RootElement;
                JsonElement m = genaiRoot.TryGetProperty("model", out var model) ? model : genaiRoot;

                chunkSamples = GetInt(m, "chunk_samples", chunkSamples);
                preEncodeCacheFrames = GetInt(m, "pre_encode_cache_size", preEncodeCacheFrames);

                // audio_processor_config takes precedence; genai_config only provides fallback.
                logGuard = GetFloat(p, "log_zero_guard_value", GetFloat(m, "log_eps", logGuard));
            }

            return new AudioFeatureExtractor(
                GetInt(p, "sample_rate", 16000),
                GetInt(p, "n_fft", 512),
                GetInt(p, "hop_length", 160),
                GetInt(p, "n_mels", 128),
                GetInt(p, "window_length", 400),
                GetFloat(p, "fmin", 0f),
                GetFloat(p, "fmax", 8000f),
                GetFloat(p, "dither", 1e-5f),
                GetFloat(p, "preemphasis", 0.97f),
                GetBool(p, "center", true),
                logGuard,
                GetFloat(p, "mag_power", 2.0f),
                enableDither,
                chunkSamples,
                preEncodeCacheFrames);
        }

        /// <summary>
        /// Extracts exactly [PreEncodeCacheFrames + CurrentFramesPerChunk, NMels] features.
        /// For the standard Nemotron 560 ms model this returns [65, 128].
        /// </summary>
        public AudioFeatures Extract(float[] pcm)
        {
            if (pcm == null || pcm.Length == 0)
                throw new ArgumentException("PCM buffer is empty.", nameof(pcm));

            float[] normalizedChunk = NormalizeChunkLength(pcm, ChunkSamples);
            float[] emphasized = ApplyDitherAndPreemphasis(normalizedChunk);
            float[] currentFeatures = ExtractCurrentFrames(emphasized, CurrentFramesPerChunk);

            float[] output = new float[TotalFramesPerChunk * NMels];

            if (_hasFeatureCache)
                Array.Copy(_featureCache, 0, output, 0, _featureCache.Length);
            // If no cache exists yet, the leading frames remain zero-filled.

            Array.Copy(
                currentFeatures,
                0,
                output,
                PreEncodeCacheFrames * NMels,
                currentFeatures.Length);

            UpdateFeatureCache(currentFeatures);

            return new AudioFeatures(output, TotalFramesPerChunk, NMels);
        }

        /// <summary>
        /// Same features, but flattened as [n_mels, frames].
        /// Use only if the encoder expects mel-major layout. Nemotron metadata shown so far expects frame-major [1,65,128].
        /// </summary>
        public AudioFeatures ExtractMelMajor(float[] pcm)
        {
            AudioFeatures frameMajor = Extract(pcm);
            float[] transposed = new float[frameMajor.Data.Length];

            for (int frame = 0; frame < frameMajor.Frames; frame++)
            {
                for (int mel = 0; mel < frameMajor.MelBins; mel++)
                    transposed[mel * frameMajor.Frames + frame] = frameMajor.Data[frame * frameMajor.MelBins + mel];
            }

            return new AudioFeatures(transposed, frameMajor.Frames, frameMajor.MelBins);
        }

        public void ResetStreamingState()
        {
            Array.Clear(_featureCache);
            _hasFeatureCache = false;
            _hasLastRawSample = false;
            _lastRawSample = 0f;
        }

        private float[] NormalizeChunkLength(float[] pcm, int targetSamples)
        {
            if (pcm.Length == targetSamples)
                return pcm.ToArray();

            var normalized = new float[targetSamples];
            Array.Copy(pcm, 0, normalized, 0, Math.Min(pcm.Length, targetSamples));
            return normalized;
        }

        private float[] ApplyDitherAndPreemphasis(float[] pcm)
        {
            var y = new float[pcm.Length];

            for (int i = 0; i < pcm.Length; i++)
            {
                float sample = pcm[i];

                if (EnableDither && Dither > 0)
                    sample += Dither * NextGaussian();

                float previous = i == 0
                    ? (_hasLastRawSample ? _lastRawSample : sample)
                    : pcm[i - 1];

                y[i] = sample - Preemphasis * previous;
            }

            _lastRawSample = pcm[^1];
            _hasLastRawSample = true;

            return y;
        }

        private float[] ExtractCurrentFrames(float[] signal, int targetFrames)
        {
            float[] output = new float[targetFrames * NMels];
            float[] frameBuffer = new float[NFft];
            Complex[] fft = new Complex[NFft];
            float[] powerSpectrum = new float[NFft / 2 + 1];

            int centerOffset = Center ? NFft / 2 : 0;

            for (int frame = 0; frame < targetFrames; frame++)
            {
                int frameStart = frame * HopLength;

                Array.Clear(frameBuffer);
                Array.Clear(fft);

                for (int i = 0; i < NFft; i++)
                {
                    int src = frameStart + i - centerOffset;
                    float sample = src >= 0 && src < signal.Length ? signal[src] : 0f;
                    float windowed = sample * _paddedWindow[i];
                    frameBuffer[i] = windowed;
                    fft[i] = new Complex(windowed, 0.0);
                }

                FftInPlace(fft);

                for (int k = 0; k < powerSpectrum.Length; k++)
                {
                    double mag = fft[k].Magnitude;
                    powerSpectrum[k] = MagPower == 1.0f
                        ? (float)mag
                        : (float)Math.Pow(mag, MagPower);
                }

                for (int mel = 0; mel < NMels; mel++)
                {
                    double energy = 0.0;
                    float[] filter = _melFilters[mel];

                    for (int k = 0; k < powerSpectrum.Length; k++)
                        energy += powerSpectrum[k] * filter[k];

                    output[frame * NMels + mel] = (float)Math.Log(Math.Max(energy, 0.0) + LogZeroGuardValue);
                }
            }

            return output;
        }

        private void UpdateFeatureCache(float[] currentFeatures)
        {
            if (PreEncodeCacheFrames == 0)
                return;

            int currentFrames = currentFeatures.Length / NMels;
            int framesToCopy = Math.Min(PreEncodeCacheFrames, currentFrames);
            int sourceFrame = currentFrames - framesToCopy;
            int destFrame = PreEncodeCacheFrames - framesToCopy;

            Array.Clear(_featureCache);
            Array.Copy(
                currentFeatures,
                sourceFrame * NMels,
                _featureCache,
                destFrame * NMels,
                framesToCopy * NMels);

            _hasFeatureCache = true;
        }

        private float NextGaussian()
        {
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = 1.0 - _rng.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        private static float[] BuildHann(int length)
        {
            var w = new float[length];
            if (length == 1)
            {
                w[0] = 1f;
                return w;
            }

            // Periodic Hann, matching common STFT behavior.
            for (int n = 0; n < length; n++)
                w[n] = 0.5f - 0.5f * (float)Math.Cos(2.0 * Math.PI * n / length);

            return w;
        }

        private static float[] PadWindowToFft(float[] window, int nFft)
        {
            if (window.Length > nFft)
                throw new ArgumentException("Window length cannot exceed FFT length.");

            var padded = new float[nFft];
            int offset = (nFft - window.Length) / 2;
            Array.Copy(window, 0, padded, offset, window.Length);
            return padded;
        }

        private static float[][] BuildMelFilterBank(int sampleRate, int nFft, int nMels, float fMin, float fMax)
        {
            int fftBins = nFft / 2 + 1;
            float[][] weights = new float[nMels][];
            for (int i = 0; i < nMels; i++)
                weights[i] = new float[fftBins];

            double minMel = HzToMelSlaney(fMin);
            double maxMel = HzToMelSlaney(fMax);

            double[] mels = Linspace(minMel, maxMel, nMels + 2);
            double[] melFreqs = mels.Select(MelToHzSlaney).ToArray();
            double[] fftFreqs = Linspace(0, sampleRate / 2.0, fftBins);

            for (int mel = 0; mel < nMels; mel++)
            {
                double left = melFreqs[mel];
                double center = melFreqs[mel + 1];
                double right = melFreqs[mel + 2];

                for (int k = 0; k < fftBins; k++)
                {
                    double freq = fftFreqs[k];
                    double lower = (freq - left) / Math.Max(center - left, 1e-12);
                    double upper = (right - freq) / Math.Max(right - center, 1e-12);
                    double value = Math.Max(0.0, Math.Min(lower, upper));

                    // Slaney-style area normalization.
                    double enorm = 2.0 / Math.Max(right - left, 1e-12);
                    weights[mel][k] = (float)(value * enorm);
                }
            }

            return weights;
        }

        private static double HzToMelSlaney(double hz)
        {
            const double fSp = 200.0 / 3.0;
            double mel = hz / fSp;

            const double minLogHz = 1000.0;
            const double minLogMel = minLogHz / fSp;
            double logStep = Math.Log(6.4) / 27.0;

            if (hz >= minLogHz)
                mel = minLogMel + Math.Log(hz / minLogHz) / logStep;

            return mel;
        }

        private static double MelToHzSlaney(double mel)
        {
            const double fSp = 200.0 / 3.0;
            double hz = mel * fSp;

            const double minLogHz = 1000.0;
            const double minLogMel = minLogHz / fSp;
            double logStep = Math.Log(6.4) / 27.0;

            if (mel >= minLogMel)
                hz = minLogHz * Math.Exp(logStep * (mel - minLogMel));

            return hz;
        }

        private static double[] Linspace(double start, double stop, int count)
        {
            var result = new double[count];
            if (count == 1)
            {
                result[0] = start;
                return result;
            }

            double step = (stop - start) / (count - 1);
            for (int i = 0; i < count; i++)
                result[i] = start + step * i;

            return result;
        }

        private static void FftInPlace(Complex[] buffer)
        {
            int n = buffer.Length;
            if ((n & (n - 1)) != 0)
                throw new ArgumentException("FFT length must be a power of two.");

            for (int i = 1, j = 0; i < n; i++)
            {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1)
                    j ^= bit;
                j ^= bit;

                if (i < j)
                    (buffer[i], buffer[j]) = (buffer[j], buffer[i]);
            }

            for (int len = 2; len <= n; len <<= 1)
            {
                double angle = -2.0 * Math.PI / len;
                Complex wLen = new(Math.Cos(angle), Math.Sin(angle));

                for (int i = 0; i < n; i += len)
                {
                    Complex w = Complex.One;
                    int half = len >> 1;

                    for (int j = 0; j < half; j++)
                    {
                        Complex u = buffer[i + j];
                        Complex v = buffer[i + j + half] * w;
                        buffer[i + j] = u + v;
                        buffer[i + j + half] = u - v;
                        w *= wLen;
                    }
                }
            }
        }

        private static int GetInt(JsonElement element, string name, int fallback)
        {
            if (!element.TryGetProperty(name, out var p))
                return fallback;

            if (p.ValueKind == JsonValueKind.Number && p.TryGetInt32(out int i))
                return i;

            if (p.ValueKind == JsonValueKind.String && int.TryParse(p.GetString(), out i))
                return i;

            return fallback;
        }

        private static float GetFloat(JsonElement element, string name, float fallback)
        {
            if (!element.TryGetProperty(name, out var p))
                return fallback;

            if (p.ValueKind == JsonValueKind.Number)
            {
                if (p.TryGetSingle(out float f))
                    return f;
                if (p.TryGetDouble(out double d))
                    return (float)d;
            }

            if (p.ValueKind == JsonValueKind.String && float.TryParse(p.GetString(), out float parsed))
                return parsed;

            return fallback;
        }

        private static bool GetBool(JsonElement element, string name, bool fallback)
        {
            if (!element.TryGetProperty(name, out var p))
                return fallback;

            if (p.ValueKind == JsonValueKind.True || p.ValueKind == JsonValueKind.False)
                return p.GetBoolean();

            if (p.ValueKind == JsonValueKind.String && bool.TryParse(p.GetString(), out bool parsed))
                return parsed;

            return fallback;
        }
    }
}
