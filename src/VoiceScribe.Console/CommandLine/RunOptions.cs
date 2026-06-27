namespace VoiceScribe.Console.CommandLine;

internal sealed record RunOptions(
    bool Benchmark,
    int BenchmarkChunks,
    string? TranscriptPath);
