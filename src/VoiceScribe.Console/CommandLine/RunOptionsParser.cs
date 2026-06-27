namespace VoiceScribe.Console.CommandLine;

internal static class RunOptionsParser
{
    internal const int DefaultBenchmarkChunks = 20;

    internal static RunOptions Parse(string[] args)
    {
        if (args.Length == 0)
            return new RunOptions(false, DefaultBenchmarkChunks, null);

        if (!string.Equals(
            args[0],
            "--benchmark",
            StringComparison.OrdinalIgnoreCase))
        {
            return new RunOptions(
                false,
                DefaultBenchmarkChunks,
                args[0]);
        }

        int chunks = DefaultBenchmarkChunks;
        if (args.Length > 1 &&
            (!int.TryParse(args[1], out chunks) || chunks <= 0))
        {
            throw new ArgumentException(
                "--benchmark expects a positive chunk count.");
        }

        return new RunOptions(true, chunks, null);
    }
}
