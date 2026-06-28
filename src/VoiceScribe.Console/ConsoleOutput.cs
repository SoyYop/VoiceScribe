namespace VoiceScribe.Console;

internal static class ConsoleOutput
{
    internal static void Write(string message, ConsoleColor color)
    {
        ConsoleColor previousColor = System.Console.ForegroundColor;
        try
        {
            System.Console.ForegroundColor = color;
            System.Console.Write(message);
        }
        finally
        {
            System.Console.ForegroundColor = previousColor;
        }
    }

    internal static void WriteLine(string message, ConsoleColor color)
    {
        ConsoleColor previousColor = System.Console.ForegroundColor;
        try
        {
            System.Console.ForegroundColor = color;
            System.Console.WriteLine(message);
        }
        finally
        {
            System.Console.ForegroundColor = previousColor;
        }
    }
}
