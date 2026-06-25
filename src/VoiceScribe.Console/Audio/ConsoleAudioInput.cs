using NAudio.Wave;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Console.Audio;

internal sealed record AudioInputDevice(int Number, string ProductName, int Channels);

internal static class ConsoleAudioInput
{
    internal static IReadOnlyList<AudioInputDevice> GetDevices()
    {
        var devices = new List<AudioInputDevice>(WaveIn.DeviceCount);
        for (int i = 0; i < WaveIn.DeviceCount; i++)
        {
            WaveInCapabilities capabilities = WaveIn.GetCapabilities(i);
            devices.Add(new AudioInputDevice(
                i,
                capabilities.ProductName,
                capabilities.Channels));
        }

        return devices;
    }

    internal static int SelectDeviceNumber(
        IReadOnlyList<AudioInputDevice> devices,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (devices.Count == 0)
            throw new InvalidOperationException("No input microphones found.");

        if (devices.Count == 1)
        {
            WriteSelectedDevice(devices[0]);
            return devices[0].Number;
        }

        System.Console.ForegroundColor = ConsoleColor.Yellow;
        System.Console.WriteLine("\n[Audio] Multiple input microphones found:");
        System.Console.ResetColor();

        foreach (AudioInputDevice device in devices)
        {
            System.Console.WriteLine(
                $"  {device.Number}: {device.ProductName} - Channels: {device.Channels}");
        }

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            System.Console.ForegroundColor = ConsoleColor.Yellow;
            System.Console.Write("\nSelect microphone number: ");
            System.Console.ResetColor();

            string? input = System.Console.ReadLine();
            cancellationToken.ThrowIfCancellationRequested();

            if (int.TryParse(input, out int number))
            {
                AudioInputDevice? selected = devices.FirstOrDefault(
                    device => device.Number == number);
                if (selected != null)
                {
                    WriteSelectedDevice(selected);
                    return selected.Number;
                }
            }

            System.Console.ForegroundColor = ConsoleColor.Red;
            System.Console.WriteLine("[Audio] Invalid microphone number. Try again.");
            System.Console.ResetColor();
        }
    }

    internal static WaveInEvent CreateWaveSource(
        int deviceNumber,
        AudioCaptureOptions options) =>
        new()
        {
            DeviceNumber = deviceNumber,
            WaveFormat = new WaveFormat(
                options.SampleRate,
                options.BitsPerSample,
                options.Channels),
            BufferMilliseconds = options.BufferMilliseconds
        };

    private static void WriteSelectedDevice(AudioInputDevice device)
    {
        System.Console.ForegroundColor = ConsoleColor.Green;
        System.Console.WriteLine(
            $"[Audio] Microphone selected: {device.Number} - {device.ProductName}");
        System.Console.ResetColor();
    }
}
