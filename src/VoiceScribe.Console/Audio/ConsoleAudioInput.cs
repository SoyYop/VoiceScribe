using NAudio.Wave;
using VoiceScribe.Console;
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

        ConsoleOutput.WriteLine("\n[Audio] Multiple input microphones found:", ConsoleColor.Yellow);

        foreach (AudioInputDevice device in devices)
        {
            System.Console.WriteLine(
                $"  {device.Number}: {device.ProductName} - Channels: {device.Channels}");
        }

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            ConsoleOutput.Write("\nSelect microphone number: ", ConsoleColor.Yellow);

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

            ConsoleOutput.WriteLine("[Audio] Invalid microphone number. Try again.", ConsoleColor.Red);
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
        ConsoleOutput.WriteLine(
            $"[Audio] Microphone selected: {device.Number} - {device.ProductName}",
            ConsoleColor.Green);
    }
}
