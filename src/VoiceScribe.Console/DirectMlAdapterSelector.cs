using System.Runtime.InteropServices;
using VoiceScribe.Console;
using VoiceScribe.Core.Configuration;

internal sealed record DirectMlAdapter(
    int DeviceId,
    string Description,
    ulong DedicatedVideoMemoryMiB,
    ulong SharedSystemMemoryMiB,
    uint VendorId,
    uint DeviceVendorId);

internal static class DirectMlAdapterSelector
{
    private const int DxgiErrorNotFound = unchecked((int)0x887A0002);
    private const uint DxgiAdapterFlagSoftware = 2;

    internal static bool IsDirectMlRequested(OnnxRuntimeOptions options) =>
        options.GetEncoderProvider() == OnnxExecutionProvider.DirectMl ||
        options.GetDecoderProvider() == OnnxExecutionProvider.DirectMl ||
        options.GetJoinerProvider() == OnnxExecutionProvider.DirectMl;

    internal static IReadOnlyList<DirectMlAdapter> GetAdapters()
    {
        var factoryId = typeof(IDXGIFactory1).GUID;
        int result = CreateDXGIFactory1(
            ref factoryId,
            out IDXGIFactory1? factory);

        if (result < 0 || factory is null)
            return [];

        try
        {
            var adapters = new List<DirectMlAdapter>();
            for (uint i = 0; ; i++)
            {
                result = factory.EnumAdapters1(i, out IDXGIAdapter1? adapter);
                if (result == DxgiErrorNotFound)
                    break;
                if (result < 0 || adapter is null)
                    continue;

                try
                {
                    adapter.GetDesc1(out DxgiAdapterDesc1 description);
                    if ((description.Flags & DxgiAdapterFlagSoftware) != 0)
                        continue;

                    adapters.Add(new DirectMlAdapter(
                        checked((int)i),
                        description.Description.TrimEnd('\0'),
                        ToMiB(description.DedicatedVideoMemory),
                        ToMiB(description.SharedSystemMemory),
                        description.VendorId,
                        description.DeviceId));
                }
                finally
                {
                    Marshal.FinalReleaseComObject(adapter);
                }
            }

            return adapters;
        }
        finally
        {
            Marshal.FinalReleaseComObject(factory);
        }
    }

    internal static int SelectDeviceId(
        IReadOnlyList<DirectMlAdapter> adapters,
        int configuredDeviceId,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (adapters.Count == 0)
        {
            ConsoleOutput.WriteLine(
                "[Inference] No DirectML adapters were enumerated. " +
                $"Keeping configured DeviceId={configuredDeviceId}.",
                ConsoleColor.Yellow);
            return configuredDeviceId;
        }

        if (adapters.Count == 1)
        {
            WriteSelectedAdapter(adapters[0]);
            return adapters[0].DeviceId;
        }

        ConsoleOutput.WriteLine(
            "\n[Inference] Multiple DirectML adapters found:",
            ConsoleColor.Yellow);

        foreach (DirectMlAdapter adapter in adapters)
        {
            string selected = adapter.DeviceId == configuredDeviceId
                ? " [configured]"
                : "";
            System.Console.WriteLine(
                $"  {adapter.DeviceId}: {adapter.Description} " +
                $"- Dedicated: {adapter.DedicatedVideoMemoryMiB} MiB, " +
                $"Shared: {adapter.SharedSystemMemoryMiB} MiB{selected}");
        }

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            ConsoleOutput.Write(
                $"\nSelect DirectML device number [{configuredDeviceId}]: ",
                ConsoleColor.Yellow);

            string? input = System.Console.ReadLine();
            cancellationToken.ThrowIfCancellationRequested();

            if (string.IsNullOrWhiteSpace(input))
            {
                DirectMlAdapter? configured = adapters.FirstOrDefault(
                    adapter => adapter.DeviceId == configuredDeviceId);
                if (configured is not null)
                {
                    WriteSelectedAdapter(configured);
                    return configured.DeviceId;
                }
            }

            if (int.TryParse(input, out int number))
            {
                DirectMlAdapter? selected = adapters.FirstOrDefault(
                    adapter => adapter.DeviceId == number);
                if (selected is not null)
                {
                    WriteSelectedAdapter(selected);
                    return selected.DeviceId;
                }
            }

            ConsoleOutput.WriteLine(
                "[Inference] Invalid DirectML device number. Try again.",
                ConsoleColor.Red);
        }
    }

    private static void WriteSelectedAdapter(DirectMlAdapter adapter)
    {
        ConsoleOutput.WriteLine(
            $"[Inference] DirectML device selected: {adapter.DeviceId} - {adapter.Description}",
            ConsoleColor.Green);
    }

    private static ulong ToMiB(UIntPtr bytes) =>
        bytes.ToUInt64() / 1024 / 1024;

    [DllImport("dxgi.dll")]
    private static extern int CreateDXGIFactory1(
        ref Guid riid,
        [MarshalAs(UnmanagedType.Interface)] out IDXGIFactory1? ppFactory);

    [ComImport]
    [Guid("770aae78-f26f-4dba-a829-253c83d1b387")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIFactory1
    {
        [PreserveSig] int SetPrivateData(ref Guid name, uint dataSize, IntPtr data);
        [PreserveSig] int SetPrivateDataInterface(ref Guid name, IntPtr unknown);
        [PreserveSig] int GetPrivateData(ref Guid name, ref uint dataSize, IntPtr data);
        [PreserveSig] int GetParent(ref Guid riid, out IntPtr parent);
        [PreserveSig] int EnumAdapters(uint adapter, out IntPtr adapterInterface);
        [PreserveSig] int MakeWindowAssociation(IntPtr windowHandle, uint flags);
        [PreserveSig] int GetWindowAssociation(out IntPtr windowHandle);
        [PreserveSig] int CreateSwapChain(IntPtr device, IntPtr swapChainDescription, out IntPtr swapChain);
        [PreserveSig] int CreateSoftwareAdapter(IntPtr module, out IntPtr adapter);
        [PreserveSig] int EnumAdapters1(uint adapter, [MarshalAs(UnmanagedType.Interface)] out IDXGIAdapter1? adapterInterface);
        [PreserveSig] int IsCurrent();
    }

    [ComImport]
    [Guid("29038f61-3839-4626-91fd-086879011a05")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIAdapter1
    {
        [PreserveSig] int SetPrivateData(ref Guid name, uint dataSize, IntPtr data);
        [PreserveSig] int SetPrivateDataInterface(ref Guid name, IntPtr unknown);
        [PreserveSig] int GetPrivateData(ref Guid name, ref uint dataSize, IntPtr data);
        [PreserveSig] int GetParent(ref Guid riid, out IntPtr parent);
        [PreserveSig] int EnumOutputs(uint output, out IntPtr outputInterface);
        [PreserveSig] int GetDesc(out DxgiAdapterDesc description);
        [PreserveSig] int CheckInterfaceSupport(ref Guid interfaceName, out long umdVersion);
        [PreserveSig] int GetDesc1(out DxgiAdapterDesc1 description);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Luid
    {
        public uint LowPart;
        public int HighPart;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct DxgiAdapterDesc
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string Description;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public UIntPtr DedicatedVideoMemory;
        public UIntPtr DedicatedSystemMemory;
        public UIntPtr SharedSystemMemory;
        public Luid AdapterLuid;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct DxgiAdapterDesc1
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string Description;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public UIntPtr DedicatedVideoMemory;
        public UIntPtr DedicatedSystemMemory;
        public UIntPtr SharedSystemMemory;
        public Luid AdapterLuid;
        public uint Flags;
    }
}
