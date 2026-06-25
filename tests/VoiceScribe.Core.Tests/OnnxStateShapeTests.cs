using VoiceScribe.Core.Engine;

namespace VoiceScribe.Core.Tests;

public sealed class OnnxStateShapeTests
{
    [Fact]
    public void ResolveInitialDimensions_ResolvesDynamicTargetsSequence()
    {
        int[] dimensions =
            OnnxStateShape.ResolveInitialDimensions([1, -1], "targets");

        Assert.Equal([1, 1], dimensions);
    }

    [Fact]
    public void ResolveInitialDimensions_PreservesStaticStateDimensions()
    {
        int[] dimensions =
            OnnxStateShape.ResolveInitialDimensions([2, 1, 640], "h_in");

        Assert.Equal([2, 1, 640], dimensions);
    }

    [Fact]
    public void ResolveInitialDimensions_RejectsScalarState()
    {
        Assert.Throws<InvalidOperationException>(
            () => OnnxStateShape.ResolveInitialDimensions([], "targets"));
    }
}
