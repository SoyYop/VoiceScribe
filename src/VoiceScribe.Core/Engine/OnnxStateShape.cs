namespace VoiceScribe.Core.Engine
{
    internal static class OnnxStateShape
    {
        internal static int[] ResolveInitialDimensions(
            IReadOnlyList<int> metadataDimensions,
            string inputName)
        {
            if (metadataDimensions.Count == 0)
                throw new InvalidOperationException(
                    $"Required ONNX input '{inputName}' has no tensor dimensions.");

            int[] dimensions = metadataDimensions.ToArray();
            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] <= 0)
                    dimensions[i] = 1;
            }

            return dimensions;
        }
    }
}
