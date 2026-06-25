using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.Logging;
using VoiceScribe.Core.ModelAssets;

namespace VoiceScribe.Core.Configuration
{

    /// <summary>
    /// Configuration class for the speech application. It defines properties for model file management and provides a method to load
    /// configuration from a JSON file, with fallback to default values if the file is missing or invalid.
    /// </summary>
    public class VoiceAppConfig
    {
        /// <summary>
        /// Path where the model files will be downloaded and stored. If not specified, it defaults to a "NemotronWeights" folder in the application's base directory.
        /// </summary>
        public string? ModelDownloadsPath { get; set; } 


        /// <summary>
        /// List of model files required for the application. This should include the names of the files that the ModelDownloader will check for and download if missing. If not specified, it defaults to a predefined list of expected model files.
        /// </summary>
        public List<string> ModelFiles { get; set; } = NemotronModelFiles.CreateRequiredFileList();


        /// <summary>
        /// URL of the repository where the model files can be downloaded from. This should point to a valid location where the ModelDownloader can access the required files. If not specified, it defaults to a predefined URL.
        /// </summary>
        public string RepoUrl { get; set; } = "";

        /// <summary>
        /// Audio capture and silence filtering settings.
        /// </summary>
        public AudioCaptureOptions Audio { get; set; } = new();

        /// <summary>
        /// Nemotron inference and RNN-T decoding settings.
        /// </summary>
        public NemotronModelOptions Nemotron { get; set; } = new();


        /// <summary>
        /// Creates a SpeechAppConfig instance by reading and deserializing a JSON file. If the file is missing or invalid, it returns a default configuration instance.
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public static async Task<VoiceAppConfig?> FromJsonFileAsync(ILogger logger, string fileName, VoiceAppConfig? defaultConfig = null)
        {
            if (!File.Exists(fileName))
            {                
                logger.LogWarning($"Config file '{fileName}' not found. Using default configuration.");
                return defaultConfig;
            }

            VoiceAppConfig config = null!;
            try
            {
                using (FileStream configStream = new FileStream(fileName, FileMode.Open, FileAccess.Read))
                    config = await System.Text.Json.JsonSerializer.DeserializeAsync<VoiceAppConfig>(configStream) ?? new VoiceAppConfig();
            }
            catch (Exception ex)
            {
                logger.LogError(ex, $"Failed to read or parse config file: {ex.Message}. Using default configuration.");
                return defaultConfig;
            }

            return config;
        }
    }
}
