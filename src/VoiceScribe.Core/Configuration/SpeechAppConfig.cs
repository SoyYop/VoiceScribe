using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.Logging;

namespace VoiceScribe.Core.Configuration
{

    /// <summary>
    /// Configuration class for the speech application. It defines properties for model file management and provides a method to load configuration from a JSON file, with fallback to default values if the file is missing or invalid.
    /// </summary>
    public class SpeechAppConfig
    {
        /// <summary>
        /// Path where the model files will be downloaded and stored. If not specified, it defaults to a "NemotronWeights" folder in the application's base directory.
        /// </summary>
        public string? ModelDownloadsPath { get; set; } 


        /// <summary>
        /// List of model files required for the application. This should include the names of the files that the ModelDownloader will check for and download if missing. If not specified, it defaults to a predefined list of expected model files.
        /// </summary>
        public List<string> ModelFiles {get;set;} = [];


        /// <summary>
        /// URL of the repository where the model files can be downloaded from. This should point to a valid location where the ModelDownloader can access the required files. If not specified, it defaults to a predefined URL.
        /// </summary>
        public string RepoUrl { get; set; } = "";


        /// <summary>
        /// Creates a SpeechAppConfig instance by reading and deserializing a JSON file. If the file is missing or invalid, it returns a default configuration instance.
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public static async Task<SpeechAppConfig?> FromJsonFileAsync(ILogger logger, string fileName, SpeechAppConfig? defaultConfig = null)
        {
            if (!File.Exists(fileName))
            {                
                logger.LogWarning($"Config file '{fileName}' not found. Using default configuration.");
                return defaultConfig;
            }

            SpeechAppConfig config = null!;
            try
            {
                using (FileStream configStream = new FileStream(fileName, FileMode.Open, FileAccess.Read))
                    config = await System.Text.Json.JsonSerializer.DeserializeAsync<SpeechAppConfig>(configStream) ?? new SpeechAppConfig();
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
