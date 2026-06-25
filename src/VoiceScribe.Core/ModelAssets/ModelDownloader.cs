using System;
using System.Collections.Generic;
using System.Text;

namespace VoiceScribe.Core.ModelAssets
{
    /// <summary>
    /// Downloads models from HuggingFace repository and saves them to a local folder. It also provides a progress bar for each download.
    /// </summary>
    public class ModelDownloader
    {
        
        private readonly string _modelFolder;
        private readonly string _repoUrl;

        /// <summary>
        /// ctor
        /// </summary>
        /// <param name="repoUrl">Repo url (HugginFace)</param>
        /// <param name="modelFolder">Folder to store weight model files</param>        
        public ModelDownloader( string repoUrl, string modelFolder)
        {
            _modelFolder = modelFolder;
            _repoUrl = repoUrl;
        }


        /// <summary>
        /// The method will loop through the list and download each file, showing a progress bar for each download.
        /// </summary>
        /// <param name="modelFiles">List of files to download from the HuggingFace repository.</param>
        /// <returns></returns>
        public async Task HandleModelDownload(
            HttpClient httpClient,
            IEnumerable<string> modelFiles,
            CancellationToken cancellationToken = default)
        {
            Directory.CreateDirectory(_modelFolder);
            Console.WriteLine("\nStarting automated download loop...");

            foreach (var file in modelFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string fileUrl = $"{_repoUrl}/{file}";
                Console.WriteLine($"\nFetching: {file}");
                await DownloadWithProgress(
                    httpClient,
                    fileUrl,
                    file,
                    cancellationToken: cancellationToken);
            }
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n[Success] All streaming assets downloaded successfully.");
            Console.ResetColor();
        }


        /// <summary>
        /// Downloads each file and shows a progress bar in the console. If the file already exists, it will skip the download unless overwrite is set to true.
        /// </summary>
        /// <param name="url"></param>
        /// <param name="fileName"></param>
        /// <param name="overwrite"></param>
        /// <returns></returns>
        private async Task DownloadWithProgress(
            HttpClient httpClient,
            string url,
            string fileName,
            bool overwrite = false,
            CancellationToken cancellationToken = default)
        {
            string destinationPath = Path.Combine(_modelFolder, fileName);

            if (File.Exists(destinationPath))
            {
                if (overwrite)
                {
                    File.Delete(destinationPath);
                }
                else
                {
                    Console.WriteLine($"File '{destinationPath}' already exists. Skipping download.");
                    return;
                }
            }
            ;


            try
            {
                using (var response = await httpClient.GetAsync(
                    url,
                    HttpCompletionOption.ResponseHeadersRead,
                    cancellationToken))
                {
                    response.EnsureSuccessStatusCode();
                    long? totalBytes = response.Content.Headers.ContentLength;

                    using (var downloadStream =
                        await response.Content.ReadAsStreamAsync(cancellationToken))
                    using (var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true))
                    {
                        var buffer = new byte[8192];
                        long totalReadBytes = 0;
                        int readBytes;

                        while ((readBytes = await downloadStream.ReadAsync(
                            buffer.AsMemory(),
                            cancellationToken)) > 0)
                        {
                            await fileStream.WriteAsync(
                                buffer.AsMemory(0, readBytes),
                                cancellationToken);
                            totalReadBytes += readBytes;

                            if (totalBytes.HasValue)
                            {
                                double progressPercentage =
                                    (double)totalReadBytes / totalBytes.Value * 100;
                                DrawProgressBar(progressPercentage);
                            }
                        }
                    }
                }
            }
            catch
            {
                if (File.Exists(destinationPath))
                    File.Delete(destinationPath);

                throw;
            }
        }


        /// <summary>
        /// Draws a simple progress bar in the console based on the percentage of completion. The progress bar consists of filled blocks (█) representing completed progress and empty blocks (-) representing remaining progress. The percentage is displayed at the end of the progress bar.
        /// </summary>
        /// <param name="percentage"></param>
        private void DrawProgressBar(double percentage)
        {
            int blockCount = (int)(percentage / 2);
            string progressBlocks = new string('█', blockCount);
            string emptyBlocks = new string('-', 50 - blockCount);

            Console.Write($"\r[{progressBlocks}{emptyBlocks}] {percentage:0.0}%");
        }


        /// <summary>
        /// Verifies if all the required model files exist in the local model folder. If any file is missing, it returns false; otherwise, it returns true.
        /// </summary>
        /// <param name="modelFiles">List of files to verify</param>
        /// <returns></returns>
        public bool VerifyLocalWeights(IEnumerable<string> modelFiles)
        {
            if (!Directory.Exists(_modelFolder)) return false;
            return modelFiles.All(file => File.Exists(Path.Combine(_modelFolder, file)));
        }
    }
}
