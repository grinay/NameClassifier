using Newtonsoft.Json.Linq;

namespace NameClassifier.dataset;

public static class DatasetParser
{
    public static void CleanNames(string inputPath, string outputPath)
    {
        if (File.Exists(outputPath))
            return;

        var names = JToken.Parse(File.ReadAllText(inputPath)).Select(x => ((JProperty)x).Name);
        File.WriteAllLines(outputPath, names);
    }
}