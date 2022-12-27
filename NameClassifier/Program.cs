// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using NameClassifier.dataset;

var cleanedFirstNameFile = "dataset/cleaned-first-names.json";
var cleanedLastNameFile = "dataset/cleaned-last-names.json";
var firstNamesFile = "dataset/first_names.json";
var lastNamesFile = "dataset/last_names.json";


DatasetParser.CleanNames(firstNamesFile, cleanedFirstNameFile);
DatasetParser.CleanNames(lastNamesFile, cleanedLastNameFile);

var firstNames = File.ReadAllLines(cleanedFirstNameFile);
var lastNames = File.ReadAllLines(cleanedLastNameFile);

// Initialize MLContext
var mlContext = new MLContext();

// Load your data
var names = firstNames
    .Select(x => new { Sentence = x, Label = "firstname" })
    .Concat(lastNames.Select(x => new { Sentence = x, Label = "lastname" })).ToArray();


var namesDataView = mlContext.Data.LoadFromEnumerable(names);

var dataSplit = mlContext.Data.TrainTestSplit(namesDataView, testFraction: 0.2);
var trainData = dataSplit.TrainSet;
var testData = dataSplit.TestSet;


//Define your training pipeline
var pipeline =
    mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
        .Append(mlContext.MulticlassClassification.Trainers.TextClassification(sentence1ColumnName: "Sentence"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

Console.WriteLine("Start training");

// Train the model
var model = pipeline.Fit(trainData);

// Evaluate the model's performance against the TEST data set
Console.WriteLine("Evaluating model performance...");

// We need to apply the same transformations to our test set so it can be evaluated via the resulting model
var transformedTest = model.Transform(testData);
var metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

// Display Metrics
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");
Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");
Console.WriteLine($"Log Loss: {metrics.LogLoss}");
Console.WriteLine();

// Generate the table for diagnostics
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());