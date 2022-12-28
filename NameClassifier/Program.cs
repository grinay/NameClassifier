// See https://aka.ms/new-console-template for more information

using Common;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Text;
using NameClassifier;
using NameClassifier.dataset;

var cleanedFirstNameFile = "dataset/cleaned-first-names.json";
var cleanedLastNameFile = "dataset/cleaned-last-names.json";
var firstNamesFile = "dataset/first_names.json";
var lastNamesFile = "dataset/last_names.json";


DatasetParser.CleanNames(firstNamesFile, cleanedFirstNameFile);
DatasetParser.CleanNames(lastNamesFile, cleanedLastNameFile);

var firstNames = File.ReadAllLines(cleanedFirstNameFile);
var lastNames = File.ReadAllLines(cleanedLastNameFile);

using var loggerFactory = LoggerFactory.Create(builder =>
{
    builder
        .AddFilter("Microsoft", LogLevel.Debug)
        .AddFilter("System", LogLevel.Debug)
        .AddConsole();
});
ILogger logger = loggerFactory.CreateLogger<Program>();
// Initialize MLContext
var mlContext = new MLContext()
{
    GpuDeviceId = 0,
};

// Load your data
var names = firstNames
    .Select(x => new NameInput(x))
    .Concat(lastNames.Select(x => new NameInput(x, true)))
    .ToArray();


var namesDataView = mlContext.Data.LoadFromEnumerable(names);

var dataSplit = mlContext.Data.TrainTestSplit(namesDataView, testFraction: 0.2, seed: Random.Shared.Next());
var trainData = dataSplit.TrainSet;
var testData = dataSplit.TestSet;

// if (!File.Exists("model.zip"))
// {
// var pipeline =
//     mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
//         // .Append(
//         // mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized"))
//         .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
//             labelColumnName: "Label",
//             sentence1ColumnName: "Name",
//             architecture: BertArchitecture.Roberta,
//             maxEpochs: 3,
//             batchSize: 1500))
//         .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"))
//         .AppendCacheCheckpoint(mlContext);


// var pipeline =
//     mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized")
//         .Append(mlContext.BinaryClassification.Trainers.FastForest(featureColumnName: "NameFeaturized",
//             labelColumnName: "Label", numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 20))
//         .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
//         .AppendCacheCheckpoint(mlContext);

//THE BEST SO FAR  
//Accuracy: 0.7149557927407334
// F1 Score : 0.7638783508841452
//ALL Manual tests correct
var pipeline =
    // mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized")
    mlContext.Transforms.Text.NormalizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized",
            keepDiacritics: false, keepNumbers: false, keepPunctuations: false)
        .Append(mlContext.Transforms.Text.ProduceWordBags(inputColumnName: "NameFeaturized",
            outputColumnName: "NameFeaturized", weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf))
        // .Append(mlContext.Transforms.Conversion.ConvertType("NameFeaturized", "NameFeaturized",DataKind.String))
        // .Append(mlContext.Transforms.Text.TokenizeIntoWords("NameFeaturized", "NameFeaturized"))
        // .Append(mlContext.Transforms.Text. (inputColumnName: "NameFeaturized", outputColumnName: "NameFeaturized"))
        // .Append(mlContext.Transforms.Text.ApplyWordEmbedding(inputColumnName: "NameFeaturized",
        //     outputColumnName: "NameFeaturized",
        //     modelKind: WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D))
        // .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "NameFeaturized", outputColumnName: "NameFeaturized"))
        // .Append(mlContext.Transforms.Text.ApplyWordEmbedding(inputColumnName: "NameFeaturized",
        //     outputColumnName: "NameFeaturized",
        //     modelKind: WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D))
        // .Append(mlContext.Transforms.Text.TokenizeIntoCharactersAsKeys(inputColumnName:"NameFeaturized",outputColumnName:"NameFeaturized"))
        // .Append(mlContext.BinaryClassification.Trainers.LinearSvm())
        .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
            new LbfgsLogisticRegressionBinaryTrainer.Options()
            {
                FeatureColumnName = "NameFeaturized",
                LabelColumnName = "Label",
                HistorySize = 50,
                OptimizationTolerance = 1E-09f,
                DenseOptimizer = true,
            }))
        .AppendCacheCheckpoint(mlContext);
//SdcaLogisticRegression - bad results
//AveragedPerceptron - bad results
// var pipeline =
//     mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized")
//         .Append(mlContext.BinaryClassification.Trainers.Gam(featureColumnName: "NameFeaturized",
//             labelColumnName: "Label",
//             numberOfIterations: 20))
//         .AppendCacheCheckpoint(mlContext);


// var model = pipeline.Fit(trainData);
var model = pipeline.Fit(namesDataView);
mlContext.Model.Save(model, trainData.Schema, "model.zip");
// }

var trainedModel = mlContext.Model.Load("model.zip", out var schema);

// // Evaluate the model's performance against the TEST data set
Console.WriteLine("Evaluating model performance...");

// We need to apply the same transformations to our test set so it can be evaluated via the resulting model
var transformedTest = trainedModel.Transform(testData);

var metrics =
    mlContext.BinaryClassification.EvaluateNonCalibrated(transformedTest);

// Display Metrics
Console.WriteLine($"Accuracy: {metrics.Accuracy}");
Console.WriteLine($"F1 Score : {metrics.F1Score}");

// Console.WriteLine($"Log Loss: {metrics.}");
Console.WriteLine();

// Generate the table for diagnostics
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());


// Create PredictionEngines
var predictionEngine = mlContext.Model.CreatePredictionEngine<NameInput, NamePredictor>(trainedModel);

Console.WriteLine(predictionEngine.Predict(new NameInput("grinevskiy")).PredictedLabel); //true
Console.WriteLine(predictionEngine.Predict(new NameInput("bechmann")).PredictedLabel); //true
Console.WriteLine(predictionEngine.Predict(new NameInput("snow")).PredictedLabel); //true
Console.WriteLine(predictionEngine.Predict(new NameInput("alshkili")).PredictedLabel); //true
Console.WriteLine(predictionEngine.Predict(new NameInput("raikhanova")).PredictedLabel); //true
Console.WriteLine(predictionEngine.Predict(new NameInput("mask")).PredictedLabel); //true
Console.WriteLine(predictionEngine.Predict(new NameInput("levay")).PredictedLabel); //true
Console.WriteLine(predictionEngine.Predict(new NameInput("aleksander")).PredictedLabel); //false
Console.WriteLine(predictionEngine.Predict(new NameInput("elena")).PredictedLabel); //false
Console.WriteLine(predictionEngine.Predict(new NameInput("john")).PredictedLabel); //false
Console.WriteLine(predictionEngine.Predict(new NameInput("amanda")).PredictedLabel); //false

// PredictiIsFirstNameonEngine<>