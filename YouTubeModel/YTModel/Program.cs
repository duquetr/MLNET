namespace YTModel
{
    using System;
    using System.Linq;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Transforms;

    class Program
    {
        private static readonly string dataPath = "./country_vaccinations.csv";

        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<VaccineData>(dataPath, hasHeader: true);

            data = ReplaceEmptyValues(context, data, "Total_Vaccinations");
            data = ReplaceEmptyValues(context, data, "People_Vaccinated");
            data = ReplaceEmptyValues(context, data, "People_Vaccinated_Per_Hundred");
            data = ReplaceEmptyValues(context, data, "People_Fully_Vaccinated");

            var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var features = split.TrainSet.Schema
                           .Select(col => col.Name)
                           .Where(colName =>
                                    colName != "Country")
                           .ToArray();

            var pipeline = context.Transforms.Conversion
                    .MapValueToKey(inputColumnName: "Country", outputColumnName: "Label")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(split.TrainSet);

            var testDataView = context.Data.LoadFromTextFile<VaccineCountry>(dataPath, hasHeader: true);
            var testMetrics = context.MulticlassClassification.Evaluate(model.Transform(split.TestSet));

            Console.WriteLine($"confusion matrix - {testMetrics.ConfusionMatrix}");
        }

        static private IDataView ReplaceEmptyValues(MLContext context, IDataView inputData, string columnName)
        {
            // Define replacement estimator
            var replacementEstimator = context.Transforms.ReplaceMissingValues(columnName, replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean);

            // Fit data to estimator
            // Fitting generates a transformer that applies the operations of defined by estimator
            ITransformer replacementTransformer = replacementEstimator.Fit(inputData);

            // Transform data
            return replacementTransformer.Transform(inputData);
        }
    }
}
