using Microsoft.ML.Data;

namespace Common;

public class NamePredictor
{
    // [ColumnName("Label")] 
    public bool PredictedLabel;
}

public record NameInput(string Name, bool Label = false);