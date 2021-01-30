namespace YTModel
{
    using System;
    using Microsoft.ML.Data;

    public class VaccineData
    {
        [LoadColumn(0)]
        public string Country { get; set; }

        [LoadColumn(3)]
        public float Total_Vaccinations { get; set; }

        [LoadColumn(4)]
        public float People_Vaccinated { get; set; }

        [LoadColumn(5)]
        public float People_Fully_Vaccinated { get; set; }

        [LoadColumn(9)]
        public float People_Vaccinated_Per_Hundred { get; set; }
    }

    public class VaccineCountry
    {
        [ColumnName("PredictedLabel")]
        public string Country { get; set; }
    }
}
