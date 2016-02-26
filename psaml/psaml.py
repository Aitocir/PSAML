#
# Iris Analysis
#
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer


def do_continuous_input_analysis(sc, model, exp_sensitivity, ctrl_sensitivity, data_info):
    # -1) create SQLContext
    sql_context = SQLContext(sc)

    # ##########################################################################################################
    #
    # 0) Verify input
    #
    #  assert expSensitivity > 0 (int)
    #  assert ctrlSensitivity > 0 (int)
    #  assert dataInfo (DataFrame of the following format, one row for each column in the data model works on):
    #
    #                                         DataFrame of Data columns
    #                     _________________________________________________________________
    # Column purpose     | colName   | minValue  | maxValue  | shouldAnalyze   | isClass   |
    #                    |-----------|-----------|-----------|-----------------|-----------|
    # Column type        | string    | numeral   | numeral   | boolean         | boolean   |
    #                    |-----------|-----------|-----------|-----------------|-----------|
    # Example record     | "petalW"  | 4.3       | 7.9       | true            | false     |
    #                    |___________|___________|___________|_________________|___________|
    #
    try:
        assert (exp_sensitivity > 0), "Experiment Sensitivity must be a positive integer"
        assert (ctrl_sensitivity > 0), "Control Variable Sensitivity must be a positive integer"
    except AssertionError as e:
        raise ValueError(e.args)

    # ##########################################################################################################
    #
    # 1) Generate test data
    #
    #  TODO: generate test data based on a collection of mins, maxs
    #
    #  testData = a new dataframe, whose column names = all values from the colName col from dataInfo
    #  for ( x : 0 ... ctrlSensitivity ), inclusive
    #     foreach ( varCol : varCol.shouldAnalyze == true )
    #        for ( y : 0 ... expSensitivity ), inclusive
    #           load record into testData
    #           #  set all values to minValue + ((maxValue - minValue) * (x / ctrlSensitivity))
    #           #  manually set varCol to minValue + ((maxValue - minValue) * (y / ctrlSensitivity))
    #           #  value loaded into the class column does NOT matter
    #

    # ##########################################################################################################
    #
    # 2) Make predictions.
    #
    # predictions = model.transform(testData)  #  but, we're not passing in dataInfo yet, so we'll treat
    # dataInfo like already done testData
    predictions = model.transform(data_info)

    # ##########################################################################################################
    #
    # 3) Transform predictions into output DataFrame
    #
    #  Output DataFrame should use the following format:
    #
    #                                            dataframe name
    #                     _______________________________________________________________
    # Column purpose     | prediction   | varColName   | expVariance   | ctrlVariance    |
    #                    |--------------|--------------|---------------|-----------------|
    # Column types       | <classType>  | string       | num (0.0-1.0) | num (0.0-1.0)   |
    #                    |--------------|--------------|---------------|-----------------|
    # Example record     | "iris-setosa"| "PetalW"     | 0.7           | 0.2             |
    #                    |______________|______________|_______________|_________________|
    #
    # In the above example record, we get "iris-setosa" as a prediction when holding "PetalW" at 70% of potential
    # value, and everything else at 20%
    #
    #     varianceData = new DataFrame after above format
    #     for ( x : 0 ... ctrlSensitivity ), inclusive
    #        foreach ( varCol : varCol.shouldAnalyze == true )
    #           for ( y : 0 ... expSensitivity ), inclusive
    #              translate row from predictions[n] to  varianceData[n]
    #              #  they will end up being the same size
    #

    # return varianceData  # but for now, just return predictions so the code actually interprets
    return predictions
