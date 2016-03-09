#
# Iris Analysis
#
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer


# Helper: create data_info DataFrame from sample data
def make_data_info(sql, sample_data, cols_analyze, col_class):
    "create the data_info DataFrame the analysis function needs from sample data"
    
    #  start by ignoring the class column
    vars_only = sample_data.drop(col_class)
    
    #  identify the mix and max for each column
    sample_data_info = vars_only.describe().collect()
    min_row = sample_data_info[3]
    max_row = sample_data_info[4]
    
    #  create schema for our final DataFrame, one Row at a time
    sample_row = Row("colName", "minValue", "maxValue", "shouldAnalyze")
    
    #  build Python list of Rows of column names and their metadata
    sample_list = []
    cols_analyze_set = set(cols_analyze)
    idx = 1
    for col in vars_only.columns:
        should = (col in cols_analyze_set)
        sample_list.append( sample_row( col, float(min_row[idx]), float(max_row[idx]), should ) )
        idx = idx + 1
        
    #  create the DataFrame and ship it back
    return sql.createDataFrame(sample_list)

    

# 1b) Generate test data (work item #4)
def generate_analysis_data(sc, exp_sensitivity, ctrl_sensitivity, data_info):
    "build the test data from the prepped cols_* DataFrames which should make it easy"
    
    #  gather the cols to analyze first!
    exp_cols = data_info.where(data_info.shouldAnalyze==True).collect()
    all_cols = data_info.collect()
    col_names = []
    for r in all_cols:
        col_names.append(r.colName)
    
    test_list = []
    #  for all values to hold control variables at...
    for c in range(0, ctrl_sensitivity):
    
        #  for each variable we want to analyze...
        for exp_var in exp_cols:  #   THIS IS WHERE I STOPPED PYSPARK CHECKING THIS CODE, START AGAIN HERE
        
            #  for all values to hold focus variable to...
            for e in range(0, exp_sensitivity):
            
                test_row = Row(col_names)
                test_vals = []
                
                #  for each value to be found within a Row of our output DataFrame...
                for col in all_cols:
                
                    #  get min and max values for the variable in question
                    min = data_info.select(colName=col).collect()[0].minValue
                    max = data_info.select(colName=col).collect()[0].maxValue
                    
                    #  set multiplicative variables to exp or ctrl var 
                    factor = c
                    factorMax = ctrl_sensitivity
                    if exp_var == col
                        factor = e
                        factorMax = exp_sensitivity
                        
                    test_vals.append( min + ((max - min) * (factor / factorMax)) )
                    
                test_list.append(test_row(test_vals))
    
    #  bundle all of this into a single DataFrame and ship it back!
    test_data = sc.createDataFrame(test_list, schema=col_names) #  need to set schema to same column headers as data would be
    return test_data


    #  DataFrame.count() gives me number of rows (useful for looping)
    #  DataFrame.collect() gives me a list of Rows 
    #  Row members can be accessed by name, Row.colName, Row.minValue, etc
    #  DataFrame.foreach(f) runs the f function on each Row of the DataFrame
    #  DataFrame.printSchema() gives string of ASCII tree representing DataFrame, may be useful for doing input validation human-legible
    #  DataFrame.schema() gives types within DataFrame, useful for asserting valid DataFrame format    
    #  DataFrame.select(cols) gives a new DataFrame limited to the provided columns
    #  DataFrame.selectExpr()
    #  DataFrame.take(n) return the first n Rows as a list of Rows
    #  DataFrame.where() is an alias for .filter() which takes string conditions to filter Rows

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

    #  test_data = generate_analysis_data(sc, exp_sensitivity, ctrl_sensitivity, data_info)

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
