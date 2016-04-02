#
# Iris Analysis
#
from pyspark import SparkContext
from pyspark.ml import Model
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import *


# Helper: create data_info DataFrame from sample data (work item #5)
def make_data_info(sql, sample_data, cols_analyze, col_class):
    "create the data_info DataFrame the analysis function needs from sample data"    
    #  start by ignoring the class column
    vars_only = sample_data.drop(col_class)
    #  identify the mix and max for each column
    sample_data_desc = vars_only.describe()
    sample_data_cols = sample_data_desc.columns
    sample_data_cols.remove('summary')
    sample_data_info = sample_data_desc.collect()
    cont_cols = set(sample_data_cols)
    cate_cols = set(vars_only.columns) - cont_cols
    min_row = sample_data_info[3]
    max_row = sample_data_info[4]
    #  create schema for our final DataFrame, one Row at a time
    sample_row = Row("colName", "minValue", "maxValue", "shouldAnalyze", "isCategorical", "categoricalValues")
    #  build Python list of Rows of column names and their metadata
    sample_list = []
    cols_analyze_set = set(cols_analyze)
    idx = 1
    for col in sample_data_cols:
        should = (col in cols_analyze_set)
        sample_list.append( sample_row( col, float(min_row[idx]), float(max_row[idx]), should, false, [] ) )
        idx = idx + 1
    for col in cate_cols:
        should = (col in cols_analyze_set)
        cate_values = []
        thresh = 0.24
        while len(cate_values) < 4 and thresh > 0.009:
            cate_values = sample_data.freqItems([col], thresh).first()[0]
            thresh = thresh - 0.01
        if len(cate_values) > 0:
            sample_list.append( sample_row( col, 0.0, 0.0, should, true, cate_values ) )
    #  create the DataFrame and ship it back
    return sql.createDataFrame(sample_list)
    

# 1b) Generate test data (work item #4)
def _generate_analysis_data(sql, exp_sensitivity, ctrl_sensitivity, data_info):
    "build the test data from the prepped cols_* DataFrames which should make it easy"
    #  gather the cols to analyze first!
    exp_cols = data_info.where(data_info.shouldAnalyze==True && data_info.isCategorical==False).collect()
    cont_cols = data_info.where(data_info.isCategorical==False).collect()
    cate_cols = data_info.where(data_info.isCategorical==True).collect()
    all_cols = data_info.collect()
    col_names = []
    for r in all_cols:
        col_names.append(r.colName)
    cont_names = []
    for r in cont_cols:
        cont_names.append(r.colName)
    cate_names = []
    for r in cate_cols:
        cate_names.append(r.colName)
    col_names_set = set(col_names)
    #  gather the min/max values once for efficiency
    mins = {}
    maxs = {}
    for col in cont_names:
        colrow = data_info.where(data_info.colName==col).first()
        mins[col] = colrow.minValue
        maxs[col] = colrow.maxValue
    values = {}
    currValues = {}
    cntValues = {}
    valueCombos = 1
    nextValue = 0
    for col in cate_names:
        colrow = data_info.where(data_info.colName==col).first()
        values[col] = colrow.categoricalValues
        currValues[col] = 0
        cntValues[col] = len(colrow.categoricalValues)
        valueCombos = valuesCombos * cntValues[col]
    test_list = []
    #  for all combinations of categorical variables...    
    for i in range(valueCombos):
        #  for all values to hold control variables at...
        for c in range(0, ctrl_sensitivity+1):
            #  for each variable we want to analyze...
            for exp_var in exp_cols: 
                #  for all values to hold focus variable to...
                for e in range(0, exp_sensitivity+1):
                    #  test_row = Row(col_names)
                    test_vals = []
                    #  for each value to be found within a Row of our output DataFrame...
                    for col in col_names:
                        if col.isCategorical:
                            #  fetch current control value for categorical variable
                            test_vals.append( values[col.colName][currValues[col.colName]] )
                        else:
                            #  get min and max values for the variable in question
                            min = mins[col]
                            max = maxs[col]
                            #  set multiplicative variables to exp or ctrl var 
                            factor = float(c)
                            factorMax = float(ctrl_sensitivity)
                            if exp_var.colName == col:
                                factor = float(e)
                                factorMax = float(exp_sensitivity)
                            #  hard-wiring factorMax to 1 so that a 0 results in values only analyzed at 0% of possible value range
                            if factorMax == 0:
                                factorMax = float(1)
                            test_vals.append( min + ((max - min) * (factor / factorMax)) )
                    test_list.append(test_vals)
        #  advance categorical values
        multFactor = 1
        for v in range(len(cate_names)):
            name = cate_names[v]
            max = cntValues[name]
            currValues[name] = (i / multFactor) % max
    #  bundle all of this into a single DataFrame and ship it back!
    test_data = sql.createDataFrame(test_list, schema=col_names)
    return test_data


def _format_output(sql, exp, ctrl, data_info, predictions):
    """Used to take raw predictions and label with the variance levels that led to the prediction"""
    #  NOTE: in some cases, a mapping from prediction to real-data label would be nice, such as iris, but seems to be awkward for numeric labels... use raw prediction for now
    #  1) Package prediction with exp% and ctrl% used to acquire it
    #       Indexes appear to be order-preserved... this will need tested on supercomputer and results diff'd
    infos = data_info.collect()
    preds = predictions.collect()
    exp_idx = -1
    ll = []
    for i in range(0, len(preds)):
        if (i % ((exp+1)*(ctrl+1))) == 0:
            exp_idx = exp_idx + 1
            while not infos[exp_idx].shouldAnalyze:
                exp_idx = exp_idx + 1
        e = 0
        if exp <> 0:
            e = float(i % (exp+1)) / float(exp)
        c = 0
        if ctrl <> 0:
            c = float((i/(exp+1)) % (ctrl+1)) / float(ctrl)
        l = []
        l.append(preds[i].prediction)
        l.append(infos[exp_idx].colName)
        l.append(e)
        l.append(c)
        ll.append(l)
    #  2) Take list of 1) results and make and return DataFrame
    results = sql.createDataFrame(ll, schema=['Prediction', 'AnalyzedVariable', 'ExpSensitivity', 'CtrlSensitivity'])
    return results

def do_continuous_input_analysis(sc, model, exp_sensitivity, ctrl_sensitivity, data_info):
    # ##########################################################################################################
    #
    # 0) Verify input
    #
    try:
        assert (exp_sensitivity >= 0), "Experiment Sensitivity must be a non-negative integer"
        assert (ctrl_sensitivity >= 0), "Control Variable Sensitivity must be a non-negative integer"
    except AssertionError as e:
        raise ValueError(e.args)
    try:
        assert (type(sc) is SparkContext), "Invalid SparkContext"
        assert (isinstance(model, Model)), "Invalid ML Model; Model given: {0}".format(str(type(model)))
        assert (type(data_info) is DataFrame), "data_info is not a valid DataFrame"
    except AssertionError as e:
        raise TypeError(e.args)
    try:
        assert (len(data_info.columns) == 4), \
            "data_info is invalid; Len should be 4 instead of {0}".format(len(data_info.columns))
        assert (set(data_info.columns) == {'colName', 'minValue', 'maxValue', 'shouldAnalyze'}), \
            "data_info is invalid; Contains incorrect columns"
    except AssertionError as e:
        raise RuntimeError(e.args)
    # 0.5) create SQLContext
    sql = SQLContext(sc)
    # ##########################################################################################################
    #
    # 1) Generate test data
    #
    test_data = _generate_analysis_data(sql, exp_sensitivity, ctrl_sensitivity, data_info)
    # ##########################################################################################################
    #
    # 2) Make predictions.
    #
    predictions = model.transform(test_data)
    # ##########################################################################################################
    #
    # 3) Transform predictions into output DataFrame
    #
    packaged_predictions = _format_output(sql, exp_sensitivity, ctrl_sensitivity, data_info, predictions)
    return packaged_predictions
