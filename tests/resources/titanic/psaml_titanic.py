#
# Test PSAML using the Kaggle Titanic dataset
# https://www.kaggle.com/c/titanic 
#

import os
import sys

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
import plotly
import plotly.graph_objs as go
import pandas as pd

# init
sc = SparkContext('local', 'PSAML_Titanic')

# Get parent directory of the tests directory
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(parent_dir, 'psaml'))
import psaml

sc.addPyFile(os.path.join(parent_dir, 'psaml/psaml.py'))

sql_context = SQLContext(sc)

# header=false so the columns aren't named after the first row values
# inferSchema=true so that data is read in as correct data type, not just strings
data = sql_context.read.load('tests/resources/titanic/train.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')

# now we create a vector of the input columns so they can be one column
ignore = ['Survived', 'Name', 'Ticket', 'Cabin']  # ignore the output column and nonquantifiable data
assembler = VectorAssembler(inputCols=[x for x in data.columns if x not in ignore], outputCol='features')

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
# (maxCategories is not set at the moment, however)
#  feature_indexer = VectorIndexer(inputCol="features", outputCol="indexed")
class_indexer = StringIndexer(inputCol="C4", outputCol="label")

# Read in data for sensitivity analysis
test_data = sql_context.read.load('tests/resources/iris_test_data.csv',
                                  format='com.databricks.spark.csv',
                                  header='false',
                                  inferSchema='true')

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="features", labelCol="label")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[assembler, class_indexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(data)

# Get our data_info frame, courtesy of PSAML
cols_to_analyze = ['C0', 'C1', 'C2', 'C3']
data_info = psaml.make_data_info(sql_context, test_data, cols_to_analyze, 'C4')

# Make predictions.
predictions = psaml.do_continuous_input_analysis(sc, model, 5, 5, data_info)


# Select example rows to display.
# predictions.show()  # opt param: number of records to show

fig = plotly.tools.make_subplots(rows=len(cols_to_analyze), cols=1)
sql_context.registerDataFrameAsTable(predictions, "predictions")

for i in range(len(cols_to_analyze)):
    ctrl_sensitivity_values = sql_context.sql("SELECT DISTINCT CtrlSensitivity FROM predictions WHERE AnalyzedVariable='{0}'".format(cols_to_analyze[i])).toPandas().sort('CtrlSensitivity').CtrlSensitivity
    for ctrl_val in ctrl_sensitivity_values:
        temp = sql_context.sql("SELECT Prediction, ExpSensitivity FROM predictions WHERE AnalyzedVariable='{0}' AND CtrlSensitivity={1}".format(cols_to_analyze[i], ctrl_val)).toPandas()

        trace = go.Scatter (
            x = temp.ExpSensitivity,
            y = temp.Prediction,
            name = '{0} - Ctrl: {1}'.format(cols_to_analyze[i], ctrl_val),
            line=dict(
                shape='spline'
            )
        )

        fig.append_trace(trace, i+1, 1)


plot(fig, filename='tests/basic-line.html')