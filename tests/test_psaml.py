#
# Iris Analysis
#
import os
import sys

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *

# init
sc = SparkContext('local', 'Test_PSAML')

# Get parent directory of the tests directory
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(parent_dir, 'psaml'))
import psaml

sc.addPyFile(os.path.join(parent_dir, 'psaml/psaml.py'))

sql_context = SQLContext(sc)

# header=false so the columns aren't named after the first row values
# inferSchema=true so that data is read in as correct data type, not just strings
data = sql_context.read.load('tests/resources/iris.csv', format='com.databricks.spark.csv', header='false', inferSchema='true')

# now we create a vector of the input columns so they can be one column
ignore = ['C4']  # ignore the output column
assembler = VectorAssembler(inputCols=[x for x in data.columns if x not in ignore], outputCol='features')

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
# (maxCategories is not set at the moment, however)
feature_indexer = VectorIndexer(inputCol="features", outputCol="indexed")
class_indexer = StringIndexer(inputCol="C4", outputCol="label")

# Read in data for sensitivity analysis
test_data = sql_context.read.load('tests/resources/iris_test_data.csv',
                                  format='com.databricks.spark.csv',
                                  header='false',
                                  inferSchema='true')

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexed", labelCol="label")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[assembler, feature_indexer, class_indexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(data)

# Get our data_info frame, courtesy of PSAML
data_info = psaml.make_data_info(sql_context, test_data, ['C0', 'C1', 'C2', 'C3'], 'C4')

# Make predictions.
predictions = psaml.do_continuous_input_analysis(sc, model, 1, 1, data_info)

# print (predictions)

# Select example rows to display.
predictions.show()  # opt param: number of records to show
