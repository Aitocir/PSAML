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
data = sql_context.read.load('resources/iris.csv', format='com.databricks.spark.csv', header='false', inferSchema='true')

# now we create a vector of the input columns so they can be one column
ignore = ['C4']  # ignore the output column
assembler = VectorAssembler(inputCols=[x for x in data.columns if x not in ignore], outputCol='features')

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
feature_indexer = VectorIndexer(inputCol="features", outputCol="indexed")
class_indexer = StringIndexer(inputCol="C4", outputCol="label")

# Read in data for sensitivity analysis
test_data = sql_context.read.load('resources/iris_test_data.csv',
                                  format='com.databricks.spark.csv',
                                  header='false',
                                  inferSchema='true')

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexed", labelCol="label")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[assembler, feature_indexer, class_indexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(data)

# Make predictions.
predictions = psaml.do_continuous_input_analysis(sc, model, 1, 1, test_data)

# Select example rows to display.
predictions.select("prediction", "label", "features").show()  # opt param: number of records to show

