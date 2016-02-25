#
# Iris Analysis
#
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer

# init
sc = SparkContext('local', 'Test_PSAML')
sc.addPyFile("/Users/aitocir/Documents/cs401r/ThinkBig/PSAML/psaml/psaml.py")
from psaml import *
sqlContext = SQLContext(sc)

# header=false so the columns aren't named after the first row values
# inferSchema=true so that data is read in as correct datatype, not just strings
data = sqlContext.read.load('data/iris.csv', format='com.databricks.spark.csv', header='false', inferSchema='true')

# now we create a vector of the input columns so they can be one column
ignore = ['C4']  # ignore the output column
assembler = VectorAssembler(inputCols=[x for x in data.columns if x not in ignore], outputCol='features')

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed")
classIndexer = StringIndexer(inputCol="C4", outputCol="label")

# Read in data for sensitivity analysis
testData = sqlContext.read.load('data/iris_test_data.csv', format='com.databricks.spark.csv', header='false', inferSchema='true')

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexed", labelCol="label")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[assembler, featureIndexer, classIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(data)

# Make predictions.
predictions = doContinuousInputAnalysis(sc, model, 1, 1, testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show()  # opt param: number of records to show

