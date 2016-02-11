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
sc = SparkContext("local", "Simple App")
sqlContext = SQLContext(sc)

# Load the data stored in LIBSVM format as a DataFrame.
# data = sqlContext.read.format("libsvm").load("data/mllib/sample_iris_data.txt")
# data = sqlContext.read.format("csv").load("data/mllib/iris.csv")

# header=false so the columns aren't named after the first row values
# inferSchema=true so that data is read in as correct datatype, not just strings
data = sqlContext.read.load('data/mllib/iris.csv', format='com.databricks.spark.csv', header='false', inferSchema='true')
print("DATA:\n")
print(data)

# now we create a vector of the input columns so they can be one column
ignore = ["C4"]  # ignore the output column
assembler = VectorAssembler(inputCols=[x for x in data.columns if x not in ignore], outputCol='features')
#  data = assembler.transform(data)  # now we've added a 6th column named 'features'

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed")
classIndexer = StringIndexer(inputCol="C4", outputCol="label")

# Read in data for sensitivity analysis
testData = sqlContext.read.load('data/mllib/iris_test_data.csv', format='com.databricks.spark.csv', header='false', inferSchema='true')

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexed", labelCol="label")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, classIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(data)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show()  # opt param: number of records to show

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = model.stages[1]
# summary only
print(treeModel)