# PSAML

PySpark Sensitivity Analysis of ML models

THIS IS A WORK IN PROGRESS; TOTALLY ALPHA RIGHT NOW

This is our research for ThinkBig to make a PySpark helper that performs sensitivity analysis on spark.ml models for BYU's CS 428: Software Engineering class.

---

The use case is you have a Model already trained against some data you have in a DataFrame (or a sampling thereof) which contains ONLY continuous, numerical input data, and you would like to perofrm sensitivity analysis on some or all of the input variables. From a 10,000 foot view, this is the workflow and what we've got working so far:

- [WORKING] Prepare a DataFrame that describes the statistics of your data and what variables you want to have analyzed.
- [WORKING] Use the above DataFrame to create a DataFrame of generated data. The Model's predictions on this data will give you the final sensitivity analysis.
- [WORKING] Provide your Model and this DataFrame of analysis data, and get back a prediction DataFrame from the Model's .transform() function.
- [-------] Mutate the prediction DataFrame into a descriptive DataFrame that will clearly show you what levels on each input variable led to the prediction gained (variable X was varied, X was held at 30%, all other variables were held at 50%).

The above workflow will be hidden behind a function API that will boil down to two steps for the caller: build input DataFrame in one call with your data, and get final prediction DataFrame by providing the first step's output and your Model. We are looking into supporting categorical input as well if we have time!

---

Running the script:

`$ {SPARK_DIR}/bin/spark-submit --master local[2] --packages com.databricks:spark-csv_2.11:1.3.0 iris_analysis.py`
