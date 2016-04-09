# PSAML

PySpark Sensitivity Analysis of ML models

This is our research for ThinkBig Analytics to make a PySpark package that performs sensitivity analysis on spark.ml models for BYU's CS 401R: Data Mining class.

---

The use case is you have a Model already trained against some data you have in a DataFrame (or a sampling thereof) which contains ONLY continuous, numerical input data, and you would like to perofrm sensitivity analysis on some or all of the input variables. For now, only floating-point input columns are supported. Performing an analysis is as easy as making two function calls:

- ```make_data_info()``` build input DataFrame in one call with your training data, which as a parameter to the next function will tell PSAML how to perform your analysis.
- ```do_continuous_input_analysis()``` get final prediction DataFrame by providing the first step's output and your Model. We are looking into supporting categorical input as well in the near future!

---

The test_psaml file is useful for just that: testing. It is suppose to mimic a PSAML caller. To use PSAML in your own environment with your own Models, simply import the psaml.py file and call the two functions listed above. The other functions are private helpers to perform the analysis.

---

To use CSV files as input data, include the following in your spark-submit or pyspark launch:

` --packages com.databricks:spark-csv_2.11:1.3.0 `
