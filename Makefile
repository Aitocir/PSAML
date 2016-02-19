test: ; cd tests; spark-submit --master local[2] --packages com.databricks:spark-csv_2.11:1.3.0 test_psaml.py; cd ..;
