# ==========================
# Cloud-Based Data Processing App
# SICT 4313 - Cloud & Distributed Systems
# Using Python, PySpark, and Streamlit
# Author:
# 1-khaled zain al din 120212536
# 

# ==========================

import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import col


spark = SparkSession.builder \
    .appName("Cloud Data Processing") \
    .getOrCreate()

st.title("Cloud-Based Distributed Data Processing App")
st.write("""
Upload your dataset (CSV/JSON/TXT) and select processing options.
""")


uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "json", "txt"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    
    
    if file_type == 'csv':
        df_pd = pd.read_csv(uploaded_file)
    elif file_type == 'json':
        df_pd = pd.read_json(uploaded_file)
    else:
        df_pd = pd.read_csv(uploaded_file, sep="\t")  
    
    st.write("### Preview of your dataset")
    st.dataframe(df_pd.head(10))
    
    
    df = spark.createDataFrame(df_pd)
    
    st.write("### Descriptive Statistics")
    
    
    num_rows = df.count()
    num_cols = len(df.columns)
    null_counts = {col_name: df.filter(col(col_name).isNull()).count() for col_name in df.columns}
    unique_counts = {col_name: df.select(col_name).distinct().count() for col_name in df.columns}
    
    st.write(f"- Number of rows: {num_rows}")
    st.write(f"- Number of columns: {num_cols}")
    st.write(f"- Null/missing values per column: {null_counts}")
    st.write(f"- Unique values per column: {unique_counts}")
    
    
    st.write("### Machine Learning Jobs")
    
    ml_option = st.selectbox("Select ML job", ["KMeans Clustering", "Linear Regression", "FPGrowth"])
    
    if st.button("Run ML Job"):
        if ml_option == "KMeans Clustering":
          
            numeric_cols = [c for c, t in df.dtypes if t in ('int', 'double')]
            if len(numeric_cols) < 2:
                st.warning("Not enough numeric columns for clustering.")
            else:
                
                from pyspark.ml.feature import VectorAssembler
                assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
                df_features = assembler.transform(df)
                
                kmeans = KMeans(k=3, seed=1, featuresCol="features")
                model = kmeans.fit(df_features)
                predictions = model.transform(df_features)
                
                st.write("### Cluster Centers")
                st.write(model.clusterCenters())
                st.write("### Predictions")
                st.dataframe(predictions.select(numeric_cols + ["prediction"]).toPandas().head(10))
        
        elif ml_option == "Linear Regression":
         
            numeric_cols = [c for c, t in df.dtypes if t in ('int', 'double')]
            if len(numeric_cols) < 2:
                st.warning("Not enough numeric columns for regression.")
            else:
                target = numeric_cols[0]
                features = numeric_cols[1:]
                
                from pyspark.ml.feature import VectorAssembler
                assembler = VectorAssembler(inputCols=features, outputCol="features")
                df_features = assembler.transform(df).select("features", target)
                
                lr = LinearRegression(featuresCol="features", labelCol=target)
                model = lr.fit(df_features)
                
                st.write("### Regression Coefficients")
                st.write(model.coefficients)
                st.write(f"### Intercept: {model.intercept}")
                st.write(f"### R2: {model.summary.r2}")
        
        elif ml_option == "FPGrowth":
           
            string_cols = [c for c, t in df.dtypes if t == 'string']
            if not string_cols:
                st.warning("No categorical columns found for FPGrowth.")
            else:
              
                df_fp = df.select(string_cols[0].alias("items"))
                df_fp = df_fp.na.drop()
                
                fpGrowth = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.5)
                model = fpGrowth.fit(df_fp)
                
                st.write("### Frequent Itemsets")
                st.dataframe(model.freqItemsets.toPandas().head(10))
                st.write("### Association Rules")
                st.dataframe(model.associationRules.toPandas().head(10))

st.write("### End of App")
