from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf
from pyspark.context import SparkContext
import py4j

import atexit
import os
import platform
import warnings

def PrepareSpark(obj):
    SparkContext._ensure_initialized()
    try:
        # Try to access HiveConf, it will raise exception if Hive is not added
        conf = SparkConf()
        if conf.get('spark.sql.catalogImplementation', 'hive').lower() == 'hive':
            SparkContext._jvm.org.apache.hadoop.hive.conf.HiveConf()
            obj.spark = SparkSession.builder\
                                .enableHiveSupport()\
                                .getOrCreate()
        else:
            obj.spark = SparkSession.builder.getOrCreate()
    except py4j.protocol.Py4JError:
        if conf.get('spark.sql.catalogImplementation', '').lower() == 'hive':
            warnings.warn("Fall back to non-hive support because failing to access HiveConf, "
                          "please make sure you build spark with hive")
        obj.spark = SparkSession.builder.getOrCreate()
    except TypeError:
        if conf.get('spark.sql.catalogImplementation', '').lower() == 'hive':
            warnings.warn("Fall back to non-hive support because failing to access HiveConf, "
                          "please make sure you build spark with hive")
        obj.spark = SparkSession.builder.getOrCreate()


    #print(1)
    obj.sc = obj.spark.sparkContext
    sqlContext = obj.spark._wrapped
    sqlCtx = sqlContext
    #print(2)
    obj.sqlContext = SQLContext(obj.sc)
    #print(3)

