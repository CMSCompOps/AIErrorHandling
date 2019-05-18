from datetime import datetime
import json
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf
from pyspark.context import SparkContext
import py4j

import atexit
import os
import platform
import warnings


class wfmonit_indexing:
    def __init__(self):
        SparkContext._ensure_initialized()
        try:
            # Try to access HiveConf, it will raise exception if Hive is not added
            conf = SparkConf()
            if conf.get('spark.sql.catalogImplementation', 'hive').lower() == 'hive':
                SparkContext._jvm.org.apache.hadoop.hive.conf.HiveConf()
                self.spark = SparkSession.builder\
                                    .enableHiveSupport()\
                                    .getOrCreate()
            else:
                self.spark = SparkSession.builder.getOrCreate()
        except py4j.protocol.Py4JError:
            if conf.get('spark.sql.catalogImplementation', '').lower() == 'hive':
                warnings.warn("Fall back to non-hive support because failing to access HiveConf, "
                              "please make sure you build spark with hive")
            self.spark = SparkSession.builder.getOrCreate()
        except TypeError:
            if conf.get('spark.sql.catalogImplementation', '').lower() == 'hive':
                warnings.warn("Fall back to non-hive support because failing to access HiveConf, "
                              "please make sure you build spark with hive")
            self.spark = SparkSession.builder.getOrCreate()


        print(1)
        self.sc = self.spark.sparkContext
        sqlContext = self.spark._wrapped
        sqlCtx = sqlContext
        print(2)
        self.sqlContext = SQLContext(self.sc)
        print(3)
        self.LoadData()
        print(9)

    def LoadData(self):
        print(4)
        self.AllData = self.sqlContext.read.option("compression", "gzip").json('hdfs:///project/monitoring/archive/toolsandint/raw/metric/2019/*/*/*.gz')
        print(5)
        self.AllData.createOrReplaceTempView("wfdata")
        print(6)
        self.DataFrame = self.spark.sql("SELECT data.data.name, data.metadata.timestamp FROM wfdata")
        print(7)
        self.AllInfo = self.DataFrame.collect()
        print( len(self.AllInfo) )
        #print( self.AllInfo )
        print(8)

    def WriteToFile(self, out='Index.json'):
        theDict = {}
        print( len(self.AllInfo) )
        for row in self.AllInfo:
            name = str( row.name )
            date = datetime.fromtimestamp(row.timestamp)
            date_str = "{0:%Y/%m/%d}".format( date )
            try:
                theDict[str(name)].append( date_str )
            except KeyError:
                theDict[str(name)] = [ date_str ]
        with open(out, 'w') as outfile:
            json.dump( theDict , outfile )

if __name__ == "__main__":
    if os.environ.get("SPARK_EXECUTOR_URI"):
        SparkContext.setSystemProperty("spark.executor.uri", os.environ["SPARK_EXECUTOR_URI"])

    print("starting 1")
    monit_index = wfmonit_indexing()
    print("created")
    monit_index.WriteToFile()
    print("file written")
