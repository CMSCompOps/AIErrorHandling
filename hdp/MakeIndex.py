from datetime import datetime
import json
import os

from PrepareSpark import PrepareSpark

class wfmonit_indexing:
    def __init__(self):
        PrepareSpark(self)
        self.LoadData()

    def LoadData(self):
        self.AllData = self.sqlContext.read.option("compression", "gzip").json('hdfs:///project/monitoring/archive/toolsandint/raw/metric/*/*/*/*.gz')
        self.AllData.createOrReplaceTempView("wfdata")
        self.DataFrame = self.spark.sql("SELECT data.data.name, data.metadata.timestamp FROM wfdata")
        self.AllInfo = self.DataFrame.collect()
        #print( len(self.AllInfo) )
        #print( self.AllInfo )

    def WriteToFile(self, out='Index.json'):
        theDict = {}
        print( len(self.AllInfo) )
        for row in self.AllInfo:
            name = str( row.name )
            date = datetime.fromtimestamp(row.timestamp)
            date_str = "{0:%Y/%m/%d}".format( date )
            try:
                theDict[str(name)]['all_dates'].append( date_str )
                if theDict[str(name)]['last'] < row.timestamp :
                    theDict[str(name)]['last'] = row.timestamp 
            except KeyError:
                theDict[str(name)] = { 'all_dates':[date_str ] , 'last':row.timestamp }
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
