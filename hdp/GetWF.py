from datetime import datetime
import json
from PrepareSpark import PrepareSpark
import os

class get_wf:
    def __init__(self , inputjson="Index.json"):
        with open( inputjson ) as infile :
            self.AllWFs = json.load( infile )

        self.AllLoadedWFs = {}
        PrepareSpark(self)

    def LoadData(self , wf=None):
        self.WF = wf
        if wf :
            self.AllData = self.sqlContext.read.option("compression",
                                                       "gzip").json( [ 'hdfs:///project/monitoring/archive/toolsandint/raw/metric/{}/*'
                                                                       '.gz'.format(date) for date in set( self.AllWFs[ wf ]['all_dates'] ) ] )
        else:
            self.AllData = self.sqlContext.read.option("compression",
                                                       "gzip").json( 'hdfs:///project/monitoring/archive/toolsandint/raw/metric/*/*/*/*.gz' )

        self.AllData.createOrReplaceTempView("wfdata")
        if wf :
            self.DataFrame = self.spark.sql("SELECT data.data.name, data.data.tasks, data.metadata.timestamp FROM wfdata WHERE data.data.name='{}' SORT BY data.metadata.timestamp".format(wf) )
        else:
            self.DataFrame = self.spark.sql("SELECT data.data.name, data.data.tasks, data.metadata.timestamp FROM wfdata SORT BY data.metadata.timestamp")
            #self.AllInfo = self.DataFrame.collect()

    def GetLast(self , wf = None):
        if not wf:
            wf = self.WF
        if not wf :
            raise ValueError("workflow name should be given, either while loading data or while getting information")

        #return self.DataFrame.filter( "name=='{}' and timestamp={}".format( wf , self.AllWFs[wf]['last'] ) ).collect()
        self.LastData = self.DataFrame.filter( "name=='{}'".format( wf ) ).collect()
        return self.LastData
        

    def GetWFTask(self, wf , tsk_name):
        if wf in self.AllLoadedWFs:
            if tsk_name in self.AllLoadedWFs[wf] :
                return self.AllLoadedWFs[wf][tsk_name]
            else:
                print("tsk {} is not available in wf {}".format( tsk_name , wf ) )
                return None

        allinfo = self.GetLast( wf )
        lastrow = None
        parsed_tasks = {}
        for row in allinfo:
            if row['name'] == wf :
                for tsk in row['tasks']:
                    tsk_name = tsk['name']
                    errors = {-10:1}
                    if tsk_name in parsed_tasks:
                        errors = parsed_tasks[tsk_name]
                        errors[-10] += 1
                    else:
                        parsed_tasks[tsk_name] = errors
                    for err in tsk['errors'] :
                        errcode = str( err['errorCode'] )
                        site = err['siteName']
                        count = err['counts']

                        if errcode in errors:
                            if site in errors[errcode] :
                                errors[errcode][site] += count
                            else:
                                errors[errcode][site] = count
                        else:
                            errors[errcode] = {site:count}

        for tsk in parsed_tasks:
            count = parsed_tasks[tsk][-10]
            del parsed_tasks[tsk][-10]
            for err in parsed_tasks[tsk] :
                for site in parsed_tasks[tsk][err]:
                    parsed_tasks[tsk][err][site] /= count

        self.AllLoadedWFs[wf] = parsed_tasks
        if tsk_name in self.AllLoadedWFs[wf] :
            return self.AllLoadedWFs[wf][tsk_name]
        else:
            print("tsk {} is not available in wf {}".format( tsk_name , wf ) )
            return None

    def GetErroCodeSiteMatrix(self , wf = None , tsk_name = ''):
        if not wf:
            wf = self.WF
        #row = self.GetLast( wf )
        row = self.GetWFTask( wf , tsk_name )
        if not row:
            #raise ValueError("only one row is expected but {} rows are returned".format( len(row) ) )
            #print( "no row is found for {}".format( wf + "/" + tsk_name ) )
            return None
        #print(row)
        # they should be converted to a format similar to output json file of the actionhistory json file

        return row

wf = get_wf()
wf.LoadData() 
#a = wf.GetErroCodeSiteMatrix( "prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929" )
#print(a)

actions = None
with open('/home/webservice/AIEH/AIErrorHandling/training/SitesErrorCodes/data/actionshistory_06062019.json') as actionhistory :
    actions = json.load( actionhistory )

for wf_tsk in actions:
    wf_ = wf_tsk.split("/")[1]
    if not '1905' in wf_:
        continue
    tsk_ = wf_tsk.split("/")[-1] #wf_tsk[ len(wf_)+2 : ]
    if wf_ in list( wf.AllWFs.keys() ) :
        errors_console_a = actions[ wf_tsk ]['errors']
        errors_console = {** errors_console_a['good_sites'] , ** errors_console_a['bad_sites']}
        if '-1' in errors_console:
            del errors_console['-1']
        errors_hdp = wf.GetErroCodeSiteMatrix( wf_  , tsk_)
        if errors_hdp == -1 :
            #print( 'tsk' , tsk_ , 'for wf', wf_ , 'is not available in hdp')
            continue
        if not errors_hdp :
            #print( wf_tsk , 'is not found in hdp' )
            continue
        shared_items = { err: errors_hdp[err] for err in errors_console if err in errors_hdp and errors_hdp[err] == errors_console[err] }
        if len(shared_items) == len( errors_hdp ) == len(errors_console):
            print(wf_tsk, 'passed the test')
        else:
            print('-' , wf_tsk)
            print('\t- workflowmonit:' +str( errors_hdp) )
            print('\t- console:'+ str(errors_console) )
    #else :
    #    print(wf_tsk, "is not in the hdp")
#print(wf.AllInfo)
#wf.AllInfo[0]['tasks'][0]['errors'][0]['errorCode']
#wf.AllInfo[0]['tasks'][0]['errors'][0]['siteName']
#wf.AllInfo[0]['tasks'][0]['errors'][0]['counts']
