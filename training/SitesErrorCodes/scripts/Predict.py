from AIErrorHandling.models.SiteErrorCodeModelLoader import *
from AIErrorHandling.models import Predictor


import json
#jj = json.load( open("../data/actionshistory_03042019.json") )
#for a in jj :
#    jjb = jj[a]['errors']
    #print(jjb)
#    print( Predictor( **jjb  ) )

wf = "/vlimant_ACDC0_task_HIG-RunIIFall17wmLHEGS-01415__v1_T_180706_002124_986/HIG-RunIIFall17DRPremix-02001_1/HIG-RunIIFall17DRPremix-02001_1MergeAODSIMoutput/HIG-RunIIFall17MiniAODv2-01299_0"
print( Predictor( **{"wf":wf , "sourcejson":"../data/actionshistory_03042019.json"} ) )

wf = "prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929"
tsk = "/prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929/DataProcessing"

#print ( Predictor( **{"wf":wf , "tsk":tsk } ) )
