from AIErrorHandling.models.SiteErrorCodeModelLoader import *
from AIErrorHandling.models import Predictor


import json
#jj = json.load( open("../data/actionshistory_03042019.json") )
#for a in jj :
#    jjb = jj[a]['errors']
    #print(jjb)
#    print( Predictor( **jjb  ) )

wf = "prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929"
tsk = "/prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929/DataProcessing"

print ( Predictor( **{"wf":wf , "tsk":tsk } ) )
