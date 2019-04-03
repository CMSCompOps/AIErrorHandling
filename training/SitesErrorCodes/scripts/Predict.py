from AIErrorHandling.models.SiteErrorCodeModelLoader import *
from AIErrorHandling.models import Predictor


import json
#jj = json.load( open("../data/actionshistory_03042019.json") )
#for a in jj :
#    jjb = jj[a]['errors']
    #print(jjb)
#    print( Predictor( **jjb  ) )

print ( Predictor( **{"wf":"pdmvserv_task_HIG-RunIIFall18wmLHEGS-01108__v1_T_190218_104151_7070" , "tsk":'/pdmvserv_task_HIG-RunIIFall18wmLHEGS-01108__v1_T_190218_104151_7070/HIG-RunIIFall18wmLHEGS-01108_0/HIG-RunIIAutumn18DRPremix-01079_0/HIG-RunIIAutumn18DRPremix-01079_1/HIG-RunIIAutumn18DRPremix-01079_1MergeAODSIMoutput/HIG-RunIIAutumn18MiniAOD-01079_0/HIG-RunIIAutumn18MiniAOD-01079_0MergeMINIAODSIMoutput' } ) )
