from AIErrorHandling.models.SiteErrorCodeModelLoader import *
from AIErrorHandling.models import Predictor

wf = "prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929"
tsk = "/prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929/DataProcessing"

print ( Predictor( **{"wf":wf , "tsk":tsk } ) )
