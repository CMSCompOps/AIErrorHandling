# AIErrorHandling
Tools to produce suggestion for the CMS workflow failures recovery panel.
The idea is to train neural networks based on the previous actions and the details of the failed jobs. 

There are two subdirectories :
## training
all the codes that are used for training should go here. For the moment there are two approaches for the training : 
### Based on the error codes and site names and statuses
codes are added under the training/SitesErrorCodes/ codes
### Add the log/error files information for training
a new sub-package under the training can be created under the training directory
## models
API to retrieve the suggestion values.

Testing API
```
curl -d "wf=prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929 \
     &tsk=/prebello_HIRun2018A-v1-HIHardProbesPeripheral-04Apr2019_1033p1_190404_203542_6929/DataProcessing" \
     -X POST http://localhost:8050/predict/
```

# Requirements
* tensorflow
* workflowwebtools
* matplotlib
* django2.2
* tkinter