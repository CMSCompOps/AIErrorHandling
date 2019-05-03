"""
a class to load a model that has been trained using the table of sites and errors
:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

import numpy as np
import json

from tensorflow import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.keras as keras

from AIErrorHandling.training.SitesErrorCodes.Tasks import Task
from . import Prediction, TheTFGraph, TheTFSession

from workflowwebtools import workflowinfo
from cmstoolbox import sitereadiness


class SiteErrorCodeModelLoader :
    def __init__(self , model_file_name ):
        """
        loads the trained model
        :param str model_file_name: two files are expected to exist in this address : one json and one hdf5 file. They should be stored using the SaveModel method of the DNNTrain class
        """
        logging.set_verbosity(logging.ERROR)
        with TheTFGraph.as_default():
            with TheTFSession.as_default():
                self.model = keras.models.load_model( model_file_name + ".hdf5" , compile=False )
        JSON = json.load( open(model_file_name + ".json" ) )
        self.all_sites = list(JSON['all_sites'])
        self.all_errors = list(JSON['all_errors'])
        self.all_actions = list(JSON['all_actions'])
        self.IsBinary = bool(JSON['IsBinary'])
        self.TiersOnly = bool(JSON['TiersOnly'])
        self.Task = Task({} , "TaskLoader" , self)
        self.Name = model_file_name.split('/')[-1]
        self.ModelID = int( JSON['model'] )
        self.InputTrainingDataID = int( JSON['trainingdata'])

        self.Prediction = Prediction.Prediction( self.ModelID , self.InputTrainingDataID ) 
        
    def __call__(self , good_sites={} , bad_sites={} , wf=None , tsk=None , sourcejson=None):
        """
        returns the prediction of the model for the input good/bad site errors
        :param dict good_sites: map of good sites and number of failed jobs in eash site, like what is provided by 'actionhistory' in old-console
        :param dict bad_sites: map of bad sites and number of failed jobs in eash site, like what is provided by 'actionhistory' in old-console
        """
        if wf :
            if sourcejson :
                jj = json.load( open(sourcejson) )
                jjb = jj[wf]['errors']
                good_sites = jjb['good_sites']
                bad_sites = jjb['bad_sites']

            else:
                wfinfo = workflowinfo.WorkflowInfo( wf )
                errors = wfinfo.get_errors()[ tsk ] 
                for err in errors :
                    try:
                        a = int(err)
                    except :
                        print( "error %s skipped" % err )
                        continue
                    for site in errors[err] :
                        stat = sitereadiness.site_readiness( site )
                        if stat == 'green' :
                            good_sites.setdefault( err , {} )[ site ] = errors[err][site]
                        else:
                            bad_sites.setdefault( err , {} )[ site ] = errors[err][site]
                #print good_sites
                #print bad_sites

        self.Task.normalize_errors( good_sites , bad_sites , TiersOnly=self.TiersOnly )
        with TheTFGraph.as_default():
            with TheTFSession.as_default():
                prediction = self.model.predict( np.array( [ self.Task.Get2DArrayOfErrors() ] ) )
        PRED = str(prediction)
        if self.IsBinary:
            PRED = self.all_actions[ prediction[0][0] > 0.5 ]
        self.Prediction.SetValues( PRED , "" , "" )
        return str(self.Prediction)

