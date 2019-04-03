"""
The main class to load all the predictions with different trained algorithms. One instance of it is automatically created in the initializing of the package and its called "Predictor". 
Error-site codes and log files should be passed to it and a list of Prediction instances will be returned.
:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

from . import SiteErrorCodeModelLoader as SECML
from AIErrorHandling.training.SitesErrorCodes import SitesErrorCodes_path 

class Loader :
    def __init__(self):
        """
        new models should be added manually to the list of AllModels.
        """        
        self.AllModels = []

        from os import listdir
        from os.path import isfile, join, splitext

        sites_errors_models_dir = join(SitesErrorCodes_path,"models")
        files_sites_errors_models_dir = listdir( sites_errors_models_dir )
        models_sites_errors_models_dir = set( [ join(sites_errors_models_dir, splitext(f)[0]) for f in files_sites_errors_models_dir if isfile( join(sites_errors_models_dir, f) ) ] )
        print(models_sites_errors_models_dir)
        for f in models_sites_errors_models_dir:
            print f
            self.AllModels.append( SECML.SiteErrorCodeModelLoader( f ) )
        

    def __call__(self , **inputs):
        """
        gets all the information about the failed job and returns a list of Predictions
        :param kwargs inputs: all the possible inputs from the console
        :param dict good_sites: map of good sites and number of failed jobs in eash site, like what is provided by 'actionhistory' in old-console
        :param dict bad_sites: map of bad sites and number of failed jobs in eash site, like what is provided by 'actionhistory' in old-console
        """
        ret = [model(**inputs) for model in self.AllModels]
        return ret
