"""
The `Prediction` class provides the data structure of the information that should be stored in the database.
:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

class Prediction:    
    def __init__(self , ModelID , InputTrainingDatasetID , model_file_name):
        """
        creates and instance of the prediction class
        :param int ModelID: the integer id of the model. this value will be stored in database for future references. authors and developers should keep track of its values to make sense.
        :param int InputTrainingDatasetID: the integer id of the dataset that was used for training the model. this value will be stored in database for future references. authors and developers should keep track of its values to make sense.
        """
        self.ModelFileName = model_file_name
        self.ModelID = ModelID
        self.InputTrainingDatasetID = InputTrainingDatasetID
        self.PredictedActionName = ""
        self.Significance = 0.0
        self.Details = ""
        
    def SetValues(self, PredictedAction , Significance , Details ):
        """
        set the prediction values
        :param str PredictedAction: the value of the predicted action
        :param float Significance: the significance of the prediction
        :param str Details: a place holder to store more details about this prediction
        """
        self.PredictedActionName = PredictedAction
        self.Significance = Significance
        self.Details = Details

    def GetDictionary(self):
        ret = { 'value':self.PredictedActionName ,
                'model_description_link': 'https://github.com/CMSCompOps/AIErrorHandling/tree/aieh_machine/training/SitesErrorCodes/models/' + self.ModelFileName + '.json' ,
                'performance_outputs': 'https://github.com/CMSCompOps/AIErrorHandling/tree/aieh_machine/training/SitesErrorCodes/models/' + self.ModelFileName ,
                }
        return ret

    def __str__(self):
        return str(self.GetDictionary())


    def __repr__(self):
        return str(self.GetDictionary())
