from AIErrorHandling.training.SitesErrorCodes import *
#from AIErrHand.Tasks import *
#from AIErrHand.DNNTrain import *
import numpy
import random
import tensorflow as tf
dir()
random_seed = random_seed
random.seed(random_seed)
numpy.random.seed(random_seed)
tf.set_random_seed(random_seed)

tasks = Tasks.Tasks( "../data/actionshistory_03042019.json"  , binary=True ) # , TiersOnly=True )
trainer = DNNTrain.DNNTrain( tasks , 0.85)
trainer.MakeModel(layers=[(100,None,'tanh'),(100,None,'relu'),(200,None,'relu')])
fit_res = trainer.Fit(batch_size=1000 , epochs=10 , validation_split=0.25)
trainer.SaveModel( "f1test" , 1 , 1 )

