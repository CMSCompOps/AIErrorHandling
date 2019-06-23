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

#tasks_allvsmem = Tasks.Tasks( "../data/actionshistory.json"  , classes={'acdc':[0,1,-1] , 'mem':[2,3,4]} , countclasses = True )
tasks_binary = Tasks.Tasks( "../data/actionshistory.json"  , classes={'acdc':[0,-1] , 'others':[1,2,3,4]} , countclasses = True )
#tasks_mclass = Tasks.Tasks( "../data/actionshistory.json"  , classes={'acdc':[0,-1] , 'cloned':[1] , 'acdcmemory':[2,3,4]} , countclasses = True )
#tasks_tiers = Tasks.Tasks( "../data/actionshistory.json"  , classes={'acdc':[0,-1] , 'others':[1,2,3,4]} , countclasses = True , TiersOnly=True )
tasks = tasks_binary
#tasks_good_sites = Tasks.Tasks("../data/actionshistory.json"  , site_statuses = ['good_sites'] )
#tasks_bad_sites = Tasks.Tasks("../data/actionshistory.json"  , site_statuses = ['bad_sites'] )

def Plot2DErrors():
    return tasks.PlotDominantErrors()

def PlotCorrelations():
    h0 = tasks.PlotErrorInfoForAction( 0 , 4 , 20 , 'acdc with site list modification' )
    h1 = tasks.PlotErrorInfoForAction( 1 , 1 , 34 , 'clone with site list modification' )
    h2 = tasks.PlotErrorInfoForAction( [2,3,4] , 2 , 21 , 'acdc with memory modification2')
    h0.DrawNormalized("P0")
    h1.DrawNormalized("P0 SAMES")
    h2.DrawNormalized("P0 SAMES")

def PlotErrorSummary():
    hGood = tasks.PlotNTotalErrs('Good Sites' , 'good' )
    hGood.SetLineColor( 2 )
    hBad = tasks.PlotNTotalErrs( 'Bad Sites' , 'bad' )
    hBad.SetLineColor( 4 )
    hGood.DrawNormalized()
    hBad.DrawNormalized("sames")

def Optimize():
    X_val_ , Y_val_, X_train_, y_train_, X_test_, y_test_ = tasks.GetTrainTestDS( 0.85 , True , 0.3)

    from tensorflow import keras
    ### hack tf-keras to appear as top level keras
    import sys
    sys.modules['keras'] = keras

    import talos as ta
    import tensorflow as tf

    #TheTFGraph = tf.get_default_graph()
    #TheTFGraph.as_default()
    #TheTFSession = tf.Session()

    def MakeModelOptimizer(xtrain, ytrain, xval, yval, p):
        trainer = DNNTrain.DNNTrain( x_train = xtrain, y_train = ytrain, x_val = xval, y_val = yval , x_test = X_test_ , y_test = y_test_ , tasks = None , train_ratio = None )
        trainer.Tasks = tasks

        Layers = []
        for nl in range(0 , p['nl'] ):
            Layers.append( (p["l{}nn".format(nl)],p['l{}reg'.format(nl)],p['l{}act'.format(nl)]) )
        trainer.MakeModel( flatten=True ,
                           layers = Layers ,
                           optimizer='adam' , loss=p['loss'] , LR=p['lr'] ) #, moremetrics=[ta.live()] )
        trainer.Fit(batch_size=p['batch_size'], epochs=p['epochs'] , verbose=0 , weight=p['w'])
        return trainer.FitHistory , trainer.model


    '''Initial optimization
    parameters = {
        'nl':[6],
        'lr':[0.02] , #(0.005, 0.030 , 5 ),
        'batch_size':[5000], # , 5000 , 10000] ,
        'epochs':[25],
        'w':[True],
        'loss':[None , 'mean_squared_error' ]
    }

    for nl in range(0 , max(parameters['nl']) ):
        parameters['l{}nn'.format(nl)] = [100]
        parameters['l{}reg'.format(nl)] =[None,0.2]
        parameters['l{}act'.format(nl)] = ['relu']
    '''

    ''' Optimize lr '''
    parameters = {
        'nl':[6],
        'lr':(0.005 , 0.15, 100),
        'batch_size':[2000,5000], # , 5000 , 10000] ,
        'epochs':[20,40],
        'w':[True],
        'loss':[None]
    }

    parameters['l0nn'] = [100]
    parameters['l0reg'] =[None]
    parameters['l0act'] = ['relu']

    parameters['l1nn'] = [100]
    parameters['l1reg'] =[None]
    parameters['l1act'] = ['tanh']

    parameters['l2nn'] = [100]
    parameters['l2reg'] =[None]
    parameters['l2act'] = ['tanh']

    parameters['l3nn'] = [100]
    parameters['l3reg'] =[None]
    parameters['l3act'] = ['relu']

    parameters['l4nn'] = [100]
    parameters['l4reg'] =[None]
    parameters['l4act'] = ['relu']

    parameters['l5nn'] = [100]
    parameters['l5reg'] =[None]
    parameters['l5act'] = ['relu']
    
    
    t = ta.Scan(X_train_, y_train_, parameters, MakeModelOptimizer , x_val=X_val_ , y_val=Y_val_,shuffle=False,grid_downsample=0.2,print_params=False, last_epoch_value=True,disable_progress_bar=False, clear_tf_session=True)


def TrainOptimizedBinaryJuneCMSWorkshop(model_name , tsks = tasks , nepochs = 10):
    trainer = DNNTrain.DNNTrain( tasks=tsks , train_ratio=0.85)
    trainer.MakeModel(layers=[(50,None,'relu'),(100,None,'tanh'),(100,None,'tanh'),(100,None,'relu')] ,
                      flatten=True , optimizer='adam' , loss=None , LR=0.02 )
    fit_res = trainer.Fit(batch_size=500 , epochs=nepochs , validation_split=0.3 , weight=False)
    trainer.SaveModel( "OptimizedBinaryJuneCMSWorkshop" + model_name , 1 , 1 )
    
def TrainOptimizedBinaryJuneCMSWorkshopWeighted(model_name,b_size=500,lr=0.015, tsks = tasks , nepochs = 10):
    trainer = DNNTrain.DNNTrain( tasks=tsks , train_ratio=0.85)
    trainer.MakeModel(layers=[(100,None,'relu'),(100,0.2,'relu'),(100,0.2,'relu'),(100,None,'relu'),(200,0.2,'relu'),(100,None,'relu')] ,
                      flatten=True , optimizer='adam' , loss=None , LR=lr )
    fit_res = trainer.Fit(batch_size=b_size , epochs=nepochs , validation_split=0.3 , weight=True)
    trainer.SaveModel( "OptimizedBinaryJuneCMSWorkshopWeighted" + model_name , 1 , 1 , skipfirstbins=5 )

def TrainMultiClass(b_size=500,lr=0.015, toskipinplot = 5 , model_name=''):
    trainer = DNNTrain.DNNTrain( tasks=tasks_mclass , train_ratio=0.85)
    trainer.MakeModel(layers=[(100,None,'relu'),(100,0.2,'relu'),(100,0.2,'relu'),(100,None,'relu'),(200,0.2,'relu'),(100,None,'relu')] ,
                      flatten=True , optimizer='adam' , loss=None , LR=lr )
    fit_res = trainer.Fit(batch_size=b_size , epochs=50 , validation_split=0.3 , weight=True)
    trainer.SaveModel( "MultiClass" + model_name , 1 , 1 , skipfirstbins=toskipinplot )
    
