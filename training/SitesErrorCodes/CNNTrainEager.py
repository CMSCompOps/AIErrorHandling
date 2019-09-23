"""
DNN training module to train on the action history
:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

from . import random_seed, SitesErrorCodes_path, SortingLayer, Metrics
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import tensorflow as tf
from tensorflow.contrib import eager
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve,auc
import math
import os
#from Metrics import *
#from exceptions import RuntimeError

class CNNTrainEager(tf.keras.Model) :
    def __init__(self , sorting ,tasks, train_ratio , val_ratio, xtrain=None, ytrain=None, xval=None, yval=None , xtest=None , ytest=None ):
        """
        :param Tasks tasks: an instance of Tasks class
        :param float train_ratio: a number beween 0 and 1, specifying the ratio of data that is to be used for training
        """
        tf.enable_eager_execution()
        super(CNNTrainEager, self).__init__()

        self.err_sorting = sorting
        
        self.Fit_validation_split = val_ratio
        self.Fit_train_ratio = train_ratio
        if tasks and 0<train_ratio<1:
            self.Tasks = tasks
            self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_val, self.Y_val = tasks.GetTrainTestImages( train_ratio , sorting , True , val_ratio)
        else:
            self.X_train = xtrain
            self.Y_train = ytrain
            self.X_test  = xtest
            self.Y_test  = ytest
            self.X_val   = xval
            self.Y_val   = yval
            
        self.AllFigures = {}
        
    def AddLayers(self, layers=[] , optimizer='adam' , loss=None , LR = 0.001 , moremetrics = [] ):
        """
        to make the model and compile it, if the input are binary a layer with sigmoid activation is added at the end. otherwise, a layer with softmax is inserted
        :param bool flatten: by default for the Task object it should be true
        :param list layers: list of layer, each item should be of the format of (nNeurons, regularizer, activation). if regularizer is None, no regularization is done at this layer
        :param optimizer: name of the optimizer, or an instance of the optimizer to be used
        :param str loss: name of the loss function
        """
        self.Model_layers = layers
        self.Model_optimizer = optimizer
        self.Model_loss = loss
        
        flatted = False
        firstlayer = True

        self.SortingLayer1 = SortingLayer.SortingLayer(tau=0.01 , axis = 1 , paxis=1)
        self.Layers = []
        for layer in layers :
            tpe = layer[0]

            if tpe == "dense":
                if not flatted:
                    self.Layers.append(Flatten())
                    flatted = True
            
                nNeurons = layer[1]
                regularizer = layer[2]
                activation = layer[3]

                self.Layers.append(Dense(nNeurons,
                                         kernel_regularizer= regularizers.l2(regularizer) if regularizer else None ,
                                         kernel_initializer=keras.initializers.RandomNormal(seed=random_seed),
                                         bias_initializer=keras.initializers.RandomNormal(seed=random_seed*2),
                                         activation=activation ) )
                
            elif tpe == "conv":

                filters = layer[1]
                kernel_size = layer[2]
                strides = layer[3]
                activation = layer[4]

                if firstlayer:
                    self.Layers.append( Convolution2D( filters=filters , kernel_size=kernel_size , strides=strides , activation=activation ,
                                                       input_shape = self.X_train[0].shape ,
                                                       kernel_initializer=keras.initializers.RandomNormal(seed=random_seed) ) )
                    firstlayer = False
                else:
                    self.Layers.append( Convolution2D( filters=filters , kernel_size=kernel_size , strides=strides , activation=activation ,
                                                       kernel_initializer=keras.initializers.RandomNormal(seed=random_seed) ) )
            elif tpe == 'pool' :
                self.Layers.append( layer[1] )

        self.loss = loss
        if self.Tasks.IsBinary :
            self.Layers.append( Dense( 1 , activation='sigmoid' , kernel_initializer=keras.initializers.RandomNormal(seed=random_seed*3) , bias_initializer=keras.initializers.RandomNormal(seed=random_seed*4) ) )
            if not loss:
                self.loss = tf.losses.sigmoid_cross_entropy

            self.metrics_ = [ tfe.metrics.BinaryAccuracy(0.5) , Metrics.auroc , Metrics.f1K  ]
        else:
            self.Layers.append( Dense( len(self.Tasks.all_actions) ,
                                       activation='softmax' ,
                                       kernel_initializer=keras.initializers.RandomNormal(seed=random_seed*5),
                                       bias_initializer=keras.initializers.RandomNormal(seed=random_seed*6) ) )
            
            if not loss :
                self.loss = tf.losses.softmax_cross_entropy

            self.metrics_ = [ tfe.metrics.CategoricalAccuracy(0.5), top_first_categorical_accuracy(1,"kfirst"), top_first_categorical_accuracy(2,"kfirsttwo"),top_first_categorical_accuracy(3,"kfirstthree")]

        self.metrics_ += moremetrics
            
        if optimizer == "sgd" :
            Optimizer = SGD(lr=.5)
        elif optimizer == "adam":
            Optimizer =  Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else :
            Optimizer = optimizer


    def call(self, input):
        output = self.SortingLayer1( input )
        for layer in self.Layers :
            output = layer( output )

        return output

    def Loss(self, input , true_vals):
        out = self.call( input )
        ret = self.loss( out , true_vals )
        return ret
        
    def Fit(self,batch_size=100, epochs=10 , verbose=1 , weight=True):
        """
        do the fit of training. standard parameters of keras.Model.fit
        """
        self.Fit_batch_size = batch_size
        self.Fit_epochs = epochs

        #weights = self.Tasks.ClassCounts
        #print('weights' , weights)
        for epoch in range(epochs):
            
        self.FitHistory = self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, epochs=epochs,
                                         verbose=verbose , validation_data=(self.X_val , self.Y_val) , shuffle=False , class_weight=weights if weight else None )
                                             

        self.Pred_on_Train = self.model.predict( self.X_train )
        self.Pred_on_Train_List = zip( self.Pred_on_Train.ravel() , self.Y_train )
        self.Pred_on_Test =  self.model.predict( self.X_test  )
        self.Pred_on_Test_List  = zip( self.Pred_on_Test.ravel()  , self.Y_test )
        return self.FitHistory
    
    def PlotFitAndTestMetrics(self , fig=None , nshuffels=100 , skipfirstbins=0):
        """
        plot the evolution of metrics and loss during the fit for training and evaluation samples and comapre it with the test results
        if fit has not been done, it is called here with default parameters
        it evaluates the model on the test sample
        to assess the uncertainty on the final metric values, it evaluates the model on shuffled subsets of the dataset and draw the mean and stdev of the distribution. the size of each subset is equal the batch size given for the fit.
        :param obj fig: the figure to plot metric and loss evolution. if None a new one is created
        :param int nshuffels: number of subsets to evaluate uncertainty
        """
        if not hasattr(self, "FitHistory"):
            self.Fit()
            
        if fig:
            self.FitFig = plt.figure( fig.number )
        else:
            self.FitFig = plt.figure()
        
        
        self.EvaluationTest = self.model.evaluate( self.X_test , self.Y_test , batch_size=int(len(self.Y_test)) )
        self.EvaluationOnShuffels = [ [] for a in self.EvaluationTest ]
        for i in range(0,nshuffels):
            x_test_sh , y_test_sh = self.Tasks.GetShuffledImages( self.Fit_batch_size , self.err_sorting ) 
            evaluation_i = self.model.evaluate( x_test_sh , y_test_sh , batch_size=self.Fit_batch_size )
            for a in range(0,len(evaluation_i) ):
                self.EvaluationOnShuffels[a].append( evaluation_i[a] )
        self.EvaluationOnShuffels = [ (np.mean(a) , np.std(a)) for a in self.EvaluationOnShuffels ]
        x_epochs = range(1,self.Fit_epochs+1)
        plots = {}
        _colors = 'bgrcmykw'
        for index,metric in enumerate(self.model.metrics_names) :
            color = _colors[index]
            p_train = plt.plot( x_epochs[skipfirstbins:] , self.FitHistory.history[metric][skipfirstbins:] , color, label=metric  , linestyle="-" , linewidth=2 )
            p_validation = plt.plot( x_epochs[skipfirstbins:] , self.FitHistory.history['val_'+metric][skipfirstbins:] , color, label=metric  , linestyle="--" , linewidth=1 )
            p_test = plt.plot( [self.Fit_epochs] , self.EvaluationTest[index] , color, label=metric  , marker='o' )
            if nshuffels>0 :
                p_band = plt.errorbar( [self.Fit_epochs] , [self.EvaluationOnShuffels[index][0]] , yerr=[self.EvaluationOnShuffels[index][1]] , fmt=color, label=metric  , marker='x' )
                plots[metric] = ( p_train[0] , p_validation[0] , p_test[0] , p_band[0] )
            else :
                plots[metric] = ( p_train[0] , p_validation[0] , p_test[0] ) 

        plt.legend( [ plots[a] for a in plots] , [a for a in plots] , loc='best' , numpoints=1,
                    handler_map={tuple: HandlerTuple(ndivide=None)} )

        return self.FitFig

    def OverTrainingPlot(self):
        if hasattr(self, "OvertrainingPlot"):
            return self.OvertrainingPlot
        if not hasattr(self, "FitHistory"):
            self.Fit()
        #self.ROC()

        y_sig_train = self.Pred_on_Train[ self.Y_train > 0.5 ]
        y_bkg_train = self.Pred_on_Train[ self.Y_train < 0.5 ]

        y_sig_test = self.Pred_on_Test[ self.Y_test > 0.5 ]
        y_bkg_test = self.Pred_on_Test[ self.Y_test < 0.5 ]
        
        self.OvertrainingPlot = plt.figure()
        start, end, bin_width = 0.5 , 1.0 , 0.02
        bins = np.arange( start , end+bin_width , bin_width )
        Type= 'step' # if test else 'bar'
        plt.hist( [y_sig_train,y_bkg_train],density=True,histtype=Type,color=['r','b'] ,bins=bins , linestyle="--" , linewidth=2 , label=['a' , 'b'] )

        y_sig_test_hist, y_sig_test_bins = np.histogram( y_sig_test , bins=bins , density=True )
        y_bkg_test_hist, y_bkg_test_bins = np.histogram( y_bkg_test , bins=bins , density=True )
        bin_centers = bins+bin_width/2
        plt.errorbar( bin_centers[:-1] , y_sig_test_hist, yerr=np.sqrt(y_sig_test_hist)/math.sqrt(len(y_sig_test_hist)), xerr=[bin_width/2]*len(y_sig_test_hist) , fmt='r*' , label='c' )
        plt.errorbar( bin_centers[:-1] , y_bkg_test_hist , yerr=np.sqrt(y_bkg_test_hist)/math.sqrt(len(y_bkg_test_hist)), xerr=[bin_width/2]*len(y_bkg_test_hist) , fmt='b*' , label='d' )

        plt.axvline( self.roc_test_80p_info[3] , color='k', linestyle='dashed', linewidth=1 , label='e' )
        plt.legend(loc='best')
        
        #trans_angle = plt.gca().transData.transform_angles( np.array((90,)),l2.reshape((1, 2)))[0]
        
        return self.OvertrainingPlot
        

    def PlotPredictionForActions(self , ACTION_INDEX , ttc=0):
        """
        Plot predictions for all the tasks with a given action. It returns the produced plot.
        :param int ACTION_INDEX: the index of the action
        :param str ttc: 0 for test dataset, 1 for train dataset, 2 for test+train
        """
        if self.Tasks.IsBinary :
            raise RuntimeError("DNNTrain::PlotPredictionForActions is not callable for binary datasets")
        if ACTION_INDEX >= len(self.Tasks.all_actions) :
            raise RuntimeError("DNNTrain::PlotPredictionForActions is called with an ACTION_INDEX greater than available actions")
        
        truth = [ i==ACTION_INDEX for i in range(len(self.Tasks.all_actions) ) ]
        prediction_on_train_1 = self.Pred_on_Train[ np.all( self.Y_train == truth , 1)]
        prediction_on_test_1 = self.Pred_on_Test[ np.all( self.Y_test == truth , 1)]
        if ttc==0:
            prediction = prediction_on_test_1
        elif ttc==1:
            prediction = prediction_on_train_1
        elif ttc==2:
            prediction = np.concatenate( (prediction_on_test_1 , prediction_on_train_1) )
        else:
            raise RuntimeError("DNNTrain::PlotPredictionForActions ttc parameter should be 0,1 or 2. %d is given" % ttc)
        argsorted = prediction.argsort()
        a = np.take_along_axis( prediction , argsorted , 1 )
        figName = "PlotPredictionForActions%d" % ACTION_INDEX
        self.AllFigures[figName] = plt.figure()
        plt.hist( [ prediction[:,0]/a[:,2] , prediction[:,1]/a[:,2] , prediction[:,2]/a[:,2] ] , density=True , histtype='step' , color=['r', 'g' , 'b' ] , label=self.Tasks.all_actions  )
        plt.legend(loc='best')
        plt.title( "Predictions for " + self.Tasks.all_actions[ACTION_INDEX] + " actions")
        return self.AllFigures[figName]
    
    def ROC(self):
        """
        plot ROC curve for test dataset
        """
        if hasattr(self, "roc_plot"):
            return self.roc_plot
        
        self.roc_fpr, self.roc_tpr, self.roc_thresholds = roc_curve(self.Y_test, self.Pred_on_Test.ravel() )
        self.roc_auc = auc(self.roc_fpr , self.roc_tpr)

        self.roc_plot = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(self.roc_fpr, self.roc_tpr, label='Keras (area = {:.3f})'.format(self.roc_auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        idx80p = np.abs(self.roc_tpr - 0.8).argmin()
        self.roc_test_80p_info = ( idx80p , self.roc_tpr[idx80p] , self.roc_fpr[ idx80p ] , self.roc_thresholds[ idx80p ] )
        print(self.roc_test_80p_info)
        return self.roc_plot
        #plt.show()
        
    def Test(self , plot_roc = True):
        """
        run the test and returns a map of predictions and true values.
        """

        if plot_roc:
            self.ROC()

        
        self.OverTrainingPlot( self.Y_train , pred_on_train , self.Y_test  , self.y_prediction.ravel() )

        if not self.Tasks.IsBinary :
            average_per_true = np.zeros( [len(self.Tasks.all_actions)+1, len(self.Tasks.all_actions)] )
            for pre,true in results:
                index = list( true ).index(1)
                for i in range (0, len(average_per_true[ index ]) ):
                    average_per_true[index][i] += pre[i]
                average_per_true[-1][ index ] += 1

            for iii in range(0,len(self.Tasks.all_actions) ) :
                row = average_per_true[iii]
                total = average_per_true[-1][iii]
                if total != 0 :
                    row /= total

            print(average_per_true)


    def SaveModel(self, file_name , model_details , trainingdata_details , skipfirstbins=0):
        """
        Save the model to two files :
        one hdf5 file and one json file under the models subdirectory of the current package are created with file_name
        :param str file_name: the name of the file, without any extra extension
        :param int model_details: the integer id of the model. this value will be stored in json file for future references. authors and developers should keep track of its values to make sense.
        :param int trainingdata_details: the integer id of the dataset that was used for training the model. this value will be stored in json file for future references. authors and developers should keep track of its values to make sense.
        """
        params = { 'Fit_validation_split':self.Fit_validation_split , 'Fit_batch_size' : self.Fit_batch_size ,
                   'Fit_epochs' : self.Fit_epochs , 'Fit_train_ratio' : self.Fit_train_ratio , 
                   'Model_layers' : self.Model_layers ,
                   'Model_optimizer' : self.Model_optimizer , 'Model_loss': self.Model_loss }

        save_model( self.model , SitesErrorCodes_path + "/models/" + file_name + ".hdf5" )
        with open( SitesErrorCodes_path + "/models/" + file_name + '.json', 'w') as fp:
            json.dump({'all_sites':self.Tasks.all_sites ,
                       'all_errors':self.Tasks.all_errors ,
                       'all_actions':self.Tasks.all_actions ,
                       'TiersOnly':self.Tasks.TiersOnly ,
                       'IsBinary':self.Tasks.IsBinary ,
                       'model':model_details,
                       'trainingdata':trainingdata_details,
                       'informatino':params} , fp)
        if not os.path.isdir( SitesErrorCodes_path + "/models/" + file_name ):
            os.mkdir( SitesErrorCodes_path + "/models/" + file_name )
        plt.switch_backend('agg')
        self.PlotFitAndTestMetrics(skipfirstbins=skipfirstbins).savefig( SitesErrorCodes_path + "/models/" + file_name + '/Metrics.png' )
        if self.Tasks.IsBinary:
            self.ROC().savefig( SitesErrorCodes_path + "/models/" + file_name + '/ROC.png' )
            self.OverTrainingPlot().savefig( SitesErrorCodes_path + "/models/" + file_name + '/OverTrainingPlot.png' )
        else :
            for action_index in range(0,len(self.Tasks.all_actions) ) :
                self.PlotPredictionForActions( action_index , 0 ).savefig( SitesErrorCodes_path + "/models/" + file_name + "/pred_for_action%d_intest.png" % action_index )
                self.PlotPredictionForActions( action_index , 1 ).savefig( SitesErrorCodes_path + "/models/" + file_name + "/pred_for_action%d_intrain.png" % action_index )
                self.PlotPredictionForActions( action_index , 2 ).savefig( SitesErrorCodes_path + "/models/" + file_name + "/pred_for_action%d_inall.png" % action_index )


                
