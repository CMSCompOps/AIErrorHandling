"""
DNN training module to train on the action history
:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

from . import random_seed, SitesErrorCodes_path
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
from sklearn.metrics import roc_curve,auc,roc_auc_score
import math

def my_roc_auc_score(y_true , y_pred):
    return roc_auc_score( y_true.numpy() , y_pred.numpy() )

def auroc(y_true, y_pred):
    """
    create a method to calculated roc score as a metric
    """
    auroc_py_function = eager.py_func(my_roc_auc_score, (y_true, y_pred), tf.double)
    return auroc_py_function


def top_first_categorical_accuracy(kk , name):
    """
    a method to create methods to be used as a metric.
    :param int kk: the accuracy in the first kk categories will be checked
    :param str name: the name of the metric
    """
    def ktop(x , y ):
        return metrics.top_k_categorical_accuracy(x, y, kk)

    ktop.__name__ = name
    return ktop

class DNNTrain :
    def __init__(self , tasks , train_ratio=0.8):
        """
        :param Tasks tasks: an instance of Tasks class
        :param float train_ratio: a number beween 0 and 1, specifying the ratio of data that is to be used for training
        """
        self.Fit_train_ratio = train_ratio
        
        self.Tasks = tasks
        self.X_train, self.y_train, self.X_test, self.y_test = tasks.GetTrainTestDS( train_ratio , True )

        self.X_train = self.X_train.astype('int16')
        self.X_test = self.X_test.astype('int16')

        if self.Tasks.IsBinary :
            self.Y_train = self.y_train.astype('int16')
            self.Y_test = self.y_test.astype('int16')
        else:
            self.Y_train = np_utils.to_categorical(self.y_train, len(tasks.all_actions) , 'int8')
            self.Y_test = np_utils.to_categorical(self.y_test, len(tasks.all_actions) , 'int8')

        print( set( self.Y_test ) )

    def MakeModel(self, flatten=True , layers=[] , optimizer='adam' , loss='categorical_crossentropy' ):
        """
        to make the model and compile it, if the input are binary a layer with sigmoid activation is added at the end. otherwise, a layer with softmax is inserted
        :param bool flatten: by default for the Task object it should be true
        :param list layers: list of layer, each item should be of the format of (nNeurons, regularizer, activation). if regularizer is None, no regularization is done at this layer
        :param optimizer: name of the optimizer, or an instance of the optimizer to be used
        :param str loss: name of the loss function
        """
        self.Model_Flatten = flatten
        self.Model_layers = layers
        self.Model_optimizer = optimizer
        self.Model_loss = loss
        
        self.model = Sequential()
        if flatten :
            self.model.add(Flatten())

        for layer in layers :
            nNeurons = layer[0]
            regularizer = layer[1]
            activation = layer[2]

            self.model.add(Dense(nNeurons,
                                 kernel_regularizer= regularizers.l2(regularizer) if regularizer else None ,
                                 kernel_initializer=keras.initializers.RandomNormal(seed=random_seed),
                                 bias_initializer=keras.initializers.RandomNormal(seed=random_seed*2),
                                 activation=activation ) )

        if self.Tasks.IsBinary :
            self.model.add( Dense( 1 , activation='sigmoid' , kernel_initializer=keras.initializers.RandomNormal(seed=random_seed*3) , bias_initializer=keras.initializers.RandomNormal(seed=random_seed*4) ) )
        else:
            self.model.add( Dense( len(self.Tasks.all_actions) ,
                                   activation='softmax' ,
                                   kernel_initializer=keras.initializers.RandomNormal(seed=random_seed*5),
                                   bias_initializer=keras.initializers.RandomNormal(seed=random_seed*6) ) )

            
        if optimizer == "sgd" :
            Optimizer = SGD(lr=.5)
        elif optimizer == "adam":
            Optimizer =  Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else :
            Optimizer = optimizer

        self.model.compile(
            loss=loss ,#'categorical_crossentropy', 'mean_squared_error', 'categorical_crossentropy' 'mean_absolute_error'
            optimizer=Optimizer,
            metrics=['accuracy' , auroc]
            # , 'categorical_accuracy' , top_first_categorical_accuracy(1,"kfirst"), top_first_categorical_accuracy(2,"kfirsttwo"),top_first_categorical_accuracy(3,"kfirstthree")]
        )

    def Fit(self,batch_size=100, epochs=10 , validation_split=0.0 , verbose=1):
        """
        do the fit of training. standard parameters of keras.Model.fit
        """
        self.Fit_batch_size = batch_size
        self.Fit_epochs = epochs
        self.Fit_validation_split = validation_split
        
        self.FitHistory = self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose , validation_split=validation_split )
        self.Pred_on_Train = self.model.predict( self.X_train )
        #self.Pred_on_Train_List = zip( self.Pred_on_Train.ravel() , self.Y_train )
        self.Pred_on_Test =  self.model.predict( self.X_test  )
        #self.Pred_on_Test_List  = zip( self.Pred_on_Test.ravel()  , self.Y_test )
        return self.FitHistory

    def PlotFitAndTestMetrics(self , fig=None , nshuffels=100):
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
            x_test_sh , y_test_sh = self.Tasks.GetShuffledDS( self.Fit_batch_size ) 
            evaluation_i = self.model.evaluate( x_test_sh , y_test_sh , batch_size=self.Fit_batch_size )
            for a in range(0,len(evaluation_i) ):
                self.EvaluationOnShuffels[a].append( evaluation_i[a] )
        self.EvaluationOnShuffels = [ (np.mean(a) , np.std(a)) for a in self.EvaluationOnShuffels ]
        x_epochs = range(1,self.Fit_epochs+1)
        plots = {}
        _colors = 'bgrcmykw'
        for index,metric in enumerate(self.model.metrics_names) :
            color = _colors[index]
            p_train = plt.plot( x_epochs , self.FitHistory.history[metric] , color, label=metric  , linestyle="-" , linewidth=2 )
            p_validation = plt.plot( x_epochs , self.FitHistory.history['val_'+metric] , color, label=metric  , linestyle="--" , linewidth=1 )
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
        self.ROC()

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


    def SaveModel(self, file_name , model_details , trainingdata_details):
        """
        Save the model to two files :
        one hdf5 file and one json file under the models subdirectory of the current package are created with file_name
        :param str file_name: the name of the file, without any extra extension
        :param int model_details: the integer id of the model. this value will be stored in json file for future references. authors and developers should keep track of its values to make sense.
        :param int trainingdata_details: the integer id of the dataset that was used for training the model. this value will be stored in json file for future references. authors and developers should keep track of its values to make sense.
        """
        params = { 'Fit_validation_split':self.Fit_validation_split , 'Fit_batch_size' : self.Fit_batch_size ,
                   'Fit_epochs' : self.Fit_epochs , 'Fit_train_ratio' : self.Fit_train_ratio , 
                   'Model_Flatten' : self.Model_Flatten, 'Model_layers' : self.Model_layers ,
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
