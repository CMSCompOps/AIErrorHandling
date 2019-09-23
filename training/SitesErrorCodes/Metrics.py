from tensorflow.keras import backend as K
from tensorflow.contrib import eager
import tensorflow as tf
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve,auc,roc_auc_score,f1_score

def f1K(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def my_roc_auc_score(y_true , y_pred):
    return roc_auc_score( y_true.numpy() , y_pred.numpy() )

def auroc(y_true, y_pred):
    """
    create a method to calculated roc score as a metric
    """
    auroc_py_function = eager.py_func(my_roc_auc_score, (y_true, y_pred), tf.double)
    return auroc_py_function

def my_f1(y_true , y_pred):
    a,b = f1_score( y_true.numpy() , y_pred.numpy() )
    return a

def f1(y_true, y_pred):
    f1_py_function = eager.py_func( my_f1 , (y_true, y_pred) , tf.double )
    return f1_py_function


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
