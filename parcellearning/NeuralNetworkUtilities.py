#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:35:45 2017

@author: kristianeschenburg
"""

import sys
sys.path.append('..')
import numpy as np

import classifierUtilities as cu

from keras import callbacks, optimizers
from keras.models import Sequential
from keras.layers import Dense,normalization,Activation
from keras.utils import to_categorical


class ConstrainedCallback(callbacks.Callback):
    
    """
    Callback to test neighborhood-constrained accuracy.
    """
    
    def __init__(self,x_val,y_true,y_OHL,mappings,metricKeys):
        
        """
        Parameters:
        - - - - -
            x_val : validation data feature matrix
            y_true : validation data response vector
            y_OHL : validation data one-hot matrix
            
            mappings : binary matrix of mapping results, where each row is a
                            sample, and each column is a label.  If a sample
                            mapped to a label during surface registration, the
                            index [sample,label] = 1.
            
            metricKeys : if we wish to save the loss and accuracy using the 
                        contsrained method, provide key names in format 
                        [lossKeyName, accKeyName] to save these values in a 
                        dictionary
                        
        The Keras model requires one-hot arrays for evaluation of the model --
        it does not accept multi-valued integer vectors.
        """

        self.mappings = mappings
        self.x_val = x_val
        
        self.y_true = y_true
        self.y_OHL = y_OHL
        
        self.metricKeys = metricKeys
        
        self.metrics = {k: [] for k in metricKeys}

    def on_epoch_end(self, epoch, logs={}):
        
        x = self.x_val
        y_true = np.squeeze(self.y_true)
        y_OHL = self.y_OHL
        mappings = self.mappings

        # Compute the prediction probability of samples
        predProb = self.model.predict_proba(x)

        # Contrain prediction probabilities to surface registration results
        threshed = mappings*(predProb[:,1:]);
        
        # Maximum-probability classification
        y_pred = np.squeeze(np.argmax(threshed,axis=1))
        y_pred = np.squeeze(y_pred + 1)

        # Evalute the loss and accuracy of the model
        loss,_ = self.model.evaluate(x, y_OHL, verbose=0)
        acc = np.mean(y_true == y_pred)
        
        self.metrics[self.metricKeys[0]].append(loss)
        self.metrics[self.metricKeys[1]].append(acc)
        
        print('\n{}: {}, {}: {}\n'.format(self.metricKeys[0],loss,self.metricKeys[1],acc))
        
def predict(x_test,match,model,power=1):
    
        """
        Method to predict cortical map.
        
        Parameters:
        - - - - -
            x_test : member test data in numpy array format
            match : matching matrix (binary or frequency)
            power : power to raise matching data to
            
        Returns:
        - - - -
            baseline : prediction probability of each test sample, for each
                        training class
            thresholded : prediction probabilities, constrained (and
                            possibly weighted) by matches
            predicted : prediction vector of test samples
            power : power to raise mapping frequency to
        """
        
        match = cu.matchingPower(match,power)

        baseline = model.predict(x_test,verbose=0)
        thresholded = match*baseline[:,1:]
        predicted = np.argmax(thresholded,axis=1)+1
        
        return [baseline,thresholded,predicted]