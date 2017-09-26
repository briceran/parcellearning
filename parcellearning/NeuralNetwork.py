#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:57:12 2017

@author: kristianeschenburg
"""

import numpy as np
import pickle
import sklearn

import inspect

import classifierUtilities as cu
import dataUtilities as du
import NeuralNetorkCallbacks as nnc

from keras import callbacks, optimizers
from keras.models import Sequential
from keras.layers import Dense,normalization,Activation
from keras.utils import to_categorical

class Network(object):
    
    """
    Class to build, fit, and predict using a feed-forward neural network.
    """
    
    def __init__(self,layers,node_structure,eval_factor=0.1,epochs=50,
                 batch_size=256,rate=0.001,optimizer='rmsprop'):
        
        """
        Parameters:
        - - - - -
            eval_factor : percentage of training data to use as validation
                            data (will default to 1 if percentage is too high
                            or too low in relation to training size)
            
            layers : number of layers in the network
            
            node_structure : (int,list) specifies the number of nodes per layer
                                if integer is provided, will generate same
                                number of nodes per layer.  If a list is
                                provided, will compare length of list to
                                number of layers.  If they match, will add
                                number of nodes, node_structure[i] to layer[i].
                                If they don't match, will add node_structure[1]
                                nodes per layer.
                                
            input_dim : number of features in training data
            output_dim : number of expected classes to predict
            
            epochs : number of fitting iterations
            batch_size : size of each sub-training batch
            rate : gradient step size
            optimizer : optimization scheme (rmsprop,sgd)
        """
        
        if eval_factor > 1 or eval_factor < 0:
            raise ValueError('Evaluation size is outside of accepted bounds.')
        else:
            self.eval_factor = eval_factor
        
        if layers < 0:
            raise ValueError('Number of layers must be at least 0.')
        self.layers = layers
        
        if isinstance(node_structure,list):
            if len(node_structure) != layers:
                node_structure = node_structure[0]
        if isinstance(node_structure,int):
            node_structure = list(np.repeat(node_structure,layers))
        
        self.node_structure = node_structure
        
        if epochs < 1:
            raise ValueError('The number of epochs must be at least 1.')
        else:
            self.epochs = epochs
        
        if not isinstance(batch_size,int):
            raise ValueError('Batch size must be an integer value.')
        else:
            self.batch_size = batch_size
        
        if rate <= 0:
            raise ValueError('Invalid rate parameter.')
        else:
            self.rate = rate
        
        if optimizer not in ['rmsprop','sgd']:
            raise ValueError('Invalid optimization function.')
        else:
            self.optimizer = optimizer
            
        if optimizer == 'rmsprop':
            opt = optimizers.RMSprop(lr=rate, rho=0.9, epsilon=1e-08, decay=0.0)
        else:
            opt = optimizers.SGD(lr=rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.opt_object = opt
        
    def set_params(self,**params):
        
        """
        Update parameters with user-specified dictionary.
        """
        
        args,_,_,_ = inspect.getargspec(self.__init__)
        
        if params:
            for key in params:
                if key in args:
                    setattr(self,key,params[key])
            
    def build(self,input_dimension,output_dimension):
        
        """
        Build the network using the initialized parameters.
        """
        
        input_dimension = self.input_dim
        output_dimension = self.output_dim

        model = Sequential()
        model.add(Dense(128,input_dim=input_dimension,init='uniform'))
        model.add(normalization.BatchNormalization())
        model.add(Activation('relu'))
        
        for i,l in enumerate(np.arange(self.layers)):
            
            model.add(Dense(self.node_structure[i],init='uniform'))
            model.add(normalization.BatchNormalization())
            model.add(Activation('relu'))

        # we can think of this chunk as the output layer
        model.add(Dense(output_dimension+1, init='uniform'))
        model.add(normalization.BatchNormalization())
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy',optimizer=self.opt_object,
                      metrics=['accuracy'])
        
        self.model = model
        
    def fit(self,x_train,y_train,m_train,labelSet):
        
        """
        Fit the model on the training data, after processing the validation
        data from the training set.  The validation data is used to monitor
        the performance of the model, as the model is trained.  It is expected,
        and good form, to withold the validation data from the test data as
        well.  The validation is used merely to inform parameter selection.
        
        Parameters:
        - - - - -
            x_train : training set of features
            y_train : training set of labels
            m_train : training set of matches
        """

        epochs = self.epochs
        batch_size = self.batch_size

        [training,validation] = self.extractValidation(x_train,y_train,m_train)
        
        # get dimensions of training data
        input_dim = training[0].shape[1]
        output_dim = len(labelSet)
        
        # construct the network
        self.build(input_dim,output_dim)

        # generate one-hot-encoded training and validation class matrices
        train_OHL = to_categorical(training[1],num_classes=output_dim+1)
        valid_OHL = to_categorical(validation[1],num_classes=output_dim+1)

        # initialize callback functions for training and validation data
        # these are run after each epoch
        teConstraint = nnc.ConstrainedCallback(validation[0],validation[1],valid_OHL,
                                           validation[2],['ValLoss','ValAcc'])
        trConstraint = nnc.ConstrainedCallback(training[0],training[1],train_OHL,
                                           training[2],['TrainLoss','TrainAcc'])
        
        # fit model, save the accuracy and loss after each epoch
        history = self.model.fit(training[0], train_OHL, epochs=epochs,
                            batch_size=batch_size,verbose=0,shuffle=True,
                            callbacks=[teConstraint,trConstraint])
        
        self.history = history
        self.teConstraint = teConstraint
        self.trConstraint = trConstraint
        
    def predict(self,x_test,match,power=1):
    
        """
        Method to predict neural network model cortical map.
        
        Parameters:
        - - - - -
            x_test : member test data in numpy array format
            match : matching matrix (binary or frequency)
            kwargs : optional parameter arguments
            
        Returns:
        - - - -
            baseline : prediction probability of each test sample, for each
                        training class
            thresholded : prediction probabilities, constrained (and
                            possibly weighted) by matches
            predicted : prediction vector of test samples
            power : power to raise mapping frequency to
        """

        if power == None:
            match = np.power(match,0)
        elif power == 0:
            nz = np.nonzero(match)
            match[nz] = 1
        else:
            match = np.power(match,power)

        baseline = self.model.predict(x_test,verbose=0)
        thresholded = match*baseline[:,1:]
        predicted = np.argmax(thresholded,axis=1)+1
        
        return [baseline,thresholded,predicted]
    
    def save(self,baseOutput):
        
        """
        Save model and history to files.
        
        Parameters:
        - - - - -
            baseOutput : output file base name (without extension)
        """
        
        outModel = ''.join([baseOutput,'.h5'])
        outHistory = ''.join([baseOutput,'.History.p'])
        
        self.model.save(outModel)
        
        history = self.history.history
        teConstraint = self.teConstraint
        trConstraint = self.trConstraint
        
        fullHistory = dict(history.items() + teConstraint.metrics.items() + trConstraint.metrics.items())
        
        with open(outHistory,'w') as outWrite:
            pickle.dump(fullHistory,outWrite,-1)
    
    def extractValidation(self,x_train,y_train,m_train):
        
        """
        Processing the validation data from the training set.  The validation 
        data is used to monitor the performance of the model, as the model is 
        trained.  It is expected, and good form, to withold the validation data 
        from the test data as well.  The validation is used merely to inform 
        parameter selection.
        
        Parameters:
        - - - - -
            x_train : training set of features
            y_train : training set of labels
            m_train : training set of matches
        """
        
        eval_factor = self.eval_factor
        subjects = x_train.keys()
        
        # By default, will select at least 1 validation subject from
        # the training list
        full = len(subjects)
        val = max(1,int(np.floor(eval_factor*full)))
        
        # subject lists for training and validation sets
        train = list(np.random.choice(subjects,size=(full-val),replace=False))
        valid = list(set(subjects).difference(set(train)))
        
        print '{} training subjects.'.format(len(train))
        print '{} validation subjects.'.format(len(valid))
        
        training = du.subselectDictionary(train,[x_train,y_train,m_train])
        validation = du.subselectDictionary(valid,[x_train,y_train,m_train])

        mgTD = cu.mergeValues(training[0])
        mgTL = cu.mergeValues(training[1])
        mgTM = cu.mergeValues(training[2])
        
        mgVD = cu.mergeValues(validation[0])
        mgVL = np.squeeze(cu.mergeValues(validation[1]).astype(np.int32))
        mgVM = cu.mergeValues(validation[2])
        
        N = mgTD.shape[0]
        N = sklearn.utils.shuffle(np.arange(N))
        
        mgTD = mgTD[N,:]
        mgTL = np.squeeze(mgTL[N,:].astype(np.int32))
        mgTM = mgTM[N,:]
        
        training = [mgTD,mgTL,mgTM]
        validation = [mgVD,mgVL,mgVM]
        
        return [training,validation]