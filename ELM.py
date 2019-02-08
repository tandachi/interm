# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:28:20 2016

@author: rebli
"""

import numpy as np
from numpy.linalg import inv

class ELMRegressor():
    def __init__(self, n_hidden_units, params=[]):
        self.n_hidden_units = n_hidden_units
        self.default_param_kernel_function = "rbf"
        self.default_param_c = 9
        self.default_param_kernel_params = [-15]

        # Initialized parameters values
        if not params:
            self.param_kernel_function = self.default_param_kernel_function
            self.param_c = self.default_param_c
            self.param_kernel_params = self.default_param_kernel_params
        else:
            self.param_kernel_function = params[0]
            self.param_c = params[1]
            self.param_kernel_params = params[2]

        self.output_weight = []
        self.training_patterns = []


    def fit(self, X,labels):
        X = np.column_stack([X,np.ones([X.shape[0],1])])
        self.random_weights = np.random.randn(X.shape[1],self.n_hidden_units)
        '''G = 1 / (1 + np.exp(-(X.dot(self.random_weights))))'''
        G = np.tanh(X.dot(self.random_weights))
        
        self.w_elm = np.linalg.pinv(G).dot(labels)
        

    def predict(self, X):
        X = np.column_stack([X,np.ones([X.shape[0],1])])
        '''G = 1 / (1 + np.exp(-(X.dot(self.random_weights))))'''
        G = np.tanh(X.dot(self.random_weights))
        return G.dot(self.w_elm)
        
    
   
    def iters(self,x):
        if x<0.5:
          return 0
        else:
          return 1    
          
          
    def _kernel_matrix(self, training_patterns, kernel_type, kernel_param,
                       test_patterns=None):

        """ Calculate the Omega matrix (kernel matrix).
            If test_patterns is None, then the training Omega matrix will be
            calculated. This matrix represents the kernel value from each
            pattern of the training matrix with each other. If test_patterns
            exists, then the test Omega matrix will be calculated. This
            matrix  represents the kernel value from each pattern of the
            training matrix with the patterns of test matrix.
            Arguments:
                training_patterns (numpy.ndarray): A matrix containing the
                    features from all training patterns.
                kernel_type (str): The type of kernel to be used e.g: rbf
                kernel_param (list of float): The parameters of the chosen
                    kernel.
                test_patterns (numpy.ndarray): An optional parameter used to
                 calculate the Omega test matrix.
            Returns:
                numpy.ndarray: Omega matrix
        """
        
        number_training_patterns = training_patterns.shape[0]

        if kernel_type == "rbf":
            if test_patterns is None:
                temp_omega = np.dot(
                    np.sum(training_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_training_patterns)))

                temp_omega = temp_omega + temp_omega.conj().T

                omega = np.exp(
                    -(2 ** kernel_param[0]) * (temp_omega - 2 * (np.dot(
                        training_patterns, training_patterns.conj().T))))

            else:
                number_test_patterns = test_patterns.shape[0]

                temp1 = np.dot(
                    np.sum(training_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_test_patterns)))
                temp2 = np.dot(
                    np.sum(test_patterns ** 2, axis=1).reshape(-1, 1),
                    np.ones((1, number_training_patterns)))
                temp_omega = temp1 + temp2.conj().T

                omega = \
                    np.exp(- (2 ** kernel_param[0]) *
                           (temp_omega - 2 * np.dot(training_patterns,
                                                    test_patterns.conj().T)))
        elif kernel_type == "linear":
            if test_patterns is None:
                omega = np.dot(training_patterns, training_patterns.conj().T)
            else:
                omega = np.dot(training_patterns, test_patterns.conj().T)

        elif kernel_type == "poly":
            # Power a**x is undefined when x is real and 'a' is negative,
            # so is necessary to force an integer value
            kernel_param[1] = round(kernel_param[1])

            if test_patterns is None:
                temp = np.dot(training_patterns, training_patterns.conj().T) + \
                       kernel_param[0]

                omega = temp ** kernel_param[1]
            else:
                temp = np.dot(training_patterns, test_patterns.conj().T) + \
                       kernel_param[0]

                omega = temp ** kernel_param[1]

        else:
            print("Error: Invalid or unavailable kernel function.")
            return

        return omega   
        
        
    def _local_train(self, training_patterns, training_expected_targets,
                     params=[]):

        # If params not provided, uses initialized parameters values
        if not params:
            pass
        else:
            self.param_kernel_function = params[0]
            self.param_c = params[1]
            self.param_kernel_params = params[2]

        # Need to save all training patterns to perform kernel calculation at
        # testing and prediction phase
        self.training_patterns = training_patterns

        number_training_patterns = self.training_patterns.shape[0]

        # Training phase

        omega_train = self._kernel_matrix(self.training_patterns,
                                          self.param_kernel_function,
                                          self.param_kernel_params)

        # print("fy")
        # self.output_weight = np.dot(
        #     scipy.linalg.inv((omega_train + np.eye(number_training_patterns) /
        #                       (2 ** self.param_c))), training_expected_targets)\
        #     .reshape(-1, 1)
        
        self.output_weight = np.dot(
             np.linalg.pinv((omega_train + np.eye(number_training_patterns) /
                               (2 ** self.param_c))), training_expected_targets)\
             .reshape(-1, 1)
        
        
        '''self.output_weight = np.linalg.solve(
            (omega_train + np.eye(number_training_patterns) /
             (2 ** self.param_c)),
            training_expected_targets).reshape(-1, 1)'''

        training_predicted_targets = np.dot(omega_train, self.output_weight)

        return training_predicted_targets
        
    def _local_test(self, testing_patterns):

        omega_test = self._kernel_matrix(self.training_patterns,
                                          self.param_kernel_function,
                                          self.param_kernel_params,
                                          testing_patterns)

        testing_predicted_targets = np.dot(omega_test.conj().T,
                                           self.output_weight)

        return testing_predicted_targets

   
          
          
          