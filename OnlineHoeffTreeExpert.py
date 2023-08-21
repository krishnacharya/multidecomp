import numpy as np
from skmultiflow.trees import HoeffdingTreeRegressor
from ExpertsAbstract import *

class HoeffdingTree_expert(Expert):
    def __init__(self):
        self.name = "Hoeffding tree skmultiflow expert"
        self.model = HoeffdingTreeRegressor()
        self.loss_tarr = []
        self.y_predarr = [] #history of y's precicted by the expert, ALL are ridge regrssors predictions, least squares is just used for best in hindisght calcs
        
    def get_ypred_t(self, X_t):
        '''
        X_t:
            is a row of pandas dataframe X_dat.iloc[[t]]
            for skmultiflow hoeffding tree we need a np array of shape (nsamples=1, nfeatures)
        '''
        self.x_val = X_t.to_numpy().reshape(1,-1)
        ypred = np.clip(self.model.predict(self.x_val)[0], 0.0, 1.0) # ypred here is a numpy float scalar
        self.y_predarr.append(ypred)
        return ypred
    
    def update_t(self, X_t, y_t):
        '''
            X_t is a row of dataframe X_dat.iloc[[t]], so is y_t, y_dat.iloc[[t]]
        '''
        self.model.partial_fit(self.x_val, [y_t.iloc[0]]) # partial fit needs both in list/np like format
        self.loss_tarr.append((self.y_predarr[-1] - y_t.iloc[0])**2) #ypred 

    def make_all_numpyarr(self):
        '''
            numpifys the lists - ypredarr and loss_tarr, saves disk space while saving the object with joblib
        '''
        self.loss_tarr = np.array(self.loss_tarr)
        self.y_predarr = np.array(self.y_predarr)