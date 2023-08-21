import numpy as np
from tqdm import tqdm
from ImplementableExpertAbstract import *
from skmultiflow.trees import HoeffdingTreeRegressor
import pandas as pd
class HoeffdingTree_alwaysactive(ImplementableExpert):
    '''
    Online Hoeffding tree, this is not a meta expert,
    its implementable
    '''
    def __init__(self):
        self.name = "Hoeffding tree skmultiflow always active"
        self.model = HoeffdingTreeRegressor()
        self.loss_tarr = []
        self.y_predarr = []
        
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
    
    def update_t(self, X_t, y_t) -> None: #X_t here is the same one above in get_ypred_t, so can use the cached self.x_val
        '''
            X_t is a row of dataframe X_dat.iloc[[t]], so is y_t, y_dat.iloc[[t]]
        '''
        self.model.partial_fit(self.x_val, [y_t.iloc[0]]) # partial fit needs both in list/np like format
        self.loss_tarr.append((self.y_predarr[-1] - y_t.iloc[0])**2) #ypred 

    def fill_subsequence_losses(self, A_t):
        self.cumloss_groupwise_oridge = []
        N = A_t.shape[1]
        loss_groupwise_oridge = []
        loss_oridge_tarr = np.array(self.loss_tarr)
        for gnum in range(N): # build cumulative loss for  on each group subsequence
            loss_groupwise_oridge.append(loss_oridge_tarr[A_t[:, gnum].astype(bool)]) # select those losses where group gnum active
            self.cumloss_groupwise_oridge.append(np.cumsum(loss_groupwise_oridge[-1])) #cumulative sum of the previous
   
