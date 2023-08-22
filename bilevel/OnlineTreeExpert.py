import numpy as np
from river import tree
import pandas as pd
from bilevel.ExpertsAbstract import *

class OnlineHoeffdingTree(Expert):
    def __init__(self, X_datdict : dict, y_dat : pd.DataFrame, max_depth = None):
        self.name = "River Hoeffing tree"
        self.model = tree.HoeffdingTreeRegressor(max_depth = max_depth) 
        self.loss_tarr = []
        self.y_predarr = [] #history of y's precicted by the expert, ALL are ridge regrssors predictions, least squares is just used for best in hindisght calcs
        self.X_datdict = X_datdict # pd dataframe to_dict in oriented in record format
        self.y_dat = y_dat
        self.cumloss_groupwise = None # this is for the expert's loss when masked on subsequences
        
    def get_ypred_t(self, t):
        '''
        CALLED FIRST to get the prediction hat{y}_t
        '''
        y_temp = self.model.predict_one(self.X_datdict[t]) # scalar output by predict_one river
        self.y_predarr.append(np.clip(y_temp, 0.0, 1.0))
        return self.y_predarr[-1]

    def update_t(self, t):
        '''
        CALLED SECOND after get_ypred_t(), to update internal state of the expert
        '''
        y_t = self.y_dat.iloc[t] #true label scalar
        self.model.learn_one(self.X_datdict[t], y_t) # partial fit needs both in list/np like format
        self.loss_tarr.append((self.y_predarr[-1] - y_t)**2) #ypred 

    def cleanup(self):
        self.X_datdict = None
        self.y_dat = None
        self.loss_tarr = np.array(self.loss_tarr)
        self.y_predarr = np.array(self.y_predarr)