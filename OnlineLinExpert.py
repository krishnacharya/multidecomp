import numpy as np
from river import linear_model
from ExpertsAbstract import *

class Online_linearexpert(Expert):
    def __init__(self, l2_pen = 1.0):
        self.name = "River_LinearExpert"
        self.model_rr = linear_model.LinearRegression(l2 = l2_pen)
        self.loss_tarr = []
        self.y_predarr = [] #history of y's precicted by the expert, ALL are ridge regrssors predictions, least squares is just used for best in hindisght calcs
        
    def get_ypred_t(self, X_t):
        '''
        feature X_t must be a pandas dataframe with single row be careful, not numpy array like the others
        '''
        ypred = np.clip(self.model_rr.predict_many(X_t).iloc[0], 0.0, 1.0) # ypred here is a numpy float scalar
        self.y_predarr.append(ypred)
        return ypred
    
    def update_t(self, X_t, y_t):
        '''
        CALLED immediately after get_ypred_t
        X_t is the pandas dataframe with single row (1 x number of features)
        y_t is the TRUE label given by environment, pandas series with single element in [0, 1], so to get that elements scalar value access using [0]
        this is another external function to be called to update the model in round t
        '''
        self.model_rr.learn_many(X_t, y_t) # update internal state for ridge, X_t is dataframe, y_t is series both have one row/element
        self.loss_tarr.append((self.y_predarr[-1] - y_t[0])**2) #ypred 

    def make_all_numpyarr(self):
        '''
            numpifys the lists - ypredarr and loss_tarr, saves disk space while saving the object with joblib
        '''
        self.loss_tarr = np.array(self.loss_tarr)
        self.y_predarr = np.array(self.y_predarr)
