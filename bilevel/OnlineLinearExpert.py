from river import linear_model
import numpy as np
import pandas as pd
from ExpertsAbstract import Expert # abstract class

class OnlineLinearExpert(Expert):
    '''
    Online linear model, could be Online ridge(l2 pen = 1) or Online Least squares(l2 pen = 0)
    '''
    def __init__(self, X_dat: pd.DataFrame, y_dat: pd.DataFrame, l2_pen = 0.0):
        self.name = "River linear model"
        self.model = linear_model.LinearRegression(l2 = l2_pen)
        self.loss_tarr = []
        self.y_predarr = []
        self.X_dat = X_dat # this just makes a reference to the dataframe, not actually copying it
        self.y_dat = y_dat

    def get_ypred_t(self, t):
        '''
        CALLED FIRST to get the prediction hat{y}_t
        '''
        y_temp = self.model.predict_many(self.X_dat.iloc[[t]]).iloc[0] # the .iloc[0] at the end is to just get the scalar
        self.y_predarr.append(np.clip(y_temp, 0.0, 1.0))
        return self.y_predarr[-1]
    
    def update_t(self, t):
        '''
        CALLED SECOND after get_ypred_t(), to update internal state of the expert
        '''
        y_row = self.y_dat.iloc[[t]]
        self.model.learn_many(self.X_dat.iloc[[t]], y_row)
        self.loss_tarr.append((self.y_predarr[-1] - y_row.iloc[0])**2)
    
    def cleanup(self):
        self.X_dat = None
        self.y_dat = None