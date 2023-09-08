from river import linear_model
import numpy as np
from tqdm import tqdm

class BestLS_Hindsight():
    '''
    Finds the best in hinsight at each time for a subsequence defined by a specific group
    '''
    def __init__(self, X_df, y_df):
        self.name = "Best_LS_Hindsight" #best least squares in hinsight, calculated online
        # self.time = 0
        self.X_df = X_df # the rows slices of the original ACS dataframe where this group is active
        self.y_df = y_df 
        self.T = len(self.y_df)
        self.loss_tarr = np.zeros(self.T)
        self.model_ls = linear_model.LinearRegression(l2 = 0.0)
        self.fill_loss()
        self.best_sqloss = np.cumsum(self.loss_tarr)
    
    def fill_loss(self):
        for t in tqdm(range(self.T)):
            x_t = self.X_df.iloc[[t]] # pandas dataframe, single row
            y_t = self.y_df.iloc[t] # pandas series type, single element
            y_pred = self.model_ls.predict_many(x_t) # predict real value for this row
            self.loss_tarr[t] = (y_pred - y_t[0])**2 # this is the term 2 in the lhs, best in hindsight acc, not clipped
            self.model_ls.learn_many(x_t, y_t) # learn many requires dataframe for X and series for y
