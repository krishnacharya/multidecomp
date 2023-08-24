import numpy as np
import pandas as pd
import random
import itertools
#  samples = 10000, dim = 20,  group_dict : dict = None, 
#                         feat_lo = 0.0, feat_hi = 1.0
class SynthGenLinear:
    def __init__(self, **kwargs):
        '''
        ----
        Parameters
            samples: number of samples in dataset
            dim : dimensionality of features
            group_dict : names of the groups e.g. {'SEX': ['Male', 'Female'], 'RACE' : ['White', 'Black', 'Asian', 'Some-other'] }
            self.Ng: total number of groups

            feat_lo, feat_hi : bounds for the Uniform distribution U[feat_lo, feat_hi] on features
            w_lo, w_hi: bounds for Uniform distribution on weights

        '''
        self.samples = kwargs['samples']
        self.dim = kwargs['dim']
        self.group_dict = kwargs['group_dict']
        self.prob_dict = kwargs['prob_dict']
        list2d = [li for li in  self.group_dict.values()]
        self.all_groupnames = list(itertools.chain(*list2d))
        self.Ng = len(self.all_groupnames)
        self.feat_lo = kwargs['feat_lo']
        self.feat_hi = kwargs['feat_hi']
        self.w_lo = kwargs['w_lo']
        self.w_hi = kwargs['w_hi']
        self.label_noise_width = kwargs['label_noise_width']
        self.drop_sensitive = kwargs['drop_sensitive']
        self.get_feat_uniform()
        self.get_A_t()
        self.get_labels()
        self.df_synlinear = self.get_dataframe()
        self.put_active_labels_dataframe()
    
    def get_feat_gaussian_skewed(self) -> np.ndarray:
        pass

    def get_feat_uniform(self) -> np.ndarray:
        '''           
        Returns
            feat_dat of shape (# of samples, # dim) sampled from U[feat_lo, feat_hi]
        '''
        self.feat_dat = np.random.uniform(low = self.feat_lo, high = self.feat_hi, size = (self.samples, self.dim))
        return self.feat_dat

    def get_A_t(self) -> np.ndarray:
        '''
        Parameters
        prob_dict : 
            probabilities of non-atomic groups e.g. {'SEX': [0.5, 0.5], 'RACE': [0.6, 0.2, 0.1, 0.1]}, keep same key ordering as group_dict!
            within each they sum to 1.0
        ---
        Returns numpy.ndarray of shape (#samples x # of groups)
        '''
        def get_group_indicators(prob_list: list) -> list[np.ndarray]:
            inds = np.eye(len(prob_list)) # indicators e.g for prob list of len 3, (1, 0, 0), (0, 1, 0), (0, 0, 1)
            return np.array(random.choices(population=inds, weights=prob_list, k = self.samples))
        self.A_t = np.hstack([get_group_indicators(prob_list) for prob_list in self.prob_dict.values()])
        return self.A_t
    
    def get_labels(self) -> np.ndarray:
        '''
        NEEDS get features to be called before this is called
        Parameters
            w_lo, w_hi : bounds for uniform distribution U[w_lo, w_hi] on weights
        Returns
            labels for each group, using its weight, shape (# samples, # groups)
        '''
        self.weights = np.random.uniform(low = self.w_lo, high = self.w_hi, size = (self.dim, self.Ng)) # shape is (dim, # of groups)
        self.labels_allg = np.matmul(self.feat_dat, self.weights)
        noise_gaussian = np.random.normal(scale = self.label_noise_width, size = (self.samples, self.Ng))
        return self.labels_allg + noise_gaussian

    def get_dataframe(self) -> pd.DataFrame:
        self.df_feat_names = ['x_'+str(i) for i in range(self.dim)]
        self.df_label_names = ['y_' + st for st in self.all_groupnames] # y_male, y_female, y_white,...
        self.df = None
        if self.drop_sensitive:
            self.df =  pd.DataFrame(np.hstack((self.feat_dat, self.labels_allg)), columns = self.df_feat_names + self.df_label_names) 
            return self.df
        else:
            self.group_ind = ['g_' + st for st in self.all_groupnames]
            self.df = pd.DataFrame(np.hstack((self.feat_dat, self.A_t, self.labels_allg)), columns= self.df_feat_names + self.group_ind + self.df_label_names)
            return self.df

    def put_active_labels_dataframe(self) -> None:
        '''
            generates a column in self.df, which will contain a nparray, this list has all the active group labels
        '''
        binary_masked = (self.df[self.df_label_names] * self.A_t) # y_t part of dataframe multiplied by A_t mask
        active_indices = binary_masked.apply(np.flatnonzero, axis=1) # get the non zero value positions in the above
        self.df['active_labels'] = None
        self.df['bin_masked_labels'] = None
        for i, ai in enumerate(active_indices):
            self.df.at[i, 'active_labels'] = self.df[self.df_label_names].iloc[i, ai].to_numpy()
            self.df.at[i, 'bin_masked_labels'] = (self.df[self.df_label_names].iloc[i] * self.A_t[i]).to_numpy()
        return self.df
        
