import numpy as np
import pandas as pd
import random
import itertools
from bilevel.utils import numeric_scaler
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
        self.samples = kwargs['samples'] # int
        self.dim = kwargs['dim'] # int
        self.group_dict = kwargs['group_dict'] # dict
        self.prob_dict = kwargs['prob_dict'] # dict
        list2d = [li for li in  self.group_dict.values()] 
        self.all_groupnames = list(itertools.chain(*list2d)) # list of all group names # TODO add always active or not
        self.Ng = len(self.all_groupnames) # int number of groups
        self.feat_lo = kwargs['feat_lo'] # float X_lo for uniform
        self.feat_hi = kwargs['feat_hi'] # float X_hi for uniform
        self.add_linear_mapping = kwargs['add_linear_mapping'] #boolean
        self.w_lo = kwargs['w_lo'] # float w_lo for uniform weight, used in matmul(X, w)
        self.w_hi = kwargs['w_hi'] # float w_hi
        self.add_quad_mapping = kwargs['add_quad_mapping'] #boolean
        self.S_lo = kwargs['S_lo'] # float, used in x^T S x, quadratic
        self.S_hi = kwargs['S_hi'] # float
        self.label_noise_width = kwargs['label_noise_width']
        self.drop_sensitive = kwargs['drop_sensitive']
        self.fixed_seed = kwargs['fixed_seed'] # for reproducibility
        np.random.seed(self.fixed_seed) # global seed for all numpy randomness
        random.seed(self.fixed_seed) # global seed for all python randomness

        self.get_feat_uniform()
        self.get_A_t()
        # self.get_weights()
        self.get_labels()
        self.get_dataframe()
        self.aggregate_group_labels()
    
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
        # TODO add always active or not
        def get_group_indicators(prob_list: list) -> list[np.ndarray]:
            inds = np.eye(len(prob_list)) # indicators e.g for prob list of len 3, (1, 0, 0), (0, 1, 0), (0, 0, 1)
            return np.array(random.choices(population=inds, weights=prob_list, k = self.samples))
        self.A_t = np.hstack([get_group_indicators(prob_list) for prob_list in self.prob_dict.values()])
        return self.A_t
    
    # def get_weights(self) -> np.ndarray:
    #     self.weights = np.random.uniform(low = self.w_lo, high = self.w_hi, size = (self.dim, self.Ng)) # shape is (dim, # of groups)
    #     return self.weights
    
    def get_labels(self) -> np.ndarray:
        '''
        NEEDS get features to be called before this is called
        Returns
            labels for each group (#samples, #number of groups)
        '''
        def add_linear():
            self.wlin = np.random.uniform(low = self.w_lo, high = self.w_hi, size = (self.dim, self.Ng))
            self.labels_allg += np.matmul(self.feat_dat, self.wlin) # rhs has shape (# samples, # group)
        
        def add_quad():
            self.Smat = np.random.uniform(low = self.S_lo, high = self.S_hi, size = (self.Ng, self.dim, self.dim))
            for g in range(self.Ng):
                self.labels_allg[:, g] += (self.feat_dat.dot(self.Smat[g])*self.feat_dat).sum(axis=1) # rhs has shape (number of samples,)  # https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v              
        
        self.labels_allg = np.zeros((self.samples, self.Ng))
        if self.add_linear_mapping:
            add_linear()
        if self.add_quad_mapping:
            add_quad()
        self.labels_allg += np.random.normal(scale = self.label_noise_width, size = (self.samples, self.Ng)) # adding gaussian noise
        return self.labels_allg

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

    def aggregate_group_labels(self) -> None:
        '''
            aggregates all the active group labels in 4 different ways
            
            mean : mean of active group labels
            min : min of active group labels
            max : max of active group labels
            dperm : first active group in dominance permutation
        '''
        def set_dominance_permutation():
            self.dperm = np.random.permutation(self.Ng)
        # TODO scaled back each y in 0-1
        self.masked_mult = np.ma.masked_array(self.A_t, mask = self.A_t == 0) *  self.labels_allg
        self.mean_ar = np.ma.getdata(np.mean(self.masked_mult, axis = 1))
        self.min_ar = np.ma.getdata(np.min(self.masked_mult, axis = 1))
        self.max_ar = np.ma.getdata(np.max(self.masked_mult, axis = 1))
        self.df['y_mean_active'] = self.mean_ar
        self.df['y_min_active'] = self.min_ar
        self.df['y_max_active'] = self.max_ar
        set_dominance_permutation()
        self.mm_dperm = self.masked_mult[:, self.dperm] #masked multiplication permuted columns
        first_nomask_index = (np.ma.getmask(self.mm_dperm) == False).argmax(axis=1) #get first non masked element location, this is the label
        self.df['y_dperm_active'] = self.mm_dperm[np.arange(self.samples), first_nomask_index]

        # the code here is just to scale each y in 0,1 minmax scale
        self.df_label_names = [col for col in self.df.columns if 'y_' in col]
        self.df = numeric_scaler(self.df, self.df_label_names)

        #ilocs below are slow, just work with fast nummpy element wise multiplication above
        
        # binary_masked = (self.df[self.df_label_names] * self.A_t) # y_t part of dataframe multiplied by A_t mask
        # active_indices = binary_masked.apply(np.flatnonzero, axis=1) # get the non zero value positions in the above
        # self.df['active_labels'] = None
        # self.df['bin_masked_labels'] = None
        # for i, ai in enumerate(active_indices):
        #     self.df.at[i, 'active_labels'] = self.df[self.df_label_names].iloc[i, ai].to_numpy()
        #     self.df.at[i, 'bin_masked_labels'] = (self.df[self.df_label_names].iloc[i] * self.A_t[i]).to_numpy()
        # return self.df