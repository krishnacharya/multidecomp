import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bilevel.manual_inv_LinearExpert import Manual_inv_LinearExpert
from bilevel.build_all_models import *
from collections import defaultdict

from scipy.linalg import lstsq
from sklearn.linear_model import LinearRegression

'''
IMPORTANT NOTE provide two A1_t's one to decide group membership, 
used for the adanormal hedge (e.g. A1_t for young, middle, old, ...,), 
(young, middle, old..., always_on) another A2_t for finding regret on arbitrary 
subsequnce, which may not even have any gurantees (A2_t for young and male, young
and female, always on), even though may not be experts
'''

class BuildGroupwise_diffseeds:
    def __init__(self, df : pd.DataFrame, target_col:str, A1_t : pd.DataFrame, A2_t: pd.DataFrame):
        '''
        df: dataframe minmax numeric, one hot categoric
        df must have all features then column for y
        target_col : regression target
        A1_t is the group membership dataframe used in the experts for adanormal hedge
        A2_t is the matrix used to mask and caclulate subsequence regret, it may not correspond to an expert, see IMPORTANT NOTE above
        '''
        self.target_col = target_col
        self.X_dat = df.drop(target_col, axis = 1)
        self.y_dat = df[target_col]
        self.A1_t = A1_t # used for the adanormal hedge group meta experts
        self.A2_t = A2_t # used only for masking then calculating subsequnece regret for arbitray subsequences
        self.T = A1_t.shape[0] # number of rows in dataframe
        self.N_metaexp = A1_t.shape[1] # number of group meta experts in Adanormal hedge # NOTE fix any mismatch of N of meta expert and number of subsequences
        self.N_subseq = A2_t.shape[1] # number of subsequences on whcih to calculae regret
        self.group_names = list(A2_t.columns) #these group names could be seperate from those we use for adanormal hedge group exeprts
        self.group_sizes = list(A2_t.sum(axis = 0).astype(int)) # same comment above
        self.rand_seeds =  [473, 503, 623, 550, 692, 989, 617, 458, 301, 205] # random seeds used to shuffle dataframe and group, to get mean & variance of cumulative loss
        self.num_runs = len(self.rand_seeds)

    def build_all_seeds(self, l2_pen : float = 1.0):
        '''
        Collects Anh and baseline performance for each of the shuffled dataframes
        also store Anh and baseline object across different random seeds for regret curves later
        '''
        def add_to_dic_res(b_ridgebase, b_Anh):
            cumloss_base = b_ridgebase.expert.cumloss_groupwise
            cumloss_groupwise_ada = b_Anh.Anh.cumloss_groupwise_ada
            for g_ind, gname in enumerate(self.group_names):
                base = cumloss_base[g_ind][-1]
                ada = cumloss_groupwise_ada[g_ind][-1]
                self.dic_res_base[gname].append(base) # stores gnmaes cumloss at end, will have 10 valeus for 10 random seeds, key value is group name
                self.dic_res_Anh[gname].append(ada)
                
        # this dictionary has cumulative loss of base, Anh across different shuffles
        self.dic_res_base = defaultdict(list)
        self.dic_res_Anh = defaultdict(list)

        self.base_obj_list = []
        self.Anh_obj_list = []
        for seed in self.rand_seeds:
            X_shuf = self.X_dat.sample(frac = 1, random_state = seed)
            y_shuf = self.y_dat.sample(frac = 1, random_state = seed)
            A1_t_shuf_np = self.A1_t.sample(frac = 1, random_state = seed).to_numpy() # used for Anh meta experts probabilityies
            A2_t_shuf_np = self.A2_t.sample(frac = 1, random_state = seed).to_numpy() #used for subsequnce regret calc
            X_dat_np = X_shuf.to_numpy()
            y_dat_np = y_shuf.to_numpy()

            dirname_base = './models_adult/baseline/'
            filename = 'manual_ridge_seed='+ str(seed)+ ' '
            b_ridgebase = build_baseline_alwayson(dirname_base, filename, A2_t_shuf_np, Manual_inv_LinearExpert(X_dat_np, y_dat_np, l2_pen = l2_pen))
            dirname_Anh = './models_adult/Anh/'
            experts = [Manual_inv_LinearExpert(X_dat_np, y_dat_np, l2_pen = l2_pen) for _ in range(self.N_metaexp)]
            b_Anh = build_Anh(dirname_Anh, filename, A1_t_shuf_np, A2_t_shuf_np, experts)
            add_to_dic_res(b_ridgebase, b_Anh)
            self.base_obj_list.append(b_ridgebase)
            self.Anh_obj_list.append(b_Anh)
    
    def build_df_res(self):
        '''
        builds the Anh and baseline results dataframe, for the ending cumulative loss
        '''
        self.df_res_base, self.df_res_Anh = pd.DataFrame(self.dic_res_base), pd.DataFrame(self.dic_res_Anh) #each row has cumulative loss of each group in a rand seed run
        self.df_base_meansd = self.df_res_base.describe().loc[['mean', 'std']].T # mean and sd are the columns
        self.df_Anh_meansd = self.df_res_Anh.describe().loc[['mean', 'std']].T
       
        self.df_base_meansd.rename(columns={'mean': 'mean_base', 'std': 'std_base'}, inplace=True)
        self.df_Anh_meansd.rename(columns={'mean': 'mean_Anh', 'std': 'std_Anh'}, inplace=True)

    def build_regret_curve(self):
        def get_Anh_regret_best_hindsight(cl_ada_g:np.array, cl_base_g: np.array, \
                                         X_dat_g:pd.DataFrame, y_dat_g:pd.DataFrame, pos_g : np.array) -> (np.array, np.array): # for a single group on single run, find regret wrt best in hind
            sse = [] # sum of squared errors till that pos p
            for p in pos_g:
                X_batch = X_dat_g[:p]
                y_batch = y_dat_g[:p]
                # Using sklearn
                # lr = LinearRegression()
                # lr.fit(X_batch, y_batch)
                # sse.append(np.sum((lr.predict(X_batch) - y_batch)**2))

                # Using scipy
                X_batch_np = X_batch.to_numpy()
                y_batch_np = y_batch.to_numpy()
                theta_ls, _, _, _ = lstsq(X_batch_np, y_batch_np)
                y_pred_ls = X_batch_np @ theta_ls # unclipped
                # y_pred_ls = np.clip(y_pred_ls, 0.0, 1.0)# clipped
                sse.append(np.sum((y_pred_ls - y_batch_np)**2))
            sse = np.array(sse)
            Base_reg_g = cl_base_g[pos_g] - sse
            Anh_reg_g = cl_ada_g[pos_g] - sse # only returning regret on num_points in Tg sequence
            return sse, Base_reg_g, Anh_reg_g
        
        self.pos = [] # linspace for each group, doesnt depend on shuffling order, its just poitns along Tg
        for Tg in self.group_sizes: # setting the positions along Tg for the regret curve 
            num_points = min(100, Tg) # TODO change this to custom integer passed in build_regret_curve
            self.pos.append(np.linspace(Tg // num_points, Tg - 1, dtype = int, num = num_points))
        self.cumloss_best_hind = [[0 for x in range(self.num_runs)] for y in range(self.N_subseq)] # to assign to num_points calculated below
        self.regret_Base_groupwise_array = [[0 for x in range(self.num_runs)] for y in range(self.N_subseq)]
        self.regret_Anh_groupwise_array = [[0 for x in range(self.num_runs)] for y in range(self.N_subseq)] # N rows, 10 columns for 10 seeds, regret to best in hindsight for Anh
        for ind in range(self.num_runs): # corresponding b_Anh has the Anh obj for that random seed
            b_Base, b_Anh = self.base_obj_list[ind], self.Anh_obj_list[ind]
            seed = self.rand_seeds[ind] # use this to get the X_dat_g, y_dat_g
            A2_t_shuf = self.A2_t.sample(frac=1, random_state = seed)
            X_dat_shuf = self.X_dat.sample(frac=1, random_state = seed)
            y_dat_shuf = self.y_dat.sample(frac=1, random_state = seed)
            for g_ind, gname in enumerate(self.group_names):
                indices_g = (A2_t_shuf[gname] == 1) # use A2 as those have the group subsequence for which we want to calculate regret
                X_dat_g = X_dat_shuf[indices_g] #only has gname==1 active rows
                y_dat_g = y_dat_shuf[indices_g]
                sse_g, Base_reg_g, Anh_reg_g = get_Anh_regret_best_hindsight(b_Anh.Anh.cumloss_groupwise_ada[g_ind], b_Base.expert.cumloss_groupwise[g_ind], \
                                                                                            X_dat_g, y_dat_g, self.pos[g_ind])
                self.cumloss_best_hind[g_ind][ind] = sse_g
                self.regret_Base_groupwise_array[g_ind][ind] = Base_reg_g
                self.regret_Anh_groupwise_array[g_ind][ind] = Anh_reg_g

def plot_subgroups(gwise_obj: BuildGroupwise_diffseeds, subgroups_list : list):
    '''
    subgroups_list [[young, middle, old], [male, female], ...] list of all the atmoic in each subgroup
    '''
    for subgroups in subgroups_list:
        df_base_sg = gwise_obj.df_base_meansd.loc[subgroups]
        df_Anh_sg = gwise_obj.df_Anh_meansd.loc[subgroups]
        group_bar_plot_df = pd.concat([df_base_sg, df_Anh_sg], axis = 1)
        yerr = group_bar_plot_df[['std_base', 'std_Anh']].to_numpy().T
        group_bar_plot_df[['mean_base', 'mean_Anh']].plot(kind='bar', yerr=yerr, alpha=0.85, error_kw=dict(ecolor='k'), capsize=3)
        plt.legend(labels = ['Baseline', 'Anh'], bbox_to_anchor=(0, 1.02, 0.4,0.2), loc ='lower left', mode='expand', ncol = 2)
        plt.ylabel('cumulative loss')
        plt.show()
    
def plot_regret_curve_with_std(gwise_obj: BuildGroupwise_diffseeds, dir_name:str):
    for g_ind, gname in enumerate(gwise_obj.group_names):
        gwise_obj.regret_Anh_groupwise_array[g_ind] = np.array(gwise_obj.regret_Anh_groupwise_array[g_ind]) # all 10 values in the row have same dim, so can make np array
        gwise_obj.regret_Base_groupwise_array[g_ind] = np.array(gwise_obj.regret_Base_groupwise_array[g_ind])
        print(gname, gwise_obj.group_sizes[g_ind])
        mean_reg_Anh, sd_reg_Anh = gwise_obj.regret_Anh_groupwise_array[g_ind].mean(axis = 0), gwise_obj.regret_Anh_groupwise_array[g_ind].std(axis = 0)
        mean_reg_Base, sd_reg_Base = gwise_obj.regret_Base_groupwise_array[g_ind].mean(axis = 0), gwise_obj.regret_Base_groupwise_array[g_ind].std(axis = 0)
        # plt.plot(self.pos[g_ind], self.regret_Anh_groupwise_array[g_ind][0], label = 'mean(ada - besthind_ls)')
        plt.plot(gwise_obj.pos[g_ind], mean_reg_Base , label = 'Baseline')
        plt.fill_between(gwise_obj.pos[g_ind], mean_reg_Base - sd_reg_Base, mean_reg_Base + sd_reg_Base, alpha = 0.5)

        plt.plot(gwise_obj.pos[g_ind], mean_reg_Anh , label = 'Our algorithm')
        plt.fill_between(gwise_obj.pos[g_ind], mean_reg_Anh - sd_reg_Anh, mean_reg_Anh + sd_reg_Anh, alpha = 0.5)
        # plt.legend(bbox_to_anchor=(0, 1.02, 0.8, 0.2), loc ='lower left', mode='expand', ncol = 2)
        plt.legend(bbox_to_anchor=(0, 1.02, 1.0, 0.2), loc ='lower left')
        plt.xlabel('time')
        plt.ylabel('Regret')
        plt.title(gname)
        plt.savefig(dir_name + '/regret_'+ gname +'.pdf')
        plt.show()

def get_end_regret_gw_df(gwise_obj: BuildGroupwise_diffseeds) -> pd.DataFrame:
    df_rows = []
    for g_ind, gname in enumerate(gwise_obj.group_names):
        gwise_obj.regret_Anh_groupwise_array[g_ind] = np.array(gwise_obj.regret_Anh_groupwise_array[g_ind]) # all 10 values in the row have same dim, so can make np array
        gwise_obj.regret_Base_groupwise_array[g_ind] = np.array(gwise_obj.regret_Base_groupwise_array[g_ind])
        gwise_obj.cumloss_best_hind[g_ind] = np.array(gwise_obj.cumloss_best_hind[g_ind])
        mean_regend_Base, std_regend_Base = gwise_obj.regret_Base_groupwise_array[g_ind].mean(axis = 0)[-1], gwise_obj.regret_Base_groupwise_array[g_ind].std(axis = 0)[-1]
        mean_regend_Anh, std_regend_Anh = gwise_obj.regret_Anh_groupwise_array[g_ind].mean(axis = 0)[-1], gwise_obj.regret_Anh_groupwise_array[g_ind].std(axis = 0)[-1]
        mean_cumloss_bhind, std_cumloss_bhind = gwise_obj.cumloss_best_hind[g_ind].mean(axis = 0)[-1], gwise_obj.cumloss_best_hind[g_ind].std(axis = 0)[-1] # last time index's cumulative loss
        df_rows.append([gname, gwise_obj.group_sizes[g_ind], mean_regend_Base, std_regend_Base, mean_regend_Anh, std_regend_Anh, mean_cumloss_bhind, std_cumloss_bhind])
    return pd.DataFrame(df_rows, columns = ['group_name', 'group_size', 'mean_regend_Base', 'std_regend_Base', 'mean_regend_Anh', 'std_regend_Anh', 'mean_hindsight', 'std_hindsight'])

