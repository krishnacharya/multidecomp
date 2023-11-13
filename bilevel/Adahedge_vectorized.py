import numpy as np
from bilevel.ExpertsAbstract import Expert

class Adanormal_sleepingexps:
  def __init__(self, A_t: np.ndarray, experts:list[Expert]):
    self.A_t = A_t # has shape T x N
    self.T = A_t.shape[0]
    self.N = A_t.shape[1] # number of meta sleeping experts
    self.experts = experts # these already have dataframes within them
    self.p_t_arr = np.zeros((self.T, self.N)) # array of numpy arrays, each numpy array is probability distribution over the meta experts at time t
    self.l_t = np.zeros((self.T, self.N)) # loss of each expert, in each round
    self.loss_ada_t_arr = np.zeros(self.T) # stores scalar loss of ada in each round
    # self.cuml_loss_adagroup_tarr = [np.zeros(self.N)] #  first term on the lhs in multigroup regret(algorithms performnace on subsequence), prev loss + (loss_ada * active or not),
    self.cumloss_groupwise_ada = [] #filled in build_cumloss_curve one for each expert
    self.cumloss_groupwise_metaexp = []

    # next 3 are important for Adanormal hedge (in the Anh paper)
    self.r_t = np.zeros(self.N) #array of numpy array, each is instantaneous regret of each expert in round t
    self.R_t = np.zeros(self.N) # cumulative regret for each expert vector is appended, R in the Adanormal hedge paper
    self.C_t = np.zeros(self.N) # absolute values of instantanoues regret summed up for each expert is appended, C in the Adanormal hedge paper

  def get_prob_over_experts(self, t):
    '''
    This will be called FIRST to get what distribution to play over active experts
    a_t[i] = 0 means group expert i is sleeping, active otherwise, its a binary array

    Returns probability over active experts, if none active returns uniform over all experts
    '''
    def w(R, C):
        '''
        vectorized weight acc to potential function \Phi in Anh paper
        '''
        dr = 3 * (C + 1)
        t1 = np.exp(np.clip(R + 1, 0.0, None)**2 / dr)
        t2 = np.exp(np.clip(R - 1, 0.0, None)**2 / dr)
        return 0.5 * (t1-t2)

    a_t = self.A_t[t]
    if np.all(a_t == 0): #if no group is active just return uniform random over experts, doesnt affect our regret, state variables etc
      self.p_t_arr[t] = np.ones(self.N) / self.N
      return self.p_t_arr[t]
    v = w(self.R_t, self.C_t) * a_t # both w(R , C) and a_t have same shape
    if np.all(v == 0): # if by chance after all this computation all weights are zero, then too predict uniformly at random over active groups? prevents nan error if np.sum(v) is zero and we divide
      self.p_t_arr[t] = a_t / np.sum(a_t)
      return self.p_t_arr[t]
    self.p_t_arr[t] = v / np.sum(v)
    return self.p_t_arr[t]

  def update_metaexps_loss(self, t):
    '''
    This is called SECOND to update the losses, regret, absolute regret etc.. as required for the next round by adanormal hedge

    Update all the active meta experts losses using the label y_t
    note this updates the losses for each Online ridge expert that is active
    a_t is numpy binary array 0 if group is inactive, contains the group indicators for round t
    t is the row of the dataframe/time step
    '''
    a_t = self.A_t[t]
    l_t_hat = 0 # loss of anh in round t
    for index, active in enumerate(a_t): # this can be parallelized/MAP operation from mapreduce, for now sequential
      if active: #if group is active (1), SIMULATE running it, i.e. get its prediction, and tell it the adversary's generate label y_t
        self.experts[index].get_ypred_t(t) #simulate getting prediction from meta expert
        self.experts[index].update_t(t)
        self.l_t[t][index] = self.experts[index].loss_tarr[-1] # loss of each expert (index) in round t
    l_t_hat = np.dot(self.p_t_arr[t], self.l_t[t]) #lthat = p_{t,i} dot l_{t,i}, scalar
    self.loss_ada_t_arr[t] = l_t_hat
    self.r_t = (l_t_hat - self.l_t[t]) * a_t # instantaneous regret, shape (N,)
    self.R_t += self.r_t #update regret cumulative sum
    self.C_t += abs(self.r_t) #update abs reg cumulative sum

  # def build_cumloss_curve(self):
  #   '''
  #     CALLED once at the end to compute regret curve for Adanormal hedge
  #     bestsqloss list of size |G| has the best square loss on the subsequence for each group, each element of bestsqloss is a list itself of length Tg
  #     Build ada normal cumulative loss on each subsequence defined by groups
  #     term1 in the multigroup regret (performance of algorithm on subsequences)
  #   '''
  #   self.cumloss_ada_allgroups = np.cumsum(self.loss_ada_t_arr.reshape(-1, 1) * self.A_t, axis = 0) # reshaped loss_ada_t_arr to shape (T, 1) so that brodcasted
  #   self.cumloss_meta_exps = np.cumsum(self.l_t, axis = 0)
  #   for ind in range(self.N):
  #     self.cumloss_groupwise_ada.append(self.cumloss_ada_allgroups[:, ind][self.A_t[:, ind].astype(bool)]) # shape Tgx1 collects the cumulative loss curve of adanormal hedge on subsequence given by group #ind, only picks roudns in which group active
  #     self.cumloss_groupwise_metaexp.append(self.cumloss_meta_exps[:, ind][self.A_t[:, ind].astype(bool)]) # not actually implementable
  
  def cleanup(self):
    '''
      CALL only after build_cumloss_curve(.,.)
      This function is just to remove internal variables used in computation that 
      dont need to be saved for external use
      Also numpify's the required variables for external use, joblib is efficient with
      numpy arrays
    '''
    self.r_t = None
    self.R_t = None
    self.C_t = None
    for gnum in range(self.N):
      self.cumloss_groupwise_ada[gnum] = np.array(self.cumloss_groupwise_ada[gnum])
      self.experts[gnum].cleanup() # makes the internal variables in the meta experts, namely loss_tarr and y_predarr into numpy arrays