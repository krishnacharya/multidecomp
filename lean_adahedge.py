import numpy as np

def relu_ew(M): #must be numpy array, does max(0,.) elementwise
    return M * (M >=0)

class Adanormal_sleepingexps:
  def __init__(self, N, experts):
    self.time = 0 # time slots till now
    self.N = N # number of sleeping experts
    self.experts = experts # River ridge regression experts set using Linear Regression with l2 penalization = 1.0
    # self.active_tarr = [] #array of numpy arrays, each numpy array is a N entry binary whether group is active (1) or not (0)
    self.proboverexps_tarr = [] # array of numpy arrays, each numpy array is probability distribution over the meta experts at time t
    # self.predvec_tarr = [] #array of numpy arrays, each has prediction of each meta expert, if inactive in round t the prediction is -1 placeholder
    self.loss_vec_tarr = [] #array of numpy arrays, each has loss of each meta expert
    self.loss_ada_tarr = [] # array of scalars, each is loss of adanormal hedge in round t
    self.cuml_loss_adagroup_tarr = [np.zeros(N)] #  first term on the lhs in multigroup regret(algorithms performnace on subsequence), prev loss + (loss_ada * active or not),
    self.cuml_loss_curve = [] #filled in build_cumloss_curve one for each expert
    self.cuml_regret_curve = []

    # next 3 are important for Adanormal hedge
    self.inst_reg_tarr = [] #array of numpy array, each is instantaneous regret of each expert in round t
    self.cuml_reg_tarr = [np.zeros(N)] # cumulative regret for each expert vector is appended, R in the Adanormal hedge paper
    self.abs_reg_tarr = [np.zeros(N)] # absolute values of instantanoues regret summed up for each expert is appended, C in the Adanormal hedge paper

  def get_prob_over_experts(self, A_t):
    '''
    This will be called FIRST to get what distribution to play over active experts
    A_t[i] = 0 means group expert i is sleeping, active otherwise, its a binary array

    Returns probability over active experts, if none active returns uniform over all experts
    '''
    # self.active_tarr.append(A_t) #appending active groups in round t
    if np.all(A_t == 0): #if no group is active just return uniform random over experts, doesnt affect our regret, state variables etc
      self.proboverexps_tarr.append(np.ones(self.N) / self.N)
      return self.proboverexps_tarr[-1]
    dr = 3 * (self.abs_reg_tarr[-1] + 1.0) # 3(C+1) in paper
    t1 = np.exp(relu_ew(self.cuml_reg_tarr[-1] + 1)**2 / dr) # exp(relu(R+1)^2 / 3C)
    t2 = np.exp(relu_ew(self.cuml_reg_tarr[-1] - 1)**2 / dr)
    v = 0.5 * (t1 - t2) * A_t #zeroing out inactive experts using A_t
    if np.all(v == 0): #if by chance after all this computation all weights are zero, then too predict uniformly at random over active groups? prevents nan error if np.sum(v) is zero and we divide
      self.proboverexps_tarr.append(A_t / np.sum(A_t))
      return self.proboverexps_tarr[-1]
    self.proboverexps_tarr.append(v / np.sum(v))
    return self.proboverexps_tarr[-1]

  def update_metaexps_loss(self, A_t, x_t, y_t):
    '''
    This is called SECOND to update the losses, regret, absolute regret etc.. as required for the next round by adanormal hedge

    Update all the active meta experts losses using the label y_t
    note this updates the losses for each Online ridge expert that is active
    A_t is numpy binary array 0 if group is inactive
    x_t is the d dimensional feature/context (actually pandas row here for consistency with River)
    y_t is the label revealed by environment in round t used to update each Online ridge experts loss
    '''
    # self.predvec_tarr.append(-1 * np.ones(self.N)) # prediction by each meta expert, for inactive experts prediction remains -1
    self.loss_vec_tarr.append(np.zeros(self.N)) #loss for each metaexpert at time t
    for index, active in enumerate(A_t): # this can be parallelized/MAP operation from mapreduce, for now sequential
      if active: #if group is active (1), SIMULATE running it, i.e. get its prediction, and tell it the adversary's generate label y_t
        #  self.predvec_tarr[-1][index] = self.experts[index].get_ypred_t(x_t) #ORidge expert predictions. TODO
        self.experts[index].get_ypred_t(x_t) #simulate getting prediction from meta expert
        self.experts[index].update_t(x_t, y_t) #update River expert TODO, changed
        self.loss_vec_tarr[-1][index] = self.experts[index].loss_tarr[-1] # TODO #(yhatclipped in [0,1] - y_t)**2 for the Online Ridge expert
    self.loss_ada_tarr.append(np.dot(self.proboverexps_tarr[-1], self.loss_vec_tarr[-1])) #lthat = p_{t,i} dot l_{t,i}, scalar
    self.cuml_loss_adagroup_tarr.append(self.cuml_loss_adagroup_tarr[-1] + (self.loss_ada_tarr[-1] * A_t)) #groupwise cumulative loss
    self.inst_reg_tarr.append((self.loss_ada_tarr[-1] - self.loss_vec_tarr[-1]) * A_t) # (lthat - loss of each expert) * whether expert active or not, see BL19 paper Sec 4
    self.cuml_reg_tarr.append(self.cuml_reg_tarr[-1] + self.inst_reg_tarr[-1]) #update regret cumulative sum
    self.abs_reg_tarr.append(self.abs_reg_tarr[-1] + abs(self.inst_reg_tarr[-1])) #update abs reg cumulative sum
    self.time += 1

  def build_cumloss_curve(self, bestsqloss_arr, A_tarr):
    '''
      CALLED once at the end to compute regret curve for Adanormal hedge
      bestsqloss list of size |G| has the best square loss on the subsequence for each group, each element of bestsqloss is a list itself of length Tg
      A_tarr has shape T x G, each row is a binary whether or note groups are active
      Build ada normal cumulative loss on each subsequence defined by groups
      term1 in the multigroup regret (performance of algorithm on subsequences)
    '''
    cl_adagroup = np.array(self.cuml_loss_adagroup_tarr)[1:]
    # all_active = np.array(self.active_tarr) #numpy array of active_tarr
    for ind in range(self.N): #ind is group number 0...N-1
      self.cuml_loss_curve.append(cl_adagroup[:, ind][A_tarr[:, ind].astype(bool)]) # shape Tgx1 collects the cumulative loss curve of adanormal hedge on subsequence given by group #ind, only picks roudns in which group active
      self.cuml_regret_curve.append(self.cuml_loss_curve[-1] - np.array(bestsqloss_arr[ind])) #bestsqloss has same dimensions as self.cuml_loss_curve[-1] the current group
      # TODO ^ 
  
  def cleanup_for_saving(self):
    '''
      CALL only after build_cumloss_curve(.,.)
      This function is just to remove internal variables used in computation that 
      dont need to be saved for external use
      Also numpify's the required variables for external use, joblib is efficient with
      numpy arrays
    '''
    self.proboverexps_tarr = None
    self.loss_vec_tarr = None
    self.loss_ada_tarr = None
    self.cuml_loss_adagroup_tarr = None
    self.inst_reg_tarr = None
    self.cuml_reg_tarr = None
    self.abs_reg_tarr = None
    for gnum in range(self.N):
      self.cuml_loss_curve[gnum] = np.array(self.cuml_loss_curve[gnum])
      self.cuml_regret_curve[gnum] = np.array(self.cuml_regret_curve[gnum])
      self.experts[gnum].make_all_numpyarr() # makes the internal variables in the meta experts, namely loss_tarr and y_predarr into numpy arrays