import numpy as np
import math
from copy import deepcopy
import gym
import gym.spaces as spaces
from gym.utils import seeding
import scipy
import portpy.photon as pp

# path of data
# data_path = r'D:\MyJupyter\RL_BAO\Data'
patient_id_list = ['Lung_Phantom_Patient_1', 'Lung_Patient_2', 'Lung_Patient_3',
                   'Lung_Patient_4', 'Lung_Patient_5', 'Lung_Patient_6',
                   'Lung_Patient_7', 'Lung_Patient_8', 'Lung_Patient_9',
                   'Lung_Patient_10']



class PortPyEnv(gym.Env):
  metadata = {
      'render.modes': ['human'],
  }

  def __init__(self, data_path, action_num = 72, dvh_dim = 128,
               step_max_num = 30, patient_id = 6):
    self.data_path = data_path
    self.step_max_num = step_max_num
    self.step_count = 0
    self.patient_id = patient_id
    self.theta_dim = action_num
    self.dvh_dim = dvh_dim
    self.action_space = spaces.Discrete(self.theta_dim)
    self.np_random = None

    # data load
    self.data = pp.DataExplorer(data_dir=data_path)
    self.data.patient_id = patient_id_list[self.patient_id - 1]
    beams_df, structs_df = self.data.display_patient_metadata(return_beams_df=True, return_structs_df=True)
    struct_names = list(structs_df["name"])
    self.struct_names = struct_names[1:] # except "GTV"

    self.theta_state = np.zeros(self.theta_dim)
    self.dvh_state = np.ones(len(self.struct_names) * self.dvh_dim)
    self.state = np.concatenate((self.theta_state, self.dvh_state), axis=0)

    self.ct = pp.CT(self.data)
    self.structs = pp.Structures(self.data)
    protocol_name = 'Lung_2Gy_30Fx'
    self.clinical_criteria = pp.ClinicalCriteria(self.data, protocol_name=protocol_name)
    # Loading hyper-parameter values for optimization problem
    self.opt_params = self.data.load_config_opt_params(protocol_name=protocol_name)
    # Creating optimization structures (i.e., Rinds) 
    self.structs.create_opt_structures(opt_params=self.opt_params, clinical_criteria=self.clinical_criteria)

    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    state = self.state
    theta_state = state[:self.theta_dim]

    next_theta_state = theta_state.copy()
    next_theta_state[action] = 1
    next_dvh_state = self.get_dvh_state(next_theta_state)
    next_state = np.append(next_theta_state, next_dvh_state)

    reward = self.Reward(state, next_state)

    done = self.is_done()

    self.state = next_state
    self.step_count += 1

    return next_state, reward, done, {}

  def is_done(self):
    theta_state = self.state[:self.theta_dim]

    if theta_state.all() or self.step_count >= self.step_max_num:
      return True
    else:
      return False

  def reset(self):
    self.theta_state = np.zeros(self.theta_dim)
    self.dvh_state = np.ones(len(self.struct_names) * self.dvh_dim)
    self.state = np.concatenate((self.theta_state, self.dvh_state), axis=0)

    return self.state

  def render(self, mode='human'):
    pass

  def get_dvh_state(self, theta):
    # start solve FMO
    beam_ids = np.where(theta == 1)[0]
    beams = pp.Beams(self.data, beam_ids=beam_ids)
    # Loading influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=self.ct, structs=self.structs, beams=beams)
    my_plan = pp.Plan(ct = self.ct, structs = self.structs, beams = beams, inf_matrix = inf_matrix,
                      clinical_criteria=self.clinical_criteria)

    opt = pp.Optimization(my_plan, opt_params=self.opt_params, clinical_criteria=self.clinical_criteria)
    opt.create_cvxpy_problem()
    # solve the cvxpy problem using Mosek5
    # need the license of MOSEK
    sol = opt.solve(solver='MOSEK', verbose=False)
    for p, s in enumerate([sol]):
        dose_1d = s['inf_matrix'].A @ (s['optimal_intensity'] * my_plan.get_num_of_fractions())

    dvh_x, dvh_y = self.dvh_processed(sol, dose_1d, struct_names_id = 0)
    dvh_state = dvh_y
    for i in range(1, len(self.struct_names)):
        dvh_x, dvh_y = self.dvh_processed(sol, dose_1d, struct_names_id = i)
        dvh_state = np.append(dvh_state, dvh_y)

    return dvh_state

  def dvh_processed(self, sol, dose_1d, struct_names_id):
    inf_matrix = sol['inf_matrix']
    # vox: volex id of struct_names[i] (corresponding to dose_1d)
    vox = inf_matrix.get_opt_voxels_idx(self.struct_names[struct_names_id])
    org_sort_dose = np.sort(dose_1d[vox])
    x = org_sort_dose
    # sort_ind: Obtain the index of doses corresponding to vox (voxel id) in ascending order
    sort_ind = np.argsort(dose_1d[vox])
    # Return the list of voxel volumes (cc: cm ^ 3) for struct_names[i]
    org_weights = inf_matrix.get_opt_voxels_volume_cc(self.struct_names[struct_names_id])
    # org_sort_weights: Get the list of voxel volume sizes corresponding to x = org_sort_dose
    org_sort_weights = org_weights[sort_ind]
    frac_vol = inf_matrix.get_fraction_of_vol_in_calc_box(self.struct_names[struct_names_id])

    # the all volume weight
    sum_weight_all = np.sum(org_sort_weights) / frac_vol
    weight_not_in_calc_box = sum_weight_all - np.sum(org_sort_weights)
    x_new  = np.insert(x, 0, 0)
    org_sort_weights_new = np.insert(org_sort_weights, 0, weight_not_in_calc_box)
    unique_x, id_x = np.unique(x_new, return_inverse = True)
    unique_weights = np.bincount(id_x, weights = org_sort_weights_new)

    y = [1]
    for j in range(len(unique_weights)):
        y.append(y[-1] - unique_weights[j] / sum_weight_all)
    
    # Unified interpolation interval
    # the prescription_gy of examples in portpy is 60, so the upper Bound of 
    # dose interpolation interval is 72
    prescription_gy = self.opt_params['prescription_gy']
    max_dose_dvh = prescription_gy * 1.2

    # problem: different beam angles lead to different max_dose_dvh, which
    # is diffcult to unified interpolation interval
    # max_dose_list = []
    # for i in range(len(self.struct_names)):
    #     m = pp.Evaluation.get_max_dose(sol = sol, struct = self.struct_names[i],
    #                                    dose_1d = dose_1d)
    #     max_dose_list.append(m)
    # max_dose_dvh = math.ceil(max(max_dose_list))
    
    unique_x = np.append(unique_x, max_dose_dvh)
    # Avoiding precision errors caused by frac_vol division
    y[-1] = 0
    
    f = scipy.interpolate.interp1d(unique_x, y)

    x_plot = np.linspace(unique_x[0], unique_x[-1], self.dvh_dim)
    y_plot = f(x_plot)

    return x_plot, y_plot

  def Reward(self, state, new_state):
    theta = state[:self.theta_dim]
    new_theta = new_state[:self.theta_dim]
    reward = np.sum(np.where((new_theta - theta) != 0)[0])

    return reward

  