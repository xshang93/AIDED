#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:24:30 2024

@author: xiao
"""
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from others.helper_func import meltpool_geom_cal,pred_2d_surface,pred_3d_cube
import matplotlib.pyplot as plt
import math
import os

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.core.callback import Callback
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.indicators.hv import Hypervolume


## ---------------------- Load the trained ML models ------------------------ #
model_path = '/home/xiao/git_projects/auto_AM_researcher/Xiao/MeltPool_MLWorkFlow/trained_models/'
para2geom = load_model(model_path+'para2geom.h5')
para2geom_pca = joblib.load(model_path+'pca_transformer.pkl')
hs2angle = joblib.load(model_path+'hs2angle.pkl')

# randomseed to initialize GA
random_seed = 0
# name of the current run
run_name = '20240703_rand_{0}_test_run'.format(random_seed)
save_dir = '/home/xiao/projects/DED/BO_processing/Final_data/surfaces/GA/NSGA/{0}/'.format(run_name)
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

## -------------------- Defined the objective functons ---------------------- #
# size of each population
pop_size = 1024
# initialize gen to 0
gen = 0
counter = 0
counter_save_frequency = int(pop_size/4) # save every counter_save_frequency evaluations
# power in % of 1kw, speed in mm/min
# Solution=[power,speed,rpm,hs]
def resolution(solution):
    global counter
    counter = counter+1
    global pop_size
    if counter%counter_save_frequency==0:
        plot=True
    else:
        plot=False
    global gen
    gen = int(counter/pop_size)
    # error handling
    try:
        width,_,_,_,scale,_,_ = meltpool_geom_cal(solution[0],solution[1],solution[2],plot=plot,save_path=save_dir+'Gen_{0}_{1}_single_track'.format(gen,counter))
    except:
        return 5 # a large value to make sure this is not selected
    
    return width*scale*solution[3]

def print_time(solution):
    # set target cube dimensions in mm
    cube_h = 5
    cube_w = 10
    cube_l =10
    if counter%counter_save_frequency==0:
        save_path=save_dir+'Gen_{0}_{1}_multi_track'.format(gen,counter)
    else:
        save_path = ''
    try:
        width,height,depth,_,scale,_,_ = meltpool_geom_cal(solution[0],solution[1],solution[2],plot=False,save_path='')
    except:
        return 500 # a large value to make sure this is not selected
    angle = hs2angle.predict([[width,solution[1],solution[3],height]])
    
    # calculate the number of tracks in each layer
    num_tracks = round((cube_w-width*scale*np.cos(math.radians(angle)))/(width*scale*solution[3])+1)
    # calculate the number of layers
    # _,layer_height,_,_,_ = pred_2d_surface(solution[0],solution[1],solution[2],solution[3],num_tracks,opt=True,save_path=save_path)
    try:
        _,layer_t,_ = pred_3d_cube(solution[0],solution[1],solution[2],solution[3],num_tracks,3,opt=True,verbose=False,save_path=save_path)
    except:
        return 500
    num_layers = round(cube_h/(layer_t*scale))
    
    total_t = cube_l/(solution[1]/60)*num_tracks*num_layers
    
    return total_t

# def dilution(solution):
#     try:
#         _,_,_,_,_,d = meltpool_geom_cal(solution[0],solution[1],solution[2],plot=False,save_path='')
#     except:
#         return 0
    
#     return d

def dilution_surface(solution):
    try:
        _,_,_,_,ds = pred_2d_surface(solution[0],solution[1],solution[2],solution[3],1,opt=True,save_path='')
    except:
        return 0
    
    return ds    

## -------------------- Set up the optimization ---------------------- #
objs = [
    lambda x: resolution(x),
    lambda x: print_time(x)
]

constr_ieq = [
    # lambda x: dilution(x)-0.5,
    # lambda x: 0.2-dilution(x),
    lambda x: dilution_surface(x)-0.5,
    lambda x: 0.2-dilution_surface(x)
    
]

n_var = 4

problem = FunctionalProblem(n_var,
                            objs,
                            constr_ieq=constr_ieq,
                            xl=[30,240,0.2,0.5],
                            xu=[60,720,0.8,0.7],
                            )

# ----------------------------- For running the optimization ------------------------- #
# NSGA iii
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=6)
crossover = SinglePointCrossover()
algorithm = NSGA3(pop_size=pop_size,
                  ref_dirs=ref_dirs,
                  crossover=crossover,
                  )

termination = DefaultMultiObjectiveTermination(
    n_max_gen=200,
    period=10
)

res = minimize(problem,
                algorithm,
                termination,
                seed=random_seed,
                # callback=MyCallback(),
                # output=MyOutput(),
                verbose=True,
                save_history=True)

X, F = res.opt.get("X", "F")
hist = res.history

n_evals = []             # corresponding number of function evaluations\
hist_F = []              # the objective space values in each generation
hist_cv = []             # constraint violation in each generation
hist_cv_avg = []         # average constraint violation in the whole population

for algo in hist:

    # store the number of function evaluations
    n_evals.append(algo.evaluator.n_eval)

    # retrieve the optimum from the algorithm
    opt = algo.opt

    # store the least contraint violation and the average in each population
    hist_cv.append(opt.get("CV").min())
    hist_cv_avg.append(algo.pop.get("CV").mean())

    # filter out only the feasible and append and objective space values
    feas = np.where(opt.get("feasible"))[0]
    hist_F.append(opt.get("F")[feas])

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)


metric = Hypervolume(ref_point= np.array([0.5,20000]),
                      norm_ref_point=False,
                      zero_to_one=True,
                      ideal=approx_ideal,
                      nadir=approx_nadir)

hv = [metric.do(_F) for _F in hist_F]

plt.figure(figsize=(7, 5))
plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
plt.xlim([-1000,30000])
plt.ylim([3000,11000])
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.savefig(save_dir+'hv_plot.svg')
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(res.F[:,0],res.F[:,1])
plt.xlim([0.25,1])
plt.ylim([0,500])
plt.savefig(save_dir+'pareto_plot.svg')



np.save(save_dir+'results_x',res.X)
np.save(save_dir+'results_y',res.F)
np.save(save_dir+'n_evals',n_evals)
np.save(save_dir+'hv',hv)
hist_F_save = np.asarray(hist_F, dtype="object")
np.save(save_dir+'hist_F',hist_F_save)