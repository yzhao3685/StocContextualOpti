import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from gurobipy import *
from sklearn.linear_model import LassoLarsCV
from scipy.stats import norm
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import linprog
from warnings import simplefilter
from matplotlib import pyplot
import multiprocessing as mp
import time
import pdb
simplefilter(action='ignore')# ignore all warnings

deg =1; erm_deg=1


def ERM_policy(d_arr):
    n=len(d_arr)
    H = np.zeros((n_products + 1, n_products + 1))
    b = np.zeros(n_products + 1)
    for i in range(n): # len(d_arr) = train_size, len(d_arr[0]) = n_products
        temp = np.append(-1, d_arr[i])
        H += alpha * np.outer(temp, temp)   
        b[1:] += d_arr[i]
    cons_lhs = np.append(0, np.ones(n_products))
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    w = m.addMVar (n_products + 1, ub=np.append(100, np.ones(n_products)), lb=0.0, name="w")
    m.setObjective(w @ H @ w - b @ w, GRB.MINIMIZE)
    m.addConstr(cons_lhs @ w == 1)
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()    
    w_0 = w[0].x
    w_star=np.zeros(n_products)
    for i in range(n_products):
        w_star[i]=w[i+1].x
    return w_star, w_0


def Integrated_decision(d_arr, gaus_std):
    # only has one parameter theta to predict
    # use grid search for theta. note the objective function here is non-convex, non-concave degree 4 polynomial
##    n=len(d_arr)
##    possible_mean_arr = np.arange(2.1,4.1,0.01) 
##    best_obj = np.inf
##    best_theta = possible_mean_arr[0]
##    pred_mean = np.arange(1, 1 + n_products) * best_theta
##    best_w_IO = np.ones(n_products) / n_products # dummy values 
##    best_w_0 = best_w_IO @ pred_mean 
##    for i in range(len(possible_mean_arr)):
##        candidate_theta = possible_mean_arr[i]
##        candidate_mean, _ = get_model_details(candidate_theta) 
##        w_arr, w_0 = optimal_policy_gaussian(candidate_mean, gaus_std)
##        obj_val = Integrate_evaluate_objective(w_arr, w_0, d_arr)
##        if obj_val <= best_obj:
##            best_obj = obj_val
##            best_theta = candidate_theta
##            best_w_IO = w_arr
##            best_w_0 = w_0

    n=len(d_arr)
    gaus_mean, _ = get_model_details(3)
    candidate_mean_arr_1 = np.arange(-2.0,2.1,0.05) + gaus_mean[0]
    candidate_mean_arr_2 = np.arange(-2.0,2.1,0.05) + gaus_mean[1]
    best_obj = np.inf
    best_prediction = np.zeros(2)
    best_w_IO = np.ones(n_products) / n_products # dummy values 
    best_w_0 = 0.0
    H, b = Integrate_helper(d_arr)
    for i in range(len(candidate_mean_arr_1)):
        for j in range(len(candidate_mean_arr_1)):
            candidate_mean = np.array([candidate_mean_arr_1[i], candidate_mean_arr_2[j]])
            w_arr, w_0 = optimal_policy_gaussian(candidate_mean, gaus_std)
            obj_val = Integrate_evaluate_objective(w_arr, w_0, H, b)
            if obj_val <= best_obj:
                best_obj = obj_val
                best_prediction = candidate_mean
                best_w_IO = w_arr
                best_w_0 = w_0
    return best_w_IO, best_w_0

def Integrate_helper(d_arr):
    # input observations d_arr; output matrix H and vector b used in IEO obj_val calculation
    H = np.zeros((n_products + 1, n_products + 1))
    b = np.zeros(n_products + 1)
    for i in range(len(d_arr)): # len(d_arr) = train_size, len(d_arr[0]) = n_products
        temp = np.append(-1, d_arr[i])
        H += alpha * np.outer(temp, temp)   
        b[1:] += d_arr[i]
    return H, b    

def Integrate_evaluate_objective(w_arr, w_0, H, b):
    w = np.append(w_0, w_arr)
    obj_val = w @ H @ w - b @ w
    return obj_val 

def optimal_policy_gaussian(mean_arr,std_arr):
    # sanity check: solver output should match closed form solution. yes, they match
##    H = alpha * np.diag(np.power(std_arr, 2))
##    cons_lhs = np.ones(n_products)
##    m = Model()
##    m.Params.LogToConsole = 0#suppress Gurobipy printing
##    w = m.addMVar (n_products, ub=1.0, lb=0.0, name="w")
##    m.setObjective(w @ H @ w - mean_arr @ w, GRB.MINIMIZE)
##    m.addConstr(cons_lhs @ w == 1)
##    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
##    #m.params.NonConvex=2
##    m.update()
##    m.optimize()    
##    w_star_solver = np.zeros(n_products)
##    for i in range(n_products):
##        w_star_solver[i]=w[i].x
##    w_0 = w_star_solver @ mean_arr
    
    w_1 = mean_arr[0] - mean_arr[1] + 2 * alpha * std_arr[1] ** 2
    w_1 = w_1 / (2 * alpha * sum(np.power(std_arr, 2)))
    w_star = np.array([w_1, 1 - w_1])
    w_0 = w_star @ mean_arr
    return w_star, w_0

def loss_f(w, w_0, d_arr):
    # closed form formula
    mean_arr, _ = get_model_details(3)
    loss_formula = alpha * np.power(w, 2) @ np.power(true_gaus_std, 2) - mean_arr @ w
    # simulation
##    loss_arr = np.zeros(len(d_arr))
##    for i in range(len(d_arr)):
##        loss_arr[i] = alpha * (w @ d_arr[i] - w_0) ** 2 - w @ d_arr[i]
##    loss_simulated = np.average(loss_arr)
    # simulated loss is inaccurate
##    print('simulated loss ', loss_simulated, 'loss formula ', loss_formula)
    return loss_formula
        
def PTO_decisions(d_train, mle_std):
    n=len(d_train)
    # only have one parameter theta to predict
##    _, b = get_model_details(3)
##    temp1 = sum((d_train - b) @ np.divide(np.arange(1,1+n_products), mle_std))
##    temp2 = np.divide(np.power(np.arange(1,1+n_products),2), mle_std)
##    mle_mean = temp1/(n*sum(temp2))
##    mle_mean, _ = get_model_details(mle_mean)
    
##    # sanity check: when n_products = 2, the closed form solution should match the mle solution. yes, they match
##    print('mle mean: ',mle_mean)
##    temp3 = sum(d_train[:,0]) / 6 + sum(d_train[:,1]) * 2 / 3
##    closed_form_theta = temp3 / 1.5 / n
##    print('closed form mean: ', closed_form_theta)

    mle_mean = np.average(d_train, axis=0)
    w_PTO, w_0 = optimal_policy_gaussian(mle_mean,mle_std)
    return w_PTO, w_0

def get_model_details(theta):
    # input the parameter theta that we want to predict, output (1) expected return of assets (2) the constant b in the model
    b = 9
    return np.arange(1, 1 + n_products) * theta + b, b

def generate_data(train_size,test_size):
    mean_arr, _ = get_model_details(3)
    std_arr = true_gaus_std
    d_train=np.random.normal(np.tile(mean_arr,(train_size,1)),np.tile(std_arr,(train_size,1)))
    d_train = np.maximum(d_train,0) 
    d_test=np.random.normal(np.tile(mean_arr,(test_size,1)),np.tile(std_arr,(test_size,1)))
    d_test = np.maximum(d_test,0)
    w_oracle, w_0 = optimal_policy_gaussian(mean_arr,std_arr)
    #w_oracle is the best in expectation. it could happen that on finite sample, some sub-optimal decision actually does better because of noise in d_test
    return  d_train,d_test,w_oracle, w_0

def get_regret(train_size, test_size):
    d_train,d_test,w_oracle, w_0 = generate_data(train_size,test_size)
    oracle_loss = loss_f(w_oracle, w_0, d_test)

    w_ERM, w_0 = ERM_policy(d_train)
    ERM_regret=loss_f(w_ERM, w_0, d_test)-oracle_loss

    w_IO, w_0 = Integrated_decision(d_train,gaus_std)
    integrated_regret = loss_f(w_IO, w_0, d_test)-oracle_loss

    w_PTO, w_0 = PTO_decisions(d_train,gaus_std)
    PTO_regret=loss_f(w_PTO, w_0, d_test)-oracle_loss
    #print('n: ',train_size,'erm ',ERM_regret,'IO ',integrated_regret,'PF ',PTO_regret)
    return ERM_regret,PTO_regret,integrated_regret
    
def get_regret_arr(n_list, n_seed=5):
    #return array of regrets
    t1=time.time()
    erm_arr = np.zeros((len(n_list),n_seed))
    pto_arr = np.zeros((len(n_list),n_seed))
    integ_arr = np.zeros((len(n_list),n_seed))
    for i in range(len(n_list)):
        n=n_list[i]
        for j in range(n_seed):
            erm_regret, pto_regret,integ_regret = get_regret(n,10000)
            if j%5==0:
                print('seed: ',j)
            erm_arr[i,j]+=erm_regret
            pto_arr[i,j] +=pto_regret
            integ_arr[i,j]+=integ_regret
        print('n: ',n,'EO ',np.average(erm_arr[i,:]),'IO ',np.average(integ_arr[i:,]),'PF ',np.average(pto_arr[i,:]))
        print('perc: ', 'EO ', np.average(erm_arr[i] * n > threshold),
                        'PF ', np.average(pto_arr[i] * n > threshold),
                        'IO ', np.average(integ_arr[i] * n > threshold))
    time_elapsed=time.time()-t1
    print('total runtime: ',round(time_elapsed))
    return erm_arr,pto_arr,integ_arr  

def plot_error_bar(n_list,erm_arr,pto_arr,integ_arr):
    pyplot.clf()
    xi = [1,2,3,4,5]# create an index for each tick position
    xi1 = [0.7,1.7,2.7,3.7,4.7]#positions of error bars
    xi2 = [0.85,1.85,2.85,3.85,4.85]
    xi3 = [1.0,2.0,3.0,4.0,5.0]
    #xi4 = [1.15,2.15,3.15,4.15]; xi5 = [1.3,2.3,3.3,4.3]
    erm_median = np.quantile(erm_arr, 0.5,axis=1)
    pto_median = np.quantile(pto_arr, 0.5,axis=1)
    integ_median = np.quantile(integ_arr, 0.5,axis=1)
    erm_lower = erm_median  - np.quantile(erm_arr, 0.25,axis=1)
    erm_upper = np.quantile(erm_arr, 0.75,axis=1) - erm_median
    pto_lower = pto_median - np.quantile(pto_arr, 0.25,axis=1)
    pto_upper = np.quantile(pto_arr, 0.75,axis=1) - pto_median
    integ_lower = integ_median - np.quantile(integ_arr, 0.25,axis=1)
    integ_upper = np.quantile(integ_arr, 0.75,axis=1) - integ_median
    title=specification   
    pyplot.title(title, fontsize=label_font_size)
    pyplot.errorbar(xi1, erm_median, np.vstack([erm_lower, erm_upper]), linestyle='None', marker='o',color='r',capsize=4,
                elinewidth=3,label='EO') #marker='_' is a horizontal line 
    pyplot.errorbar(xi2, pto_median, np.vstack([pto_lower, pto_upper]), linestyle='None', marker='o',color='g',capsize=4,
                elinewidth=3,label='ETO')
    pyplot.errorbar(xi3, integ_median, np.vstack([integ_lower, integ_upper]), linestyle='None', marker='o',color='c',capsize=4,
                elinewidth=3,label='IEO')
    pyplot.legend(loc='upper right', fontsize=label_font_size)#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    pyplot.xlabel('n', fontsize=label_font_size)
    pyplot.ylabel('$R(\hat{\omega})$', fontsize=label_font_size)
    pyplot.grid()
    pyplot.xticks(xi, n_list, fontsize=tick_font_size)#, rotation=90)
    pyplot.yticks(fontsize=tick_font_size)#, rotation=90)
    name='portfolio_opti_' + specification + '_model' 
    pyplot.savefig(name+'_.png', bbox_inches='tight')

    # plot log regret 
    erm_arr = np.log(np.maximum(erm_arr, 0))
    integ_arr = np.log(np.maximum(integ_arr, 0))
    pto_arr = np.log(np.maximum(pto_arr, 0))
    erm_median = np.quantile(erm_arr, 0.5,axis=1)
    pto_median = np.quantile(pto_arr, 0.5,axis=1)
    integ_median = np.quantile(integ_arr, 0.5,axis=1)
    erm_lower = erm_median  - np.quantile(erm_arr, 0.25,axis=1)
    erm_upper = np.quantile(erm_arr, 0.75,axis=1) - erm_median
    pto_lower = pto_median - np.quantile(pto_arr, 0.25,axis=1)
    pto_upper = np.quantile(pto_arr, 0.75,axis=1) - pto_median
    integ_lower = integ_median - np.quantile(integ_arr, 0.25,axis=1)
    integ_upper = np.quantile(integ_arr, 0.75,axis=1) - integ_median
    pyplot.cla()
    title=specification   
    pyplot.title(title, fontsize=label_font_size)
    pyplot.errorbar(xi1, erm_median, np.vstack([erm_lower, erm_upper]), linestyle='None', marker='o',color='r',capsize=4,
                elinewidth=3,label='EO') #marker='_' is a horizontal line 
    pyplot.errorbar(xi2, pto_median, np.vstack([pto_lower, pto_upper]), linestyle='None', marker='o',color='g',capsize=4,
                elinewidth=3,label='ETO')
    pyplot.errorbar(xi3, integ_median, np.vstack([integ_lower, integ_upper]), linestyle='None', marker='o',color='c',capsize=4,
                elinewidth=3,label='IEO')
    pyplot.legend(loc='upper right', fontsize=label_font_size)#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    pyplot.xlabel('n', fontsize=label_font_size)
    pyplot.ylabel('$R(\hat{\omega})$', fontsize=label_font_size)
    pyplot.grid()
    pyplot.xticks(xi, n_list, fontsize=tick_font_size)#, rotation=90)
    pyplot.yticks(fontsize=tick_font_size)#, rotation=90)
    name='portfolio_opti_' + specification + '_model_log_regret' 
    pyplot.savefig(name+'_.png', bbox_inches='tight')
    breakpoint()

def plot_asy_std(threshold, n_list,erm_arr,pto_arr,integ_arr):
    # plot the standard deviation of (n * regret)
    pyplot.clf()
    erm_perc = np.zeros(len(n_list)); 
    pto_perc = np.zeros(len(n_list)); 
    integ_perc = np.zeros(len(n_list)); 
    for i in range(len(n_list)):
        erm_perc[i] = np.average(n_list[i]* erm_arr[i] > threshold)
        pto_perc[i] = np.average(n_list[i] * pto_arr[i] > threshold)
        integ_perc[i] = np.average(n_list[i] * integ_arr[i] > threshold)
    # axis_threshold = max(max(erm_arr),max(pto_arr),max(integ_arr))
    pyplot.plot(n_list, erm_perc, label='EO')
    pyplot.plot(n_list, pto_perc, label='PF')
    pyplot.plot(n_list, integ_perc, label='IO')
    title = 'P(n * regret > ' + str(threshold) + ')'
    pyplot.legend(loc="upper right")
    pyplot.title(title)
    pyplot.xlabel('training sample size');
    pyplot.ylabel('P(n * regret > ' + str(threshold) + ')');
    pyplot.grid()
    name = str(threshold) + '_n' + str(n_list[-1]) + '_' + str(n_seed) + 'seeds'
    pyplot.savefig('asy_threshold_' + name + '.png')

def plot_moments(n_list,erm_arr,pto_arr,integ_arr):
    pyplot.clf()
    erm_avg = np.average(erm_arr,axis=1)
    pto_avg = np.average(pto_arr,axis=1)
    integ_avg = np.average(integ_arr,axis=1)
    title='First sample moment'    
    pyplot.title(title)
    pyplot.plot(n_list, erm_avg * n_list, label='EO')
    pyplot.plot(n_list, pto_avg * n_list, label='PF')
    pyplot.plot(n_list, integ_avg * n_list, label='IO')
    pyplot.legend(loc='upper right')#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    pyplot.xlabel('training sample size')
    pyplot.ylabel('First sample moment')
    pyplot.grid()
    name = '_n' + str(n_list[-1]) + '_' + str(n_seed) + 'seeds'
    pyplot.savefig('1st_moment_'+name+'_.png')

    pyplot.clf()
    # erm_arr * erm_arr is not correct! this is a 2d array 
    erm_avg = np.average(np.square(erm_arr) ,axis=1) # raised to square
    pto_avg = np.average(np.square(pto_arr), axis=1)
    integ_avg = np.average(np.square(integ_arr) ,axis=1)
    title='Second sample moment'    
    pyplot.title(title)
    pyplot.plot(n_list, erm_avg * np.square(n_list), label='EO')
    pyplot.plot(n_list, pto_avg * np.square(n_list), label='PF')
    pyplot.plot(n_list, integ_avg * np.square(n_list), label='IO')
    pyplot.legend(loc='upper right')#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    pyplot.xlabel('training sample size')
    pyplot.ylabel('Second sample moment')
    pyplot.grid()
    name = '_n' + str(n_list[-1]) + '_' + str(n_seed) + 'seeds'
    pyplot.savefig('2nd_moment_'+name+'_.png')
    

label_font_size=20
tick_font_size=15
n_seed= 100 # 500
plot_ = 'error_bar' # choose from 'convergence', and 'error_bar'
n_products = 2
threshold = 1 # threshold for verifying convergence in distribution
alpha = 0.7 # variance penalty in the objective


specification = 'Misspecified' # 'Correctly specified' or 'Misspecified'
true_gaus_std = np.arange(1, n_products + 1) * 3 
if specification == 'Correctly specified':
    gaus_std = true_gaus_std#  correctly specified model
else:
    gaus_std = 3 * np.arange(1, n_products + 1)[::-1] 

if plot_ == 'convergence':
    n_list=[100,200,300,400,500]
    erm_arr,pto_arr,integ_arr = get_regret_arr(n_list, n_seed)
    plot_asy_std(1, n_list,erm_arr,pto_arr,integ_arr)
    plot_asy_std(0.5, n_list,erm_arr,pto_arr,integ_arr)
    plot_moments(n_list,erm_arr,pto_arr,integ_arr)
    
else:
    n_list=[10,20,30,40,50]
    erm_arr,pto_arr,integ_arr = get_regret_arr(n_list, n_seed)
    plot_error_bar(n_list,erm_arr,pto_arr,integ_arr)


