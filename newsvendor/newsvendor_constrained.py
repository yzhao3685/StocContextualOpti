import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from gurobipy import *
from sklearn.linear_model import LassoLarsCV
from scipy.stats import norm
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from warnings import simplefilter
from matplotlib import pyplot
import time
import pdb
simplefilter(action='ignore')# ignore all warnings

b,h=5,1#backordering, holding cost
deg =1; erm_deg=1

def ERM_policy(d_arr):
    n=len(d_arr)
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    u=m.addMVar ((n,n_products), ub=10000.0,lb=0.0,name="u" )
    v=m.addMVar ((n,n_products), ub=10000.0,lb=0.0,name="v" )
    q=m.addMVar (n_products, ub=10000.0,lb=-10000.0,name="q" )
    ones = np.ones(n)
    m.setObjective(sum(ones@ u[:, j]*b+ones@v[:,j]*h for j in range(n_products)), GRB.MINIMIZE)
    for i in range(n):
        for j in range(n_products):
            m.addConstr(u[i,j]>=d_arr[i,j] - q[j])
            m.addConstr(v[i,j]>=q[j] - d_arr[i,j])
    m.addConstr(cons_beta@q <= cons_B)
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()    
    q_star=np.zeros(n_products)
    for i in range(n_products):
        q_star[i]=q[i].x
    return q_star

def Integrated_decision_gaussian_uncons(d_arr,gaus_std):
    n=len(d_arr)
    theta_len = n_products
    constant = norm.ppf(b/(b+h))
    #optimize
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    u=m.addMVar ((n,n_products), ub=10000.0,lb=0.0,name="u" )
    v=m.addMVar ((n,n_products), ub=10000.0,lb=0.0,name="v" )
##    gaus_mean=m.addMVar (n_products, ub=10000.0,lb=-10000.0,name="gaus_mean" )
    gaus_mean=m.addVar (ub=10000.0,lb=-10000.0,name="gaus_mean" )
    ones = np.ones(n)
    m.setObjective(sum(ones@ u[:, j]*b+ones@v[:,j]*h for j in range(n_products)), GRB.MINIMIZE)
    for i in range(n):
        for j in range(n_products):
            m.addConstr(u[i,j]>=d_arr[i,j] - ((j+1)*gaus_mean+constant*gaus_std[j]))
            m.addConstr(v[i,j]>=((j+1)*gaus_mean+constant*gaus_std[j]) - d_arr[i,j])
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()
    pred_mean=gaus_mean.x
    #print('pred mean: ',pred_mean)
    return pred_mean*np.arange(1,1+n_products)+gaus_std*constant

def Integrated_decision_gaussian(d_arr,gaus_std):
##    possible_mean_arr = np.arange(2.1,4.1,0.01)
    possible_mean_arr = np.arange(2.1,4.1,0.1)
    fun_val_arr = np.zeros(len(possible_mean_arr))
    n=len(d_arr)
    for i in range(len(possible_mean_arr)):
        #the model assumes means for different products are multiples of the mean of the first product
        model_mean_arr = np.arange(1,6)*possible_mean_arr[i] 
        w_arr = optimal_policy_gaussian(model_mean_arr,gaus_std)
        w_arr_tile = np.tile(w_arr,(n,1))
        temp = np.average(b*np.maximum(0,d_arr - w_arr_tile) + h*np.maximum(0,w_arr_tile - d_arr),axis=0)
        fun_val_arr[i] = sum(temp)
    IO_pred = possible_mean_arr[np.argmin(fun_val_arr)]
    model_mean_arr = np.arange(1,6)*IO_pred
    w_arr = optimal_policy_gaussian(model_mean_arr,gaus_std)
    return w_arr
        

def Integrated_prediction_uniform_ub(x_arr,d_arr):#to update
    n=len(d_arr)
    #transform x_arr
    M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_arr)
    theta_len = 1 #M.shape[1]
    constant = b/(b+h)
    #optimize
    penalty=0.01
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    u=m.addMVar (n, ub=10000.0,lb=0.0,name="u" )
    v=m.addMVar (n, ub=10000.0,lb=0.0,name="v" )
    theta=m.addMVar (1, ub=10000.0,lb=-10000.0,name="theta" )
    ones = np.ones(n)
    identity = np.identity(n)
    m.setObjective(ones@u*b+ones@v*h+theta@theta*penalty, GRB.MINIMIZE)
    m.addConstrs(u@identity[i]>=d_arr[i]-(theta*constant) for i in range(n))
    m.addConstrs(v@identity[i]>=theta*constant-d_arr[i] for i in range(n))
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()    
    theta_star=np.zeros(theta_len)
    for i in range(theta_len):
        theta_star[i]=theta[i].x
    return theta_star#theta@M is prediction of uniform ub


def estimate_uniform_mle(x_arr,d_arr,deg_uniform_mle):#to update
    n=len(d_arr)
    #transform x_arr
    M=PolynomialFeatures(degree=deg_uniform_mle, include_bias=True).fit_transform(x_arr)
    sum_z=np.sum(M,axis=0)
    theta_len = len(sum_z)
    #optimize
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    theta=m.addMVar (theta_len, ub=10000.0,lb=-10000.0,name="theta" )
    m.setObjective(theta@sum_z, GRB.MINIMIZE)
    m.addConstrs(theta@M[i]>=d_arr[i] for i in range(n))
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()    
    theta_star=np.zeros(theta_len)
    for i in range(theta_len):
        theta_star[i]=theta[i].x
    return theta_star

def optimal_policy_uniform(ub,lb):
    return (h*lb+b*ub)/(b+h)

def optimal_policy_gaussian(mean_arr,std_arr):
    #input cons_beta and represent coeffs and rhs in constraints respectively
    n=len(mean_arr); ratio=b/(b+h); w_arr=np.zeros(n)
    w_uncons = np.ones(n)*norm.ppf(ratio)*std_arr+mean_arr
    #now start binary search algorithm
    if cons_beta@w_uncons <= cons_B:
        return w_uncons
    r_l = min(np.divide(-b,cons_beta)) #assume beta > 0, which is reasonable
    r_u=0; tolerance = 1e-5 # 1e-10
    while abs(cons_beta@w_arr - cons_B) > tolerance:
        r=(r_l+r_u)/2
        for i in range(n):
            temp = (r*cons_beta[i]+b)/(h+b)
            if temp > norm.cdf(0,mean_arr[i],std_arr[i]):
                w_arr[i] = norm.ppf(temp,mean_arr[i],std_arr[i])
            else:
                w_arr[i]=0
        if cons_beta@w_arr < cons_B:
            r_l=r
        else:
            r_u=r
    return w_arr


def loss_f(w_arr,d_arr):
    #print('d_arr shape:',d_arr.shape,'w_arr shape',w_arr.shape)
    loss_arr = h*np.maximum(0,w_arr-d_arr)+b*np.maximum(0,d_arr-w_arr)
    return np.average(loss_arr)

def integrated_decisions(d_train,param_arr,dist='gaussian'):
    if dist=='gaussian':
        gaus_std = param_arr
        w_IO = Integrated_decision_gaussian(d_train,gaus_std)
        return w_IO
    else:#uniform  #to update
        theta = Integrated_prediction_uniform_ub(x_train,d_train)
        M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_test)
        ub_arr = theta;#estimate of uniform ub
        lb_arr = np.zeros(x_test.shape[0])
        w_PTO = optimal_policy_uniform(ub_arr,lb_arr)
        return w_PTO

def PTO_decisions(d_train,param_arr,dist='gaussian'):
    if dist=='gaussian':
        n=len(d_train)
        temp1 = sum(d_train@np.arange(1,1+n_products))
        temp2 = np.power(np.arange(1,1+n_products),2)
        mle_mean = temp1/(n*sum(temp2))
        mle_mean = mle_mean*np.arange(1,1+n_products)
        mle_std = param_arr
        w_PTO = optimal_policy_gaussian(mle_mean,mle_std)
        return w_PTO
    else:#uniform  #to update
        #theta = estimate_uniform_mle(x_train,d_train,deg)
        #M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_test)
        ub_arr = max(d_train); lb_arr = np.zeros(x_test.shape[0])
        w_PTO = optimal_policy_uniform(ub_arr,lb_arr)

def generate_data(train_size,test_size,dist='gaussian'):
    if dist=='gaussian':
        gaus_mean = np.arange(1,n_products+1)*3
        std_arr = np.ones(5)
        d_train=np.random.normal(np.tile(gaus_mean,(train_size,1)),np.tile(std_arr,(train_size,1)))
        d_train = np.maximum(d_train,0)

        d_test=np.random.normal(np.tile(gaus_mean,(test_size,1)),np.tile(std_arr,(test_size,1)))
        d_test = np.maximum(d_test,0)

        w_oracle = optimal_policy_gaussian(gaus_mean,std_arr)
        w_oracle = np.tile(w_oracle,(test_size,1))
        #w_oracle is the best in expectation. it could happen that on finite sample, some sub-optimal decision actually does better because of noise in d_test
        return  d_train,d_test,w_oracle
    else:# dist=='uniform':
        x_train=np.ones((train_size,2))
        lb = np.zeros(train_size)
        ub = 1+(x_train[:,0]+x_train[:,1])/2 #demand is uniform [0,1]
        d_train=np.random.uniform(lb,ub,size=train_size)

        x_test=np.ones((test_size,2))
        lb = np.zeros(test_size)
        ub = 1+(x_test[:,0]+x_test[:,1])/2
        d_test=np.random.uniform(lb,ub,size=test_size)

        w_oracle = optimal_policy_uniform(ub,lb)
        return  x_train,d_train,x_test,d_test,w_oracle

def get_regret(train_size,test_size,data_dist='gaussian',model_dist='gaussian'):
    d_train,d_test,w_oracle = generate_data(train_size,test_size,dist=data_dist)
    oracle_loss = loss_f(w_oracle,d_test)

    w_ERM = ERM_policy(d_train)
    w_ERM = np.tile(w_ERM, (test_size,1))
    ERM_regret=loss_f(w_ERM,d_test)-oracle_loss

    w_IO = integrated_decisions(d_train,gaus_std,dist=model_dist)
    w_IO = np.tile(w_IO, (test_size,1))
    integrated_regret = loss_f(w_IO,d_test)-oracle_loss

    w_PTO = PTO_decisions(d_train,gaus_std,dist=model_dist)
    w_PTO = np.tile(w_PTO, (test_size,1))
    PTO_regret=loss_f(w_PTO,d_test)-oracle_loss
    #print('n: ',train_size,'erm ',ERM_regret,'IO ',integrated_regret,'PF ',PTO_regret)
    return ERM_regret,PTO_regret,integrated_regret

def get_regret_arr(data_dist,model_dist,n_seed=5):
    #return array of regrets
    t1=time.time()
    erm_arr = np.zeros((len(n_list),n_seed))
    pto_arr = np.zeros((len(n_list),n_seed))
    integ_arr = np.zeros((len(n_list),n_seed))
    for i in range(len(n_list)):
        n=n_list[i]
        for j in range(n_seed):
            erm_regret, pto_regret,integ_regret = get_regret(n,10000,data_dist,model_dist)
            erm_arr [i,j]+=erm_regret
            pto_arr[i,j] +=pto_regret
            integ_arr[i,j]+=integ_regret
            if j%5==0:
                print('seed: ',j,'done')
        print('n: ',n,'EO ',np.average(erm_arr[i]),'IO ',np.average(integ_arr[i]),'PF ',np.average(pto_arr[i]))
    time_elapsed=time.time()-t1
    print('total runtime: ',round(time_elapsed))
    erm_arr = np.maximum(erm_arr, np.zeros((len(n_list),n_seed)))
    pto_arr = np.maximum(pto_arr, np.zeros((len(n_list),n_seed)))
    integ_arr = np.maximum(integ_arr, np.zeros((len(n_list),n_seed)))
    return erm_arr,pto_arr,integ_arr

def plot_curve(erm_arr,pto_arr,integ_arr):
    pyplot.clf()
    erm_arr=np.average(erm_arr,axis=1);pto_arr = np.average(pto_arr,axis=1);integ_arr = np.average(integ_arr,axis=1)
    threshold = max(max(erm_arr),max(pto_arr),max(integ_arr))
    pyplot.plot(n_list, erm_arr,label='EO')
    pyplot.plot(n_list, pto_arr,label='PF')
    pyplot.plot(n_list, integ_arr,label='IO')
    title='regret against sample size'
    pyplot.legend(loc="upper right")
    pyplot.title(title)
    pyplot.xlabel('training sample size');pyplot.ylabel('regret');pyplot.grid()
    name='true_'+data_dist+'_model_'+model_dist
    pyplot.savefig('curve_'+name+'_.png')    

def plot_histogram(n,erm_arr,pto_arr,integ_arr):
    pyplot.clf()
    hist_weights = np.ones_like(erm_arr) / n_seed
    pyplot.hist(erm_arr, weights=hist_weights, bins=10, alpha=0.5, label='EO')
    hist_weights = np.ones_like(pto_arr) / n_seed
    pyplot.hist(pto_arr, weights=hist_weights, bins=10, alpha=0.5, label='PF')
    hist_weights = np.ones_like(integ_arr) / n_seed
    pyplot.hist(integ_arr, weights=hist_weights, bins=10, alpha=0.5, label='IO')
##    pyplot.hist(erm_arr,density=True, bins=10, alpha=0.5,label='EO')
##    pyplot.hist(pto_arr,density=True, bins=10, alpha=0.5,label='PF')
##    pyplot.hist(integ_arr,density=True, bins=10, alpha=0.5,label='IO')
    title='histogram of regret'
    pyplot.legend(loc="upper right")
    pyplot.title(title)
    pyplot.xlabel('regret');pyplot.ylabel('percentage');pyplot.grid()
    #pyplot.show()
    #pyplot.axvline(x=expected_val_test,color='r',label='true rev')
    name='true_'+data_dist+'_model_'+model_dist
    pyplot.savefig('n_'+str(n)+'_'+name+'_.png')    

def plot_error_bar(erm_arr,pto_arr,integ_arr):
    pyplot.clf()
    xi = [1,2,3,4,5] # create an index for each tick position
    xi1 = [0.7,1.7,2.7,3.7,4.7]#positions of error bars
    xi2 = [0.85,1.85,2.85,3.85,4.85]
    xi3 = [1.0,2.0,3.0,4.0,5.0]
    #xi4 = [1.15,2.15,3.15,4.15]; xi5 = [1.3,2.3,3.3,4.3]
##    erm_avg = np.average(erm_arr,axis=1);pto_avg = np.average(pto_arr,axis=1);integ_avg = np.average(integ_arr,axis=1)
    erm_medium = np.median(erm_arr,axis=1);pto_medium = np.median(pto_arr,axis=1);integ_medium = np.median(integ_arr,axis=1)
    erm_lower = erm_medium  - np.quantile(erm_arr, 0.25,axis=1)
    erm_upper = np.quantile(erm_arr, 0.75,axis=1) - erm_medium
    pto_lower = pto_medium - np.quantile(pto_arr, 0.25,axis=1)
    pto_upper = np.quantile(pto_arr, 0.75,axis=1) - pto_medium
    integ_lower = integ_medium - np.quantile(integ_arr, 0.25,axis=1)
    integ_upper = np.quantile(integ_arr, 0.75,axis=1) - integ_medium
    title=specification #'constrained case'    
    pyplot.title(title, fontsize=label_font_size)
    pyplot.errorbar(xi1, erm_medium, np.vstack([erm_lower, erm_upper]), linestyle='None', marker='o',color='r',capsize=4,
                elinewidth=3,label='EO')
    pyplot.errorbar(xi2, pto_medium, np.vstack([pto_lower, pto_upper]), linestyle='None', marker='o',color='g',capsize=4,
                elinewidth=3,label='ETO')
    pyplot.errorbar(xi3, integ_medium, np.vstack([integ_lower, integ_upper]), linestyle='None', marker='o',color='c',capsize=4,
                elinewidth=3,label='IEO')
    pyplot.legend(loc='upper right', fontsize=label_font_size)#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    pyplot.xlabel('training sample size', fontsize=label_font_size)
    pyplot.ylabel('regret', fontsize=label_font_size)
    pyplot.grid()
    pyplot.xticks(xi, n_list, fontsize=tick_font_size)
    pyplot.yticks(fontsize=tick_font_size)#, rotation=90)
    #pyplot.show()
    name=specification #'true_'+data_dist+'_model_'+model_dist
    pyplot.savefig('error_bar_'+name+'_.png', bbox_inches='tight')    


label_font_size=20
tick_font_size=15
data_dist='gaussian'; model_dist='gaussian'
n_seed = 25 #25
plot_ = 0# if 0, plot  curve and error bar
n_products = 5

data_dist ='gaussian'
model_dist='gaussian'
specification = 'Misspecified' # 'Misspecified'
if specification == 'Correctly specified':
    gaus_std=np.ones(n_products)#  correctly specified model
else:
##    gaus_std = np.array([5,4,3,2,1])*6 # misspecified model
    gaus_std = np.array([5,4,3,2,1])*6 # misspecified model

cons_beta = np.ones(n_products)
cons_B = 40

# unconstrained case, sum(w_oracle) approximately 44

##n_list=[10,20,30,40,50]
# n_list=[25]
##n_list=[20,40,60,80,100]
n_list=[40,80,120,160,200]
erm_arr,pto_arr,integ_arr = get_regret_arr(data_dist,model_dist, n_seed)
    
if plot_==1:
    for i_n in range(len(n_list)):
        plot_histogram(n_list[i_n],erm_arr[i_n],pto_arr[i_n],integ_arr[i_n])
else:
    # plot_curve(erm_arr,pto_arr,integ_arr)
    plot_error_bar(erm_arr,pto_arr,integ_arr)


