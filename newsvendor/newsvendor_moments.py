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

b,h=4,5#backordering, holding cost
deg =1; erm_deg=1

def ERM_policy_separate(d_arr):
    # this is slower. can instead solve a separate optimization for each product 
    n=len(d_arr)
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    upper_bound = 50.0 # largest demand is 5*3=15. so don't need any larger order/backordering/holding quantity
    q_star=np.zeros(n_products)
    for j in range(n_products):
        u=m.addMVar(n, ub=upper_bound,lb=0.0,name="u" )
        v=m.addMVar(n, ub=upper_bound,lb=0.0,name="v" )
        q=m.addVar (ub=upper_bound,lb=0.0,name="q" )
        ones = np.ones(n)
        m.setObjective(ones@ u*b+ones@v*h, GRB.MINIMIZE)
        for i in range(n):
            m.addConstr(u[i]>=d_arr[i,j] - q)
            m.addConstr(v[i]>=q - d_arr[i,j])
        m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
        #m.params.NonConvex=2
        m.update()
        m.optimize()    
        q_star[j]=q.x
    return q_star

def ERM_policy(d_arr):
    n=len(d_arr)
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    upper_bound = 50.0 # largest demand is 5*3=15. so don't need any larger order/backordering/holding quantity 
    u=m.addMVar ((n,n_products), ub=upper_bound,lb=0.0,name="u" )
    v=m.addMVar ((n,n_products), ub=upper_bound,lb=0.0,name="v" )
    q=m.addMVar (n_products, ub=upper_bound,lb=0.0,name="q" )
    ones = np.ones(n)
    m.setObjective(sum(ones@ u[:, j]*b+ones@v[:,j]*h for j in range(n_products)), GRB.MINIMIZE)
    for i in range(n):
        for j in range(n_products):
            m.addConstr(u[i,j]>=d_arr[i,j] - q[j])
            m.addConstr(v[i,j]>=q[j] - d_arr[i,j])
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()    
    q_star=np.zeros(n_products)
    for i in range(n_products):
        q_star[i]=q[i].x
    return q_star

def ERM_policy_scipy(d_arr):
    n = len(d_arr)
    obj = np.zeros(2*n_products*n + n_products) #first n*d terms are u, next n*d terms are v, last d terms are q
    obj[:n*n_products] = np.ones(n*n_products)*b
    obj[n*n_products:2*n*n_products] = np.ones(n*n_products)*h
    A_ub = np.zeros((2*n*n_products, 2*n*n_products + n_products))
    A_ub[:, :2*n*n_products] = np.identity(2*n*n_products)
    tiled_identity = np.tile(np.identity(n_products), (n, 1))
    tiled_identity = np.vstack((tiled_identity, -tiled_identity))
    A_ub[:, 2*n*n_products:] = tiled_identity
    b_ub = np.append(d_arr.reshape((-1,1)), -d_arr.reshape((-1,1)))
    # By default, bounds are (0, None)
    result = linprog(obj, -A_ub, -b_ub, options={'tol': 1e-3})
    q_star = result.x[-5:]
    return q_star 

def Integrated_decision_gaussian(d_arr,gaus_std):
    n=len(d_arr)
    theta_len = n_products
    constant = norm.ppf(b/(b+h))
    upper_bound = 50.0 # largest demand is 5*3=15. so don't need any larger order/backordering/holding quantity 
    #optimize
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    u=m.addMVar ((n,n_products), ub=upper_bound,lb=0.0,name="u" )
    v=m.addMVar ((n,n_products), ub=upper_bound,lb=0.0,name="v" )
##    gaus_mean=m.addMVar (n_products, ub=10000.0,lb=-10000.0,name="gaus_mean" )
    gaus_mean=m.addVar (ub=upper_bound,lb=0.0,name="gaus_mean" ) 
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

def Integrated_decision_gaussian_scipy(d_arr,gaus_std):
    n = len(d_arr)
    constant = norm.ppf(b/(b+h))

    obj = np.zeros(2*n_products*n + 1) #first n*d terms are u, next n*d terms are v, last d terms are q
    obj[:n*n_products] = np.ones(n*n_products)*b
    obj[n*n_products:2*n*n_products] = np.ones(n*n_products)*h
    
    A_ub = np.zeros((2*n*n_products, 2*n*n_products + 1))
    A_ub[:, :2*n*n_products] = np.identity(2*n*n_products)
    tiled_arange = np.tile(np.arange(1, 1 + n_products), n)
    tiled_arange = np.append(tiled_arange, -tiled_arange)
    A_ub[:, -1] = tiled_arange
    b_ub = np.append(d_arr.reshape((-1,1)), -d_arr.reshape((-1,1)))
    std_tiled = np.tile(gaus_std*constant, n) 
    b_ub += np.append(-std_tiled, std_tiled)
    # By default, bounds are (0, None)
    result = linprog(obj, -A_ub, -b_ub, options={'tol': 1e-3})
    pred_mean_scipy = result.x[-1]
    return pred_mean_scipy*np.arange(1,1+n_products)+gaus_std*constant

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
    theta=m.addMVar (1, ub=50.0,lb=0.0,name="theta" )
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
    n=len(mean_arr)
    ratio=b/(b+h)
    #w_arr = norm.ppf(np.ones(n)*ratio,loc=mean_arr,scale=std_arr)#same value as below, more compute
    w_arr = np.ones(n)*norm.ppf(ratio)*std_arr+mean_arr
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
        mean_arr = np.arange(1,n_products+1)*3
        std_arr = np.ones(n_products)
        d_train=np.random.normal(np.tile(mean_arr,(train_size,1)),np.tile(std_arr,(train_size,1)))
        d_train = np.maximum(d_train,0)

        d_test=np.random.normal(np.tile(mean_arr,(test_size,1)),np.tile(std_arr,(test_size,1)))
        d_test = np.maximum(d_test,0)

        w_oracle = optimal_policy_gaussian(mean_arr,std_arr)
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

    # w_ERM = ERM_policy(d_train)
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

def collect_result(result):
    # ERM_regret,PTO_regret,integrated_regret
    erm_subprocess_list.append(result[0])
    pto_subprocess_list.append(result[1])
    io_subprocess_list.append(result[2])
    
def get_regret_arr(n_list, data_dist,model_dist,n_seed=5,multiprocess=0):
    #return array of regrets
    t1=time.time()
    erm_arr = np.zeros((len(n_list),n_seed))
    pto_arr = np.zeros((len(n_list),n_seed))
    integ_arr = np.zeros((len(n_list),n_seed))
    global erm_subprocess_list, pto_subprocess_list, io_subprocess_list
    erm_subprocess_list = []; pto_subprocess_list = []; io_subprocess_list = []
    for i in range(len(n_list)):
        n=n_list[i]
        if multiprocess == 1:
            numCPU=mp.cpu_count()
            pool=mp.Pool(numCPU)
            for j in range(n_seed):
                pool.apply_async(get_regret, args=(n,10000,data_dist,model_dist),
                         callback=collect_result)
            pool.close()
            pool.join()
            erm_arr[i] = np.array(erm_subprocess_list)
            pto_arr[i] = np.array(pto_subprocess_list)
            integ_arr[i] = np.array(io_subprocess_list)
        else:
            for j in range(n_seed):
                erm_regret, pto_regret,integ_regret = get_regret(n,10000,data_dist,model_dist)
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

def plot_curve(n_list,erm_arr,pto_arr,integ_arr):
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
    pyplot.hist(erm_arr,density=True, bins=10, alpha=0.5,label='EO')
    pyplot.hist(pto_arr,density=True, bins=10, alpha=0.5,label='PF')
    pyplot.hist(integ_arr,density=True, bins=10, alpha=0.5,label='IO')
    title='histogram of regret'
    pyplot.legend(loc="upper right")
    pyplot.title(title)
    pyplot.xlabel('regret');pyplot.ylabel('percentage');pyplot.grid()
    #pyplot.show()
    #pyplot.axvline(x=expected_val_test,color='r',label='true rev')
    name='true_'+data_dist+'_model_'+model_dist
    pyplot.savefig('n_'+str(n)+'_'+name+'_.png')    

def plot_error_bar(n_list,erm_arr,pto_arr,integ_arr):
    pyplot.clf()
    xi = [1,2,3,4]# create an index for each tick position
    xi1 = [0.7,1.7,2.7,3.7]#positions of error bars
    xi2 = [0.85,1.85,2.85,3.85]
    xi3 = [1.0,2.0,3.0,4.0]
    #xi4 = [1.15,2.15,3.15,4.15]; xi5 = [1.3,2.3,3.3,4.3]
    erm_avg = np.average(erm_arr,axis=1);pto_avg = np.average(pto_arr,axis=1);integ_avg = np.average(integ_arr,axis=1)
    erm_std = np.std(erm_arr,axis=1);pto_std = np.std(pto_arr,axis=1);integ_std = np.std(integ_arr,axis=1)
    title='unconstrained case'    
    pyplot.title(title)
    pyplot.errorbar(xi1, erm_avg, erm_std, linestyle='None', marker='o',color='r',capsize=4,
                elinewidth=3,label='EO')
    pyplot.errorbar(xi2, pto_avg, pto_std, linestyle='None', marker='o',color='g',capsize=4,
                elinewidth=3,label='PF')
    pyplot.errorbar(xi3, integ_avg, integ_std, linestyle='None', marker='o',color='c',capsize=4,
                elinewidth=3,label='IO')
    pyplot.legend(loc='upper right')#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    pyplot.xlabel('training sample size');pyplot.ylabel('regret');pyplot.grid()
    pyplot.xticks(xi, n_list)
    #pyplot.show()
    name='true_'+data_dist+'_model_'+model_dist
    pyplot.savefig('error_bar_'+name+'_.png')

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
    pyplot.plot(n_list, erm_perc, label='EO',color='r')
    pyplot.plot(n_list, pto_perc, label='ETO',color='g')
    pyplot.plot(n_list, integ_perc, label='IEO',color='c')
    # title = 'P(n * regret > ' + str(threshold) + ')'
    pyplot.legend(loc="upper right", fontsize=label_font_size)
    # pyplot.title(title, fontsize=label_font_size)
    pyplot.xlabel('n', fontsize=label_font_size);
    pyplot.ylabel('$\mathbb{P}(nR(\hat{\omega}) > $' + str(threshold) + ')', fontsize=label_font_size);
    pyplot.xticks(fontsize=tick_font_size, rotation=90)
    pyplot.yticks(fontsize=tick_font_size) #, rotation=90)
    pyplot.grid()
    name = str(threshold) + '_n' + str(n_list[-1]) + '_' + str(n_seed) + 'seeds'
    pyplot.savefig('asy_threshold_' + name + '.png', bbox_inches='tight')

def plot_moments(n_list,power,label, erm_arr,pto_arr,integ_arr):
    pyplot.clf()
    # erm_arr * erm_arr is not correct! this is a 2d array 
    erm_avg = np.average(np.power(erm_arr, power) ,axis=1) # raised to square
    pto_avg = np.average(np.power(pto_arr, power), axis=1)
    integ_avg = np.average(np.power(integ_arr, power) ,axis=1)
    # title=label # E[(n * regret)^power] = n^2 E[regret^2]
    # pyplot.title(title, fontsize=label_font_size)
    pyplot.plot(n_list, erm_avg * np.power(n_list, power), label='EO', color='r')
    pyplot.plot(n_list, pto_avg * np.power(n_list, power), label='ETO', color='g')
    pyplot.plot(n_list, integ_avg * np.power(n_list, power), label='IEO', color='c')
    pyplot.legend(loc='upper right', fontsize=label_font_size)#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    pyplot.xlabel('n', fontsize=label_font_size)
    pyplot.ylabel(label, fontsize=label_font_size)
    pyplot.xticks(fontsize=tick_font_size, rotation=90)
    pyplot.yticks(fontsize=tick_font_size)#, rotation=90)
    pyplot.grid()
    name = '_n' + str(n_list[-1]) + '_' + str(n_seed) + 'seeds'
    pyplot.savefig(str(power)+'_moment_'+name+'_.png', bbox_inches='tight')
    

label_font_size=20
tick_font_size=15
data_dist='gaussian'; model_dist='gaussian'
n_seed=500
plot_ = 'convergence' # choose from 'histogram', 'convergence', and 'error_bar'
n_products = 2
multiprocess = 0
threshold = 1 # threshold for verifying convergence in distribution

gaus_std=np.ones(n_products)#  correctly specified model
#gaus_std = np.array([5,4,3,2,1])*3 #incorrectly specified model
    
if plot_ == 'histogram':
    n_list = [25]
    erm_arr,pto_arr,integ_arr = get_regret_arr(n_list,data_dist,model_dist, n_seed)
    for i_n in range(len(n_list)):
        plot_histogram(n_list[i_n],erm_arr[i_n],pto_arr[i_n],integ_arr[i_n])
elif plot_ == 'convergence':
    n_list=[20,40,60,80,100]
#    n_list=[100,200,300,400,500]
    erm_arr,pto_arr,integ_arr = get_regret_arr(n_list,data_dist,model_dist, n_seed, multiprocess)
    plot_asy_std(1.5, n_list,erm_arr,pto_arr,integ_arr)
    plot_asy_std(1, n_list,erm_arr,pto_arr,integ_arr)
    plot_asy_std(0.5, n_list,erm_arr,pto_arr,integ_arr)
    plot_moments(n_list,1, '$\mathbb{E}[nR(\hat{\omega})]$', erm_arr,pto_arr,integ_arr)
    plot_moments(n_list,2, '$\mathbb{E}[(nR(\hat{\omega}))^2]$', erm_arr,pto_arr,integ_arr)
    plot_moments(n_list,3, '$\mathbb{E}[(nR(\hat{\omega}))^3]$', erm_arr,pto_arr,integ_arr)
    
else:
    n_list=[10,25,50]
    erm_arr,pto_arr,integ_arr = get_regret_arr(n_list,data_dist,model_dist, n_seed)
    plot_curve(n_list,erm_arr,pto_arr,integ_arr)
    plot_error_bar(n_list,erm_arr,pto_arr,integ_arr)


