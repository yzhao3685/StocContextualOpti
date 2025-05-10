import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from gurobipy import *
import gurobipy as gp
from sklearn.linear_model import LassoLarsCV
from scipy.stats import norm
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from warnings import simplefilter
from matplotlib import pyplot
import time
simplefilter(action='ignore')# ignore all warnings
with gp.Env(empty=True) as env:
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)

b,h=5,1#backordering, holding cost
deg =2

def ERM_policy(x_arr,d_arr):
    n=len(d_arr)
    #transform x_arr
    M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_arr)
    q_len = M.shape[1]
    #optimize
    penalty=0.01
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    u=m.addMVar (n, ub=10000.0,lb=0.0,name="u" )
    v=m.addMVar (n, ub=10000.0,lb=0.0,name="v" )
    q=m.addMVar (q_len, ub=10000.0,lb=-10000.0,name="q" )
    ones = np.ones(n)
    identity = np.identity(n)
    m.setObjective(ones@u*b+ones@v*h+q@q*penalty, GRB.MINIMIZE)
    m.addConstrs(u@identity[i]>=d_arr[i]-M[i]@q for i in range(n))
    m.addConstrs(v@identity[i]>=M[i]@q-d_arr[i] for i in range(n))
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()    
    q_star=np.zeros(q_len)
    for i in range(q_len):
        q_star[i]=q[i].x
    return q_star

def Integrated_prediction_gaussian_mean(x_arr,d_arr):
    n=len(d_arr)
    #transform x_arr
    M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_arr)
    theta_len = M.shape[1]
    constant = norm.ppf(b/(b+h))
    #optimize
    penalty=0.01
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    u=m.addMVar (n, ub=10000.0,lb=0.0,name="u" )
    v=m.addMVar (n, ub=10000.0,lb=0.0,name="v" )
    theta=m.addMVar (theta_len, ub=10000.0,lb=-10000.0,name="theta" )
    ones = np.ones(n)
    identity = np.identity(n)
    m.setObjective(ones@u*b+ones@v*h+theta@theta*penalty, GRB.MINIMIZE)
    m.addConstrs(u@identity[i]>=d_arr[i]-(constant+M[i]@theta) for i in range(n))
    m.addConstrs(v@identity[i]>=constant+M[i]@theta-d_arr[i] for i in range(n))
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()    
    theta_star=np.zeros(theta_len)
    for i in range(theta_len):
        theta_star[i]=theta[i].x
    return theta_star#theta@M is prediction of gaussian mean

def Integrated_prediction_uniform_ub(x_arr,d_arr):
    n=len(d_arr)
    #transform x_arr
    M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_arr)
    theta_len = M.shape[1]
    constant = b/(b+h)
    #optimize
    penalty=0.01
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    u=m.addMVar (n, ub=10000.0,lb=0.0,name="u" )
    v=m.addMVar (n, ub=10000.0,lb=0.0,name="v" )
    theta=m.addMVar (theta_len, ub=10000.0,lb=-10000.0,name="theta" )
    ones = np.ones(n)
    identity = np.identity(n)
    m.setObjective(ones@u*b+ones@v*h+theta@theta*penalty, GRB.MINIMIZE)
    m.addConstrs(u@identity[i]>=d_arr[i]-(M[i]@theta*constant) for i in range(n))
    m.addConstrs(v@identity[i]>=M[i]@theta*constant-d_arr[i] for i in range(n))
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
    #m.params.NonConvex=2
    m.update()
    m.optimize()    
    theta_star=np.zeros(theta_len)
    for i in range(theta_len):
        theta_star[i]=theta[i].x
    return theta_star#theta@M is prediction of uniform ub


def estimate_uniform_mle(x_arr,d_arr,deg_uniform_mle):
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
    w_arr = np.ones(n)*norm.ppf(ratio)+mean_arr
    return w_arr

def loss_f(w_arr,d_arr):
    loss_arr = h*np.maximum(0,w_arr-d_arr)+b*np.maximum(0,d_arr-w_arr)
    return np.average(loss_arr)

def ERM_decisions(x_train,d_train,x_test):
    q_star = ERM_policy(x_train,d_train)
    M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_test)
    w_ERM = M@q_star
    return w_ERM 

def integrated_decisions(x_train,d_train,x_test,dist='gaussian'):
    if dist=='gaussian':
        theta = Integrated_prediction_gaussian_mean(x_train,d_train)
        M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_test)
        mean_arr = M@theta#estimate of gaussian mean
        w_PTO = optimal_policy_gaussian(mean_arr,np.ones(x_test.shape[0]))
        return w_PTO
    else:#uniform
        theta = Integrated_prediction_uniform_ub(x_train,d_train)
        M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_test)
        ub_arr = M@theta;#estimate of uniform ub
        lb_arr = np.zeros(x_test.shape[0])
        w_PTO = optimal_policy_uniform(ub_arr,lb_arr)
        return w_PTO

def PTO_decisions(x_train,d_train,x_test,dist='gaussian'):
    if dist=='gaussian':
        #fit d as a function of x
        M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_train)
        model = LassoLarsCV(fit_intercept=False).fit(M, d_train)
        HT,alpha=model.coef_,model.alpha_    
        M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_test)
        mean_arr = M@HT#predicted mean of gaussian 
        w_PTO = optimal_policy_gaussian(mean_arr,np.ones(x_test.shape[0]))
        return w_PTO
    else:#uniform
        theta = estimate_uniform_mle(x_train,d_train,deg)
        M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(x_test)
        ub_arr = M@theta; lb_arr = np.zeros(x_test.shape[0])
        w_PTO = optimal_policy_uniform(ub_arr,lb_arr)
        return w_PTO

def PTO_decisions_over_parametrize(x_train,d_train,x_test,dist='gaussian'):
    deg_over_para = 10
    if dist=='gaussian':
        #fit d as a function of x
        M=PolynomialFeatures(degree=deg_over_para, include_bias=True).fit_transform(x_train)
        model = LassoLarsCV(fit_intercept=False).fit(M, d_train)
        HT,alpha=model.coef_,model.alpha_    
        M=PolynomialFeatures(degree=deg_over_para, include_bias=True).fit_transform(x_test)
        mean_arr = M@HT#predicted mean of gaussian 
        w_PTO = optimal_policy_gaussian(mean_arr,np.ones(x_test.shape[0]))
        return w_PTO
    else:#uniform
        theta = estimate_uniform_mle(x_train,d_train,deg_over_para)
        M=PolynomialFeatures(degree=deg_over_para, include_bias=True).fit_transform(x_test)
        ub_arr = M@theta; lb_arr = np.zeros(x_test.shape[0])
        w_PTO = optimal_policy_uniform(ub_arr,lb_arr)
        return w_PTO

def generate_data(train_size,test_size,dist='gaussian', feat_dim=2):
    if dist=='gaussian':
        x_train=np.random.uniform(0,1,size=(train_size, feat_dim))
        # mean_arr= 2 + (x_train[:,0] + x_train[:,1]) / 2
        mean_arr= 2 + (x_train @ np.ones(feat_dim)) / 2
        d_train=np.random.normal(mean_arr,np.ones(train_size))

        x_test=np.random.uniform(0,1,size=(test_size, feat_dim))
        # mean_arr=2+(x_test[:,0]+x_test[:,1])/2
        mean_arr= 2 + (x_test @ np.ones(feat_dim)) / 2
        d_test=np.random.normal(mean_arr,np.ones(test_size))

        w_oracle = optimal_policy_gaussian(mean_arr,np.ones(test_size))
        return  x_train,d_train,x_test,d_test,w_oracle
    else:# dist=='uniform':
        x_train=np.random.uniform(0,1,size=(train_size,feat_dim))
        lb = np.zeros(train_size)
        # ub = 1+(x_train[:,0]+x_train[:,1])/2
        ub = 1 + (x_train @ np.ones(feat_dim)) / 2
        d_train=np.random.uniform(lb,ub,size=train_size)

        x_test=np.random.uniform(0,1,size=(test_size,feat_dim))
        lb = np.zeros(test_size)
        # ub = 1+(x_test[:,0]+x_test[:,1])/2
        ub = 1 + (x_test @ np.ones(feat_dim)) / 2
        d_test=np.random.uniform(lb,ub,size=test_size)

        w_oracle = optimal_policy_uniform(ub,lb)
        return  x_train,d_train,x_test,d_test,w_oracle

def get_regret(train_size,test_size,data_dist='uniform',model_dist='gaussian', feat_dim=2):
    x_train,d_train,x_test,d_test,w_oracle = generate_data(train_size,test_size,dist=data_dist, feat_dim=feat_dim)
    oracle_loss = loss_f(w_oracle,d_test)

##    w_ERM = ERM_decisions(x_train,d_train,x_test)
##    ERM_regret=loss_f(w_ERM,d_test)-oracle_loss
    ERM_regret = 0

    w_PTO = PTO_decisions(x_train,d_train,x_test,dist=model_dist)
    PTO_regret=loss_f(w_PTO,d_test)-oracle_loss

    w_integrated = integrated_decisions(x_train,d_train,x_test,dist=model_dist)
    integrated_regret = loss_f(w_integrated,d_test)-oracle_loss

    PTO_over_param_regret=0#dummy value
    # if over_param==1:
    #     w_PTO_over_param = PTO_decisions_over_parametrize(x_train,d_train,x_test,dist=model_dist)
    #     PTO_over_param_regret=loss_f(w_PTO_over_param,d_test)-oracle_loss
    return ERM_regret,PTO_regret,integrated_regret,PTO_over_param_regret

def get_regret_arr(data_dist,model_dist,n_seed=5):
    #return array of regrets
    t1=time.time()
    erm_arr = np.zeros((len(feat_dim_lst),n_seed))
    pto_arr = np.zeros((len(feat_dim_lst),n_seed))
    integ_arr = np.zeros((len(feat_dim_lst),n_seed))
    pto_over_arr = np.zeros((len(feat_dim_lst),n_seed))
    n = 100
    for i in range(len(feat_dim_lst)):
        feat_dim = feat_dim_lst[i]
        for j in range(n_seed):
            erm_regret, pto_regret,integ_regret,pto_over_param_regret = get_regret(n,100000,data_dist,model_dist, feat_dim)
            erm_arr [i,j]+=erm_regret
            pto_arr[i,j] +=pto_regret
            integ_arr[i,j]+=integ_regret
            if j%5==0:
                print('seed: ',j,'done')
        print('n: ',n,'regret ','ERM: ',round(erm_regret,5),'PTO: ',round(pto_regret,5),
              'integrated',round(integ_regret,5))
    time_elapsed=time.time()-t1
    print('total runtime: ',round(time_elapsed))
    return erm_arr,pto_arr,integ_arr,pto_over_arr

def plot_curve(erm_arr,pto_arr,integ_arr,pto_over_arr):
    pyplot.clf()
    erm_arr=np.average(erm_arr,axis=1);pto_arr = np.average(pto_arr,axis=1);integ_arr = np.average(integ_arr,axis=1)
    if pto_over_arr!=[]:
        pto_over_arr = np.average(pto_over_arr,axis=1)
    threshold = max(max(erm_arr),max(pto_arr),max(integ_arr))
    pto_over_param_truncate = np.minimum(pto_over_arr,threshold)
    #pyplot.plot(feat_dim_lst, erm_arr,label='EO')
    pyplot.plot(feat_dim_lst, pto_arr,label='PF')
    pyplot.plot(feat_dim_lst, integ_arr,label='IO')
    title='true dist: '+data_dist+' model dist: '+model_dist
    pyplot.legend(loc="upper right")
    pyplot.title(title)
    pyplot.xlabel('training sample size');pyplot.ylabel('regret');pyplot.grid()
    pyplot.show()

def plot_histogram(n,erm_arr,pto_arr,integ_arr,pto_over_arr):
    pyplot.clf()
    #pyplot.hist(erm_arr,density=True, bins=20, alpha=0.5,label='EO')
    pyplot.hist(pto_arr,density=True, bins=20, alpha=0.5,label='PF')
    pyplot.hist(integ_arr,density=True, bins=20, alpha=0.5,label='IO')
    title='true dist: '+data_dist+' model dist: '+model_dist
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
    erm_avg = np.average(erm_arr,axis=1);pto_avg = np.average(pto_arr,axis=1);integ_avg = np.average(integ_arr,axis=1)
    erm_lower = erm_avg  - np.quantile(erm_arr, 0.25,axis=1)
    erm_upper = np.quantile(erm_arr, 0.75,axis=1) - erm_avg
    pto_lower = pto_avg - np.quantile(pto_arr, 0.25,axis=1)
    pto_upper = np.quantile(pto_arr, 0.75,axis=1) - pto_avg
    integ_lower = integ_avg - np.quantile(integ_arr, 0.25,axis=1)
    integ_upper = np.quantile(integ_arr, 0.75,axis=1) - integ_avg
    # title='true dist: '+data_dist+' model dist: '+model_dist
    title = 'dim($\\theta$) / dim($\\omega$)'
    pyplot.title(title, fontsize=label_font_size)
##    pyplot.errorbar(xi1, erm_avg, erm_std, linestyle='None', marker='o',color='r',capsize=4,
##                elinewidth=3,label='EO')
    pyplot.errorbar(xi2, pto_avg, np.vstack([pto_lower, pto_upper]), linestyle='None', marker='o',color='g',capsize=4,
                elinewidth=3,label='ETO')
    pyplot.errorbar(xi3, integ_avg, np.vstack([integ_lower, integ_upper]), linestyle='None', marker='o',color='c',capsize=4,
                elinewidth=3,label='IEO')
    pyplot.legend(loc='upper right', fontsize=label_font_size)#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    pyplot.xlabel('dim($\\theta$) / dim($\\omega$)', fontsize=label_font_size)
    pyplot.ylabel('$R(\hat{\omega})$', fontsize=label_font_size)
    pyplot.grid()
    pyplot.xticks(xi, theta_dim_lst, fontsize=tick_font_size)#, rotation=90)
    pyplot.yticks(fontsize=tick_font_size) #, rotation=90)
    #pyplot.show()
    name='true_'+data_dist+'_model_'+model_dist
    pyplot.savefig('error_bar_'+name+'_.png', bbox_inches='tight')    


label_font_size=20
tick_font_size=15
# n_list=[100,200,300,400,500]
feat_dim_lst = [1, 3, 5, 7, 9]
theta_dim_lst = (np.array(feat_dim_lst)+1).tolist()
specification = 'Misspecified' # 'Misspecified', 'Correctly specified'
if specification == 'Correctly specified':
    data_dist='gaussian'; model_dist='gaussian'
else:
    data_dist='gaussian'; model_dist='uniform'    
    
n_seed = 50 #50
erm_arr,pto_arr,integ_arr,pto_over_arr = get_regret_arr(data_dist,model_dist, n_seed)
#plot_curve(erm_arr,pto_arr,integ_arr,pto_over_arr)
plot_error_bar(erm_arr,pto_arr,integ_arr)

