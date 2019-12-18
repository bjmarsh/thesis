import os
import pickle
import numpy as np
import scipy.stats as stats
from iminuit import Minuit
import matplotlib.pyplot as plt

B = [10, 5, 2]
S = [2, 2, 2]

unc_b = [0.20, 0.20, 0.20]

def likelihood(mu, t1, t2, t3, n1, n2, n3):
    nll = 0.0
    
    t = [t1,t2,t3]
    n = [n1,n2,n3]

    for i in range(len(B)):
        b = B[i]*(1.0+unc_b[i])**t[i]
        nll -= stats.poisson.logpmf(n[i], mu*S[i] + b)
        nll -= stats.norm.logpdf(t[i], 0, 1)

    return nll

n = [15,4,2]

def get_qmu(mu, n):
    
    # get global minimum
    m = Minuit(likelihood, 
               mu=0, t1=0, t2=0, t3=0, 
               error_mu=0.1, error_t1=0.1, error_t2=0.1, error_t3=0.1,
               n1=n[0], n2=n[1], n3=n[2],
               fix_n1=True, fix_n2=True, fix_n3=True,
               errordef=0.5)
    fmin, params = m.migrad()
    global_nll = fmin.fval

    # get specific-mu minimum
    m = Minuit(likelihood, 
               mu=mu, t1=0, t2=0, t3=0, 
               error_t1=0.1, error_t2=0.1, error_t3=0.1,
               n1=n[0], n2=n[1], n3=n[2],
               fix_mu = True, fix_n1=True, fix_n2=True, fix_n3=True,
               errordef=0.5)
    fmin, params = m.migrad()
    mu_nll = fmin.fval
    
    return 2*(mu_nll - global_nll)

MU = 2.0
sMU = str(MU).replace(".","p")
fname = "data/qmu_{0}.pkl".format(sMU)
if os.path.exists(fname):
    qmus_b, qmus_sb = pickle.load(open(fname))
else:
    qmus_b, qmus_sb = [], []
for i in range(1000):
    n = []
    for j in range(len(B)):
        n.append(np.random.poisson(B[j]*(1.0+unc_b[j])**np.random.normal()))
    qmus_b.append(get_qmu(MU, n))

    n = []
    for j in range(len(B)):
        n.append(np.random.poisson(B[j]*(1.0+unc_b[j])**np.random.normal() + MU*S[j]))
    qmus_sb.append(get_qmu(MU, n))

pickle.dump((qmus_b, qmus_sb), open(fname, 'w'))

plt.hist(qmus_b, histtype='step', bins=40, range=(0,8), label="B-only")
plt.hist(qmus_sb, histtype='step', bins=40, range=(0,8), label="S+B")
plt.legend()
plt.show()
