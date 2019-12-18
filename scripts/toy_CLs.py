import os
import pickle
import glob
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

mem = {}
def get_qmu(mu, n):    
    # get global minimum
    tup = tuple((mu, n[0], n[1], n[2]))
    if tup in mem:
        return mem[tup]

    m = Minuit(likelihood, 
               mu=0, t1=0, t2=0, t3=0, 
               error_mu=0.1, error_t1=0.1, error_t2=0.1, error_t3=0.1,
               n1=n[0], n2=n[1], n3=n[2],
               fix_mu=True, fix_n1=True, fix_n2=True, fix_n3=True,
               errordef=0.5)
    fmin, params = m.migrad()
    global_nll = fmin.fval

    # get specific-mu minimum
    m = Minuit(likelihood, 
               mu=mu, t1=0, t2=0, t3=0, 
               error_t1=0.1, error_t2=0.1, error_t3=0.1,
               n1=n[0], n2=n[1], n3=n[2],
               fix_mu=True, fix_n1=True, fix_n2=True, fix_n3=True,
               errordef=0.5)
    fmin, params = m.migrad()
    mu_nll = fmin.fval

    qmu = 2*(mu_nll - global_nll)
    mem[tup] = qmu
    return qmu

nobs = [9, 5, 1]

# n = nobs
# mus = np.linspace(0.0, 2.0, 51)
# qs = []
# for mu in mus:
#     qs.append(get_qmu(mu, n))
# plt.plot(mus, qs, '-')



MU = 1.0
sMU = str(MU).replace(".","p")
fname = "data/qmu_{0}.pkl".format(sMU)
if os.path.exists(fname):
    qmus_b, qmus_sb = pickle.load(open(fname))
else:
    qmus_b, qmus_sb = [], []
for i in range(100):
    n = []
    for j in range(len(B)):
        n.append(np.random.poisson(B[j]*(1.0+unc_b[j])**np.random.normal()))
    qmus_b.append(get_qmu(MU, n))
    n = []
    for j in range(len(B)):
        n.append(np.random.poisson(B[j]*(1.0+unc_b[j])**np.random.normal() + MU*S[j]))
    qmus_sb.append(get_qmu(MU, n))
pickle.dump((qmus_b, qmus_sb), open(fname, 'w'))
qobs = get_qmu(MU, nobs)
print qobs
plt.hist(qmus_b, histtype='step', bins=80, range=(-15,15), label="B-only", density=True)
plt.hist(qmus_sb, histtype='step', bins=80, range=(-15,15), label="S+B", density=True)
plt.plot([qobs]*2, [0.0, 0.18], 'k--')
plt.text(qobs-2, 0.185, "$q_{{\mu}}^{{obs}} = {0:.2f}$".format(qobs))
plt.gca().set_ylim(0.0, 0.20)
plt.legend(fontsize='large')
plt.xlabel("$q_{\mu=1}$")
plt.savefig("../figs/results/qmu_dist.pdf")

# qs_sb = {}
# qs_b = {}
# fs = glob.glob("data/*.pkl")
# for f in fs:
#     mu = float(f.split("_")[-1].split(".")[0].replace("p","."))
#     q_b, q_sb = pickle.load(open(f))
#     qs_b[mu] = np.array(q_b)
#     qs_sb[mu] = np.array(q_sb)
# sorted_mus = sorted(qs_b.keys())
# cls = []
# for mu in sorted_mus:
#     qobs = get_qmu(mu, nobs)
#     clsb = float(np.sum(qs_sb[mu] >= qobs)) / qs_sb[mu].size
#     clb = float(np.sum(qs_b[mu] >= qobs)) / qs_b[mu].size
#     cls.append(clsb/clb)
# alpha = 0.05
# ilim = np.argmax(np.array(cls) < alpha)
# lim = sorted_mus[ilim-1] + (sorted_mus[ilim]-sorted_mus[ilim-1])/(cls[ilim]-cls[ilim-1])*(alpha-cls[ilim-1])
# print lim
# plt.plot(sorted_mus, cls, '-')
# plt.plot([sorted_mus[0], lim], [alpha]*2, 'k--')
# plt.plot([lim]*2, [0,alpha], 'k--')
# plt.gca().set_xlim(sorted_mus[0], sorted_mus[-1])
# plt.gca().set_ylim(0.0, 0.30)
# plt.xlabel('$\mu$')
# plt.ylabel('CL$_{S}(\mu)$')
# plt.text(lim, 0.08, "$\mu_{{95\%}} = {0:.2f}$".format(lim), fontsize='x-large')
# plt.savefig("../figs/results/cls.pdf")

plt.show()
