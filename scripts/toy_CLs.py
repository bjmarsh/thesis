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

nobs = [7, 6, 1]

# n = nobs
# mus = np.linspace(0.0, 2.0, 51)
# qs = []
# for mu in mus:
#     qs.append(get_qmu(mu, n))
# plt.plot(mus, qs, '-')



# MU = 1.0
# sMU = str(MU).replace(".","p")
# fname = "data/qmu_{0}.pkl".format(sMU)
# if os.path.exists(fname):
#     qmus_b, qmus_sb = pickle.load(open(fname))
# else:
#     qmus_b, qmus_sb = [], []
# for i in range(100):
#     n = []
#     for j in range(len(B)):
#         n.append(np.random.poisson(B[j]*(1.0+unc_b[j])**np.random.normal()))
#     qmus_b.append(get_qmu(MU, n))
#     n = []
#     for j in range(len(B)):
#         n.append(np.random.poisson(B[j]*(1.0+unc_b[j])**np.random.normal() + MU*S[j]))
#     qmus_sb.append(get_qmu(MU, n))
# pickle.dump((qmus_b, qmus_sb), open(fname, 'w'))
# bins=(80,-15,15)
# bw = float(bins[2]-bins[1])/bins[0]
# qobs = round((get_qmu(MU, nobs)-bins[1])/bw) * bw + bins[1]
# print qobs
# n, edges, _ = plt.hist(qmus_b, histtype='step', bins=bins[0], range=(bins[1],bins[2]), label="B-only", density=True, color='royalblue', lw=2)
# for i in range(len(n)):
#     if edges[i] >= qobs:
#         n[i] = 0
# plt.hist(edges[:-1], weights=n, histtype='stepfilled', bins=bins[0], range=(bins[1],bins[2]), alpha=0.4, color='royalblue', label="$1-CL_{B}$")
# n, edges, _ = plt.hist(qmus_sb, histtype='step', bins=bins[0], range=(bins[1],bins[2]), label="S+B", density=True, color='darkorange', lw=2)
# for i in range(len(n)):
#     if edges[i] < qobs:
#         n[i] = 0
# plt.hist(edges[:-1], weights=n, histtype='stepfilled', bins=bins[0], range=(bins[1],bins[2]), alpha=0.4, color='darkorange', label="$CL_{S+B}$")
# plt.plot([qobs]*2, [0.0, 0.18], 'k--')
# plt.text(qobs-2, 0.185, "$q_{{\mu}}^{{obs}} = {0:.2f}$".format(qobs))
# plt.gca().set_ylim(0.0, 0.20)
# plt.legend(fontsize='large')
# plt.xlabel("$q_{\mu=1}$")
# plt.savefig("../figs/results/qmu_dist.pdf")

qs_sb = {}
qs_b = {}
fs = glob.glob("data/*.pkl")
for f in fs:
    mu = float(f.split("_")[-1].split(".")[0].replace("p","."))
    q_b, q_sb = pickle.load(open(f))
    qs_b[mu] = np.array(q_b)
    qs_sb[mu] = np.array(q_sb)
sorted_mus = sorted(qs_b.keys())
cls = []
for mu in sorted_mus:
    qobs = get_qmu(mu, nobs)
    clsb = float(np.sum(qs_sb[mu] >= qobs)) / qs_sb[mu].size
    clb = float(np.sum(qs_b[mu] >= qobs)) / qs_b[mu].size
    cls.append(clsb/clb)
alpha = 0.05
ilim = np.argmax(np.array(cls) < alpha)
lim = sorted_mus[ilim-1] + (sorted_mus[ilim]-sorted_mus[ilim-1])/(cls[ilim]-cls[ilim-1])*(alpha-cls[ilim-1])
print lim
plt.plot(sorted_mus, cls, '-')
plt.plot([sorted_mus[0], lim], [alpha]*2, 'k--')
plt.plot([lim]*2, [0,alpha], 'k--')
plt.gca().set_xlim(sorted_mus[0], sorted_mus[-1])
plt.gca().set_ylim(0.0, 0.30)
plt.xlabel('$\mu$')
plt.ylabel('CL$_{S}(\mu)$')
plt.text(lim, 0.08, "$\mu_{{95\%}} = {0:.2f}$".format(lim), fontsize='x-large')
plt.savefig("../figs/results/cls.pdf")

plt.show()


# import ROOT as r
# r.gStyle.SetOptStat(0)

# c = r.TCanvas("c","c",800,650)
# c.SetBottomMargin(0.12)

# hb = r.TH1D("hb",";Bin Number", 3, 0.5, 3.5)
# hs = r.TH1D("hs",";Bin Number", 3, 0.5, 3.5)
# hu = r.TH1D("hu",";Bin Number", 3, 0.5, 3.5)
# hd = r.TH1D("hd",";Bin Number", 3, 0.5, 3.5)

# for i in range(len(B)):
#     hb.SetBinContent(i+1, B[i])
#     hs.SetBinContent(i+1, S[i])
#     hu.SetBinContent(i+1, B[i])
#     hu.SetBinError(i+1, B[i]*unc_b[i])
#     hd.SetBinContent(i+1, nobs[i])

# hb.SetLineColor(r.kBlack)
# hb.SetFillColor(r.kAzure+8)
# hs.SetLineColor(r.kBlack)
# hs.SetFillColor(r.kOrange)
# hu.SetFillStyle(3344)
# hu.SetFillColor(r.kGray+3)
# hd.SetMarkerStyle(20)
# hd.SetMarkerSize(1.2)
# hd.SetMarkerColor(r.kBlack)
# hd.SetLineColor(r.kBlack)
# hd.SetLineWidth(2)

# hdummy = r.TH1D("hdummy",";Bin Number",3,0.5,3.5)
# hdummy.GetXaxis().SetRangeUser(0.51,3.49)
# hdummy.GetXaxis().SetNdivisions(3)
# hdummy.GetXaxis().SetLabelSize(0.04)
# hdummy.GetXaxis().SetTitleSize(0.04)
# hdummy.GetXaxis().SetTitleOffset(1.1)
# hdummy.GetYaxis().SetRangeUser(0,15)
# hdummy.GetYaxis().SetLabelSize(0.04)
# hdummy.GetYaxis().SetTitleSize(0.04)
# hdummy.GetYaxis().SetTitleOffset(1.1)
# hdummy.GetYaxis().SetTitle("Events / bin")
# hdummy.Draw()

# hstack = r.THStack()
# hstack.Add(hb)
# hstack.Add(hs)

# hstack.Draw("SAME HIST")
# hu.Draw("SAME E2")
# hd.Draw("SAME PE X0")
# hdummy.Draw("AXIS SAME")

# leg = r.TLegend(0.45, 0.62, 0.88, 0.88)
# leg.SetLineColor(r.kWhite)
# leg.AddEntry(hd, "Data", 'pe')
# leg.AddEntry(hb, "Estimated background", 'f')
# leg.AddEntry(hu, "Background uncertainty", 'f')
# leg.AddEntry(hs, "Signal", 'f')
# leg.Draw()

# c.SaveAs("../figs/results/toy_exp.pdf")

# raw_input()


