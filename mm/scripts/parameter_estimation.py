"""
This script provides a demonstration of estimating and refining parameters
from input models through probabilistic meta-modeling. To keep things simple,
only two input models are considered here:

1) A macroscopic PK/PD model of how a glucose rich meal stimulates insulin
secretion at the organ (pancreas) level, referred to in the code as the
"meal model".

2) A toy realization of a Brownian Dynamics simulation of glucose attachment to
insulin vesicles that initiates their transport. While the actual stuff is a
many-body particle simulator, the version contained here assumes that we can
approximate the dynamics using simple linear gaussian forward models. This
model is referred to in the code as the "spt" (spatio-temporal) model.

For more details on these models please look at the doc strings of the model
functions in the ../models subfolder.

This script compares prior and posterior estimates of parameters for
normal (healthy) and t2d (type-2 diabetic) cases.

USAGE: python parameter_estimation.py -h will show you the different arguments
that the script needs.

Author :Tanmoy Sanyal, Sali lab
"""

import os
import sys
import numpy as np
import dill
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence

sys.path.append("../..")
from mm.core import jit, GaussianTimeSeries
from mm.models import MealModel, SPTModel
from mm.utils import get_distribution, set_start


# helper function to generate a proper variable name
# from user supplied var tuple
def _get_rvname(varname, modelname, case="normal"):
    if modelname == "meal":
        modelname += "_" + case
    return modelname + "_" + varname


# helper function to generate prior distributions
def _get_prior(model, varname, samples=100, **kwargs):
    rvlist = model.basic_RVs
    rv_ = [x for x in rvlist if x.name == rvname]
    if not rv_:
        raise ValueError("RV %s not found in model %s" % (varname, model.name))
    rv = rv_[0]
    if isinstance(rv.distribution, GaussianTimeSeries):
        prior_samples = rv.distribution.random(size=samples, quiet=False)
    else:
        prior_samples = rv.distribution.random(size=samples)
    prior = get_distribution(prior_samples, **kwargs)
    return prior


# helper function to collect generate posterior distributions
def _get_posterior(trace, varname, **kwargs):
    if not varname in trace.varnames:
        raise ValueError("RV %s not found in model trace" % varname)
    post_samples = trace[varname]
    post = get_distribution(post_samples, **kwargs)
    return post


# MBF for the meta-model
@jit
def MetaModel(name="meta", t=100,
              inputs={}, evidence={}, start={},
              hpfn=None, mealmodel_type="normal"):
    # call the meal model MBF
    mealmodel_name = "meal_%s" % mealmodel_type  # normal or t2d
    meal = MealModel(name=mealmodel_name, t=t, inputs=inputs, evidence=evidence,
                     start=start, hpfn=hpfn)

    # construct the first coupler (a modified inputs for the spt model)
    # intracellular glucose ~ 50% of extracellular glucose # assumption
    spt_inputs = {"G_in": 0.5 * meal.G}
    inputs.update(spt_inputs)

    # call the spt model MBF and pass in the first coupler
    spt = SPTModel(name="spt", t=t, inputs=inputs, evidence=evidence,
                   start=start, hpfn=hpfn)

    # combine (average) the plasma insulin (I) from meal and spt models
    I = pm.Normal("I", mu=(meal.I + spt.I) / 2., sigma=0.01)


#### MAIN ####

# I/O locations
DataDir = os.path.abspath("./datasets")
OutDir = os.path.abspath("./output")
HP_FN = os.path.abspath("./default_params.json")
if not os.path.isdir(OutDir):
    os.mkdir(OutDir)

# user input
def _tupletype(s):
    return s.split(",")

parser = argparse.ArgumentParser(description="refine/estimate parameters from \
                            a meta-model built from the meal and spt models")

parser.add_argument("-nb", "--nburn", default=50000, type=int,
                    help="Number of tuning steps for variational inference")

parser.add_argument("-ns", "--nsamples", default=500, type=int,
                    help="number of samples to draw from posterior")

parser.add_argument("-v", "--vars", nargs="+", default=[], type=_tupletype,
                    help="list with tuples of the form \
                          (varname, modelname, vartype) where \
                          modelname = 'meal' or 'spt' or 'meta' and \
                          vartype = 'static' or 'dynamic' ")

parser.add_argument("-p", "--prefix", default="mm",
                    help="prefix of all output files")

args = parser.parse_args()
NBurn = args.nburn
NSamples = args.nsamples
vartuples = [tuple(i) for i in args.vars]
prefix = os.path.join(OutDir, args.prefix)

# timestep (for the meal model)
dt = 2.0  # mins

# extract ingested glucose profile data
input_data = np.loadtxt(os.path.join(DataDir, "Gintake.dat"))
tdata = dt * input_data[:, 0]
Gintake_data = dt * input_data[:, 2]
DGintake_data = dt * input_data[:, 1]

# extract plasma insulin data for normal and t2d cases
normal_data = np.loadtxt(os.path.join(DataDir, "glucose_insulin_normal.dat"))
t2d_data = np.loadtxt(os.path.join(DataDir, "glucose_insulin_t2d.dat"))
Idata_normal = normal_data[:, 2]
Idata_t2d = t2d_data[:, 2]

# get starting points for markovian variables
start_normal = {"meal_normal_State": set_start(rvname="State",
                                               modelname="meal_normal",
                                               hpfn=HP_FN),
                "spt_S": set_start(rvname="S", modelname="spt", hpfn=HP_FN),
                "spt_I": set_start(rvname="I", modelname="spt", hpfn=HP_FN)}

start_t2d = {"meal_t2d_State": set_start(rvname="State",
                                         modelname="meal_t2d", hpfn=HP_FN),
             "spt_S": set_start(rvname="S", modelname="spt", hpfn=HP_FN),
             "spt_I": set_start(rvname="I", modelname="spt", hpfn=HP_FN)}

# ------------------
# NORMAL META-MODEL
# ------------------
tracefn_normal = prefix + "_normal_trace.pickle"

# 1. Build metamodel
print("\nCompiling meta-model for normal case")
mm_normal = MetaModel(name="meta", mealmodel_type="normal",
                      inputs={"DGintake": DGintake_data},
                      evidence={"meta_I": Idata_normal},
                      start=start_normal,
                      t=len(tdata), hpfn=HP_FN)

# 2. Generate prior distributions
prior_normal = {}
print("Generating prior distributions...")
for varname, modelname, vartype in vartuples:
    rvname = _get_rvname(varname, modelname, "normal")
    prior_normal[rvname] = _get_prior(mm_normal, rvname, samples=NSamples,
                                      vartype=vartype, smooth=True)

# 3. Train
if not os.path.isfile(tracefn_normal):
    print("Training...")
    approx_normal = pm.fit(model=mm_normal, method="advi", n=NBurn,
                           callbacks=[CheckParametersConvergence()])
    trace_normal = approx_normal.sample(draws=NSamples)
    with open(tracefn_normal, "wb") as of:
        dill.dump((approx_normal, trace_normal), of)
else:
    print("Loading saved trace...")
    with open(tracefn_normal, "rb") as of:
        approx_normal, trace_normal = dill.load(of)

# 4. Generate posterior distributions
post_normal = {}
for varname, modelname, vartype in vartuples:
    rvname = _get_rvname(varname, modelname, "normal")
    post_normal[rvname] = _get_posterior(trace_normal, rvname, vartype=vartype,
                                         smooth=True)

# ------------------
# T2D META-MODEL
# ------------------
tracefn_t2d = prefix + "_t2d_trace.pickle"

# 1. Build metamodel
print("\nCompiling meta-model for t2d case")
mm_t2d = MetaModel(name="meta", mealmodel_type="t2d",
                   inputs={"DGintake": DGintake_data},
                   evidence={"meta_I": Idata_t2d},
                   start=start_t2d,
                   t=len(tdata), hpfn=HP_FN)

# 2. Generate prior distributions
prior_t2d = {}
print("Generating prior distributions...")
for varname, modelname, vartype in vartuples:
    rvname = _get_rvname(varname, modelname, "t2d")
    prior_t2d[rvname] = _get_prior(mm_t2d, rvname, samples=NSamples,
                                   vartype=vartype, smooth=True)

# 3. Train
if not os.path.isfile(tracefn_t2d):
    print("Training...")
    approx_t2d = pm.fit(model=mm_t2d, method="advi", n=NBurn,
                        callbacks=[CheckParametersConvergence()])
    trace_t2d = approx_t2d.sample(draws=NSamples)
    with open(tracefn_t2d, "wb") as of:
        dill.dump((approx_t2d, trace_t2d), of)
else:
    print("Loading saved trace...")
    with open(tracefn_t2d, "rb") as of:
        approx_t2d, trace_t2d = dill.load(of)

# 4. Generate posterior distributions
post_t2d = {}
for varname, modelname, vartype in vartuples:
    rvname = _get_rvname(varname, modelname, "t2d")
    post_t2d[rvname] = _get_posterior(trace_t2d, rvname, vartype=vartype,
                                      smooth=True)

# -----------
# PLOT STUFF
# -----------
for varname, modelname, vartype in vartuples:
    rv1 = _get_rvname(varname, modelname, "normal")
    prior1 = prior_normal[rv1]
    post1 = post_normal[rv1]

    rv2 = _get_rvname(varname, modelname, "t2d")
    prior2 = prior_t2d[rv2]
    post2 = post_t2d[rv2]

    figname = "%s_%s_%s" % (prefix, modelname, varname)

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title("normal", fontsize=15)
    ax2.set_title("t2d", fontsize=15)

    # static variables
    if vartype == "static":
        ax1.plot(prior1[0], prior1[1], "g--", lw=4, alpha=0.5, label="prior")
        ax1.plot(post1[0], post1[1], "g-", lw=4, label="posterior")
        ax1.set_xlabel(varname, fontsize=15)
        ax1.set_ylabel("distribution", fontsize=15)
        ax1.legend(prop={"size": 12})

        ax2.plot(prior2[0], prior2[1], "r--", lw=4, alpha=0.5, label="prior")
        ax2.plot(post2[0], post2[1], "r-", lw=4, label="posterior")
        ax2.set_xlabel(varname, fontsize=15)
        ax2.set_ylabel("distribution", fontsize=15)
        ax2.legend(prop={"size": 12})

    # dynamic variables
    if vartype == "dynamic":
        ax1.plot(tdata, prior1[0], "g--", lw=2, alpha=0.4, label="prior")
        ax1.fill_between(tdata, prior1[0] - prior1[1], prior1[0] + prior1[1],
                         color="y", alpha=0.2)
        ax1.plot(tdata, post1[0], "g-", lw=2, label="posterior")
        ax1.fill_between(tdata, post1[0] - post1[1], post1[0] + post1[1],
                         color="g", alpha=0.5)
        ax1.set_xlabel("time (mins)", fontsize=15)
        ax1.set_xlabel(varname, fontsize=15)
        ax1.legend(prop={"size": 12})

        ax2.plot(tdata, prior2[0], "r--", lw=2, alpha=0.4, label="prior")
        ax2.fill_between(tdata, prior2[0] - prior2[1], prior2[0] + prior2[1],
                         color="salmon", alpha=0.2)
        ax2.plot(tdata, post2[0], "r-", lw=2, label="posterior")
        ax2.fill_between(tdata, post2[0] - post2[1], post2[0] + post2[1],
                         color="r", alpha=0.5)
        ax2.set_xlabel("time (mins)", fontsize=15)
        ax2.set_xlabel(varname, fontsize=15)
        ax2.legend(prop={"size": 12})

    fig.tight_layout()
    fig.savefig(figname, bbox_inches="tight")
