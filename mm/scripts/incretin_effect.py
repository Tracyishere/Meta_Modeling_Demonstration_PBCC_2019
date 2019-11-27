"""
This script presents a simplified demonstration of the incretin effect,
where the concentration of plasma insulin and insulin secretion rate in a
macroscopic PK/PD model (the meal model) can be modulated by the
concentration and binding affinity of an incretin hormone (in this case a
GLP1-analog). To achieve this, 4 simplified models are coupled together:

1) A macroscopic PK/PD model of how a glucose rich meal stimulates insulin
secretion at the organ (pancreas) level, referred to in the code as the
"meal" model.

2) A toy realization of a Brownian Dynamics simulation of glucose attachment to
insulin vesicles that initiates their transport. While the actual stuff is a
many-body particle simulator, the version contained here assumes that we can
approximate the dynamics using simple linear gaussian forward models. This
model is referred to in the code as the "spt" (spatio-temporal) model.

3) A toy realization of the KEGG pathway for insulin secretion
(https://www.kegg.jp/kegg-bin/highlight_pathway?scale=1.0&map=map04911&keyword=insulin)
The toy model short circuits several steps of the pathway and captures a feed-
forward network going from glucose entry to the cell to insulin secretion,
modulated by ATP and cAMP mediated Ca2+ dynamics. It does NOT contain any
feedback loops that are present in the actual pathway for ease of computation.
It also has a simple GLP1 mediated cAMP dynamics. In the code, this model
is referred to as the "net" model.

4) A toy virtual screening model that relates the concentration and binding
affinity (which is assumed to be a Z-score that encapsulates docking scores,
binding free energy estimates etc.) of GLP1-analogs i.e. incretin hormones (
relative to that of GLP1) to the activity of the GLP1-R receptor which serves as
the link between incretins and their effect on cAMP dynamics in the PBC.
This model is referred to as the "vs" model in the code.

For more details on these models please look at the doc strings of the model
functions in the ../models subfolder.

This script DOES NOT DO POSTERIOR ESTIMATION of parameters. Rather,
it demonstrates how information flows between different input models by
sampling from the overall prior created by coupling the 4 different input
models. E.g. tuning the incretin effect through inputs like incretin
concentration and Z-score can affect the glucose and insulin dynamics at the
macro-scale in the PK/PD ("meal") model for healthy (normal) and t2d scenarios.

Author: Tanmoy Sanyal, Sali lab
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

sys.path.append("../..")
from mm.core import jit, GaussianTimeSeries
from mm.models import MealModel, SPTModel, NetworkModel, VirtualScreenModel
from mm.utils import get_distribution, set_start

# random number seed for reproducibility
RNG = 85431

# I/O locations
DataDir = os.path.abspath("./datasets")
OutDir = os.path.abspath("./output")
HP_FN = os.path.abspath("./default_params.json")
if not os.path.isdir(OutDir):
    os.mkdir(OutDir)


# helper functions to save and load traces
def _load_trace(prefix, mealmodel_type, conc, zscore):
    fn = prefix + "_%s_%1.1f_%1.1f_trace.pickle" % (mealmodel_type, conc,
                                                    zscore)
    if os.path.isfile(fn):
        with open(fn, "rb") as of:
            trace = dill.load(of)
        return trace
    else:
        return None


def _save_trace(data, prefix, mealmodel_type, conc, zscore):
    fn = prefix + "_%s_%1.1f_%1.1f_trace.pickle" % (mealmodel_type, conc,
                                                    zscore)
    with open(fn, "wb") as of:
        dill.dump(data, of)


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

    # call the vs model MBF
    vs = VirtualScreenModel(name="vs", t=t, inputs=inputs, evidence=evidence,
                            start=start, hpfn=hpfn)

    # construct the second coupler (a modified inputs for the network model)
    # intracellular glucose ~ 50% of extracellular glucose # assumption
    # GLP1R_ext = GLP1R coming from the vs model
    net_inputs = {"G_in": 0.5 * meal.G, "GLP1R_ext": vs.GLP1R}
    inputs.update(net_inputs)

    # call the network model MBF
    net = NetworkModel(name="net", t=t, inputs=inputs, evidence=evidence,
                       start=start, hpfn=hpfn)

    # combine (average) the secretions from different models
    S = pm.Normal("S", mu=(meal.S + spt.S + net.S) / 3., sigma=0.01, shape=t)

    # calculate new plasma insulin concentration from the meta S.
    # In the meal model: I(t+1) = (1-gamma*dt)*I(t) + S(t)*dt
    # gamma ~ 0.5 and dt = 2.0. Plugging in these values: I(t+1) ~ 2 *S(t)
    # This is the form we'll use here.
    I = GaussianTimeSeries("I", dynamic={"node": S, "timeslices": 0}, static=[],
                           fwd_model=lambda x: 2 * x, sigma=0.1, t=t)


#### MAIN ####

# user input
def _tupletype(s):
    return [float(i) for i in s.split(",")]


parser = argparse.ArgumentParser(description="Demonstrate the incretin effect \
                                             for a metamodel built from 4 \
                                             input models: meal, spt, net, vs")

parser.add_argument("-ns", "--nsamples", default=500, type=int,
                    help="number of samples to draw from posterior")

parser.add_argument("-c", "--conc", nargs="+", default=[0.0], type=_tupletype,
                    help="concentration of incretin hormone (GLP1-analog)")

parser.add_argument("-z", "--zscores", nargs="+", default=[0.0],
                    type=_tupletype,
                    help="List of z_scores reflecting binding affinity of \
                          GLP1-analogs relative to GLP1")

parser.add_argument("-p", "--prefix", default="mm",
                    help="prefix of all output files")

args = parser.parse_args()
NSamples = args.nsamples
analog_concs = args.conc[0]
z_scores = args.zscores[0]
prefix = os.path.join(OutDir, args.prefix)

# number of incretin hormone scenarios
assert (len(analog_concs) == len(z_scores))
n_incretins = len(analog_concs)

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

# set inputs for all models (explicitly set analog conc. to be same at all time)
inputs = {"DGintake": DGintake_data}

# set starting points for all models
start_spt = {"spt_S": set_start(rvname="S", modelname="spt", hpfn=HP_FN),
             "spt_I": set_start(rvname="I", modelname="spt", hpfn=HP_FN)}

start_net = {"net_ATP": set_start(rvname="ATP", modelname="net", hpfn=HP_FN),
             "net_GLP1": set_start(rvname="GLP1", modelname="net", hpfn=HP_FN),
             "net_cAMP": set_start(rvname="cAMP", modelname="net", hpfn=HP_FN),
             "net_Ca": set_start(rvname="Ca", modelname="net", hpfn=HP_FN),
             "net_I": set_start(rvname="I", modelname="net", hpfn=HP_FN)}

start_vs = {"vs_analog_conc": set_start(rvname="analog_conc",
                                        modelname="vs", hpfn=HP_FN),
            "vs_GLP1R": set_start(rvname="GLP1R", modelname="vs", hpfn=HP_FN)}

start_meal_normal = {"meal_normal_State": set_start(rvname="State",
                                                    modelname="meal_normal",
                                                    hpfn=HP_FN)}

start_meal_t2d = {"meal_t2d_State": set_start(rvname="State",
                                              modelname="meal_t2d",
                                              hpfn=HP_FN)}

start_meta_normal = {"meta_I": start_meal_normal["meal_normal_State"][-1]}

start_meta_t2d = {"meta_I": start_meal_t2d["meal_t2d_State"][-1]}

# combine into the over-all starting dicts for normal and t2d
start_normal = {**start_meal_normal, **start_spt, **start_net, **start_vs,
                **start_meta_normal}
start_t2d = {**start_meal_t2d, **start_spt, **start_net, **start_vs,
             **start_meta_t2d}

# ------------------
# META-MODELS
# ------------------
tracefn_normal = prefix + "_normal_trace.shelf"
tracefn_t2d = prefix + "_t2d_trace.shelf"
tracedict_normal = {}  # for plotting later
tracedict_t2d = {}  # for plotting later

### for each incretin hormone scenario
for n in range(n_incretins):
    analog_conc = analog_concs[n]
    z_score = z_scores[n]
    key = ("%1.1f, %1.1f" % (analog_conc, z_score))

    print("\nANALOG CONC = %1.1f nM, Z_SCORE = %1.1f" % (analog_conc, z_score))
    print("-----------------------------------------")

    # NORMAL CASE
    data_normal = _load_trace(prefix=prefix, mealmodel_type="normal",
                              conc=analog_conc, zscore=z_score)
    if data_normal is None:
        # compile model and run training
        this_inputs = {**inputs,
                       "analog_conc": analog_conc * np.ones(len(tdata)),
                       "analog_z_score": z_score}

        print(">Compiling meta-model for normal case")
        mm_normal = MetaModel(name="meta", mealmodel_type="normal",
                              inputs=this_inputs,
                              start=start_normal,
                              t=len(tdata), hpfn=HP_FN)

        print(">Sampling from meta-model prior for normal case...")
        trace_normal = pm.sample_prior_predictive(model=mm_normal,
                                                  samples=NSamples,
                                                  random_seed=RNG)
        _save_trace(data=trace_normal,
                    prefix=prefix, mealmodel_type="normal",
                    conc=analog_conc, zscore=z_score)
        tracedict_normal[key] = trace_normal
    else:
        print(">Loading saved trace for normal case...")
        trace_normal = data_normal[-1]
        tracedict_normal[key] = trace_normal

    # T2D CASE
    data_t2d = _load_trace(prefix=prefix, mealmodel_type="t2d",
                           conc=analog_conc, zscore=z_score)
    if data_t2d is None:
        # compile model and run training
        this_inputs = {**inputs,
                       "analog_conc": analog_conc * np.ones(len(tdata)),
                       "analog_z_score": z_score}

        print(">Compiling meta-model for t2d case")
        mm_t2d = MetaModel(name="meta", mealmodel_type="t2d",
                           inputs=this_inputs,
                           start=start_t2d,
                           t=len(tdata), hpfn=HP_FN)
        print(">Sampling from meta-model prior for t2d case...")
        trace_t2d = pm.sample_prior_predictive(model=mm_t2d,
                                               draws=NSamples,
                                               random_seed=RNG)
        _save_trace(data=trace_t2d,
                    prefix=prefix, mealmodel_type="t2d",
                    conc=analog_conc, zscore=z_score)
        tracedict_t2d[key] = trace_t2d
    else:
        print(">Loading saved trace for t2d case...")
        trace_t2d = data_t2d[-1]
        tracedict_t2d[key] = trace_t2d

# -----------
# PLOT STUFF
# -----------
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

for ax in [ax1, ax2, ax3, ax4]:
    if ax == ax1:
        ax.set_title("normal", fontsize=15)
    if ax == ax2:
        ax.set_title("t2d", fontsize=15)
    if ax in [ax1, ax2]:
        ax.set_ylabel("plasma glucose (mM)", fontsize=15)
    if ax in [ax3, ax4]:
        ax.set_ylabel("plasma insulin (mM)", fontsize=15)
    if ax in [ax3, ax4]:
        ax.set_xlabel('time (mins)', fontsize=15)
    else:
        ax.set_xticklabels([])

alphas = np.linspace(0.4, 1.0, n_incretins)
for n in range(n_incretins):
    analog_conc = analog_concs[n]
    z_score = z_scores[n]
    key = ("%1.1f, %1.1f" % (analog_conc, z_score))

    # normal case
    G, Gerr = _get_posterior(tracedict_normal[key], "meal_normal_G",
                             vartype="dynamic")
    I, Ierr = _get_posterior(tracedict_normal[key], "meta_I", vartype="dynamic")
    ax1.plot(tdata, G, "r-", alpha=alphas[n])
    ax3.plot(tdata, I, "g-", alpha=alphas[n])

    # t2d case
    G, Gerr = _get_posterior(tracedict_t2d[key], "meal_t2d_G",
                             vartype="dynamic")
    I, Ierr = _get_posterior(tracedict_t2d[key], "meta_I", vartype="dynamic")
    ax2.plot(tdata, G, "r-", alpha=alphas[n])
    ax4.plot(tdata, I, "g-", alpha=alphas[n])

figname = prefix + "_incretin_effect.png"
fig.tight_layout()
plt.show()
fig.savefig(figname, bbox_inches="tight")
