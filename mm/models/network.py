import sys
import pymc3 as pm

from mm.core import jit, GaussianTimeSeries
from mm.utils import set_hp, set_input, set_default_sigma

# Time evolution functions
# -----------------------------------
# These are the "forward models" for the corresponding GaussianTimeSeries CPDs
# The weights (w_i)s are numbered exactly as defined in the doc-string of the
# model builder function.

# In these functions, x_0 and x_1 respectively refer to previous and present
# timeslice values of x, i.e. x[t-1] and x[t]. The returned value is always
# for the present timeslice.

def _f_ATP(ATP_0, G_in_0, PFK_activity):
    w1 = (1.65/3.3)
    w2 = 1.0
    return w1*ATP_0 + w2*PFK_activity*G_in_0


def _f_GLP1(GLP1_0):
    w3 = 1.0
    return w3*GLP1_0


def _f_GLP1R(GLP1_ext_1, GLP1_1, GLP1_activity):
    w4 = 0.5
    w5 = 0.5
    return w4*GLP1_ext_1 + (w5/GLP1_activity)*GLP1_1


def _f_cAMP(cAMP_0, ATP_0, GLP1R_0):
    w6 = 1/3.
    w7 = (1/3.) * 1e-3
    w8 = 0.00013
    return w6*cAMP_0 + w7*ATP_0 + w8*GLP1R_0


def _f_Ca(Ca_0, cAMP_0):
    w9 = 0.5
    w10 = 0.05/1.3
    return w9*Ca_0 + w10*cAMP_0


def _f_S(Ca_1):
    w11 = 1.0
    return w11*Ca_1


def _f_I(I_0, S_0):
    w12 = 0.5
    w13 = 25./(2. * 34.)
    return w12*I_0 + w13*S_0


@jit
def NetworkModel(inputs={}, evidence={}, start={}, t=2,
                 hpfn=None, name="net"):
    """
    Simple network model that coarse-grains the glycolytic, mitochondrial
    and GLP-1 mediated cAMP pathways to produce a feedback-less network
    connecting input cellular glucose input to insulin secretion. This model
    is a coarse-grained version of the INSULIN SECRETION KEGG pathway
    (https://www.kegg.jp/kegg-bin/highlight_pathway?scale=1.0&map=map04911&keyword=insulin)

    The basic equations (time discretized ODEs) are:
    ATP(t+1) = w1*ATP(t) + w2*PFK_activity*G_in(t)
    GLP1(t+1) = w3*GLP1(t)
    GLP1R(t) = w4*GLP1_ext(t) + (w5/GLP1_activity) * GLP1(t)
    cAMP(t+1) = w6*cAMP(t) + w7*ATP(t) + w8*GLP1R(t)
    Ca(t+1) = w9*Ca(t) + w10*cAMP(t)
    S(t) = w11*Ca(t)
    I(t+1) = w12*I(t) + w13*S(t)

    where, w's are weights that are supplied by the user. In principle these
    should be variable and amenable to Bayesian estimation, but for now they
    are constant (i.e. "baked" into the model definition)

    Random variable (RV)s that depend on previous timeslices are represented
    using the custom GaussianTimeSeries CPD while those without such
    dependencies can simply use the usual PyMC3 RV algebra. All functions
    governing time evolution of RVs are given as private functions of the form
    _f_X() where X = RV in question.

    :param inputs: dictionary containing possible inputs to the system
    This should at least contain:
    a) input glucose conc. (G_in)
    b) GLP1R conc. (GLP1R_ext)
    c) PFK_activity
    d) GLP1_activity
    NOTE: If not given, arbitrary values will be assigned.

    :param evidence: dict containing evidence (specify at compile time)

    :param start: dict containing starting (t=0) values of timeseries that
    depend on their own past timeslices

    :param t: # of timeslices

    :param hpfn: json file containing dictionary of hyperpriors. Will raise
    error if not supplied

    :param name: name of model (specify at compile time, suggested: "net")
    """

    # ATP CPD
    # -----------------------------------
    # input glucose profile
    # (set a dummy prior in case this is treated as an evidence)
    G_in_prior = pm.GaussianRandomWalk("G_in_prior", sigma=1.0, shape=t)
    G_in = set_input("G_in", inputs, prior=G_in_prior)

    # input PFK activity
    # (set a dummy prior in case this is treated as evidence)
    PFK_activity_default = set_hp("PFK_activity", name, hpfn)
    PFK_activity_prior = pm.Normal("PFK_activity_prior",
                                   mu=PFK_activity_default,
                                   sigma=set_default_sigma(
                                       PFK_activity_default))
    PFK_activity = set_input("PFK_activity", inputs, prior=PFK_activity_prior)

    # ATP
    # dynamic parents
    dp_ATP = [{"node": "me", "timeslices": 0},
              {"node": G_in, "timeslices": 0}]
    # static parents
    sp_ATP = PFK_activity
    sigma_ATP = set_hp("sigma_ATP", name, hpfn)
    ATP = GaussianTimeSeries("ATP", dynamic=dp_ATP, static=sp_ATP,
                             fwd_model=_f_ATP, sigma=sigma_ATP, t=t)

    # GLP1 CPD
    # ---------------------------------------
    # dynamic parents (no static parents)
    dp_GLP1 = {"node": "me", "timeslices": 0}
    sigma_GLP1 = set_hp("sigma_GLP1", name, hpfn)
    GLP1 = GaussianTimeSeries("GLP1", dynamic=dp_GLP1, static=[],
                              fwd_model=_f_GLP1, sigma=sigma_GLP1, t=t)

    # GLP1R CPD
    # ----------------------------------------
    # input external GLP1R profile
    # (set a dummy prior in case this is treated as an evidence)
    GLP1R_ext_prior = pm.GaussianRandomWalk("GLP1R_ext_prior", sigma=1.0,
                                            shape=t)
    GLP1R_ext = set_input("GLP1R_ext", inputs, prior=GLP1R_ext_prior)

    # input GLP1 activity
    # (set a dummy prior in this case this is treated as evidence)
    GLP1_activity_default = set_hp("GLP1_activity", name, hpfn)
    GLP1_activity_prior = pm.Normal("GLP1_activity_prior",
                                   mu=GLP1_activity_default,
                                   sigma=set_default_sigma(
                                       GLP1_activity_default))
    GLP1_activity = set_input("GLP1_activity", inputs,
                              prior=GLP1_activity_prior)

    # GLP1R
    # (use standard PyMC3 RV algebra since no dependence on past timesteps)
    sigma_GLP1R = set_hp("sigma_GLP1R", name, hpfn)
    GLP1R = pm.Normal("GLP1R", mu=_f_GLP1R(GLP1_ext, GLP1, GLP1_activity),
                      sigma=sigma_GLP1R)

    # cAMP CPD
    # -----------------------------------
    # dynamic parents (no static parents)
    dp_cAMP = [{"node": "me", "timeslices": 0},
               {"node": ATP, "timeslices": 0},
               {"node": GLP1R, "timeslices": 0}]
    sigma_cAMP = set_hp("sigma_cAMP", name, hpfn)
    cAMP = GaussianTimeSeries("cAMP", dynamic=dp_cAMP, static=[],
                              fwd_model=_f_cAMP, sigma=sigma_cAMP, t=t)

    # Ca CPD
    # -----------------------------------
    # dynamic parents (no static parents)
    dp_Ca = [{"node": "me", "timeslices": 0},
             {"node": cAMP, "timeslices": 0}]
    sigma_Ca = set_hp("sigma_Ca", name, hpfn)
    Ca = GaussianTimeSeries("Ca", dynamic=dp_Ca, static=[],
                            fwd_model=_f_Ca, sigma=sigma_Ca, t=t)

    # S CPD
    # -----------------------------------
    # (use standard PyMC3 RV algebra since no dependence on past timesteps)
    sigma_S = set_hp("sigma_S", name, hpfn)
    S = pm.Normal("S", mu=_f_S(Ca), sigma=sigma_S)

    # I CPD
    # -----------------------------------
    # dynamic parents (no static parents)
    dp_I = [{"node": "me", "timeslices": 0},
            {"node": S, "timeslices": 0}]
    sigma_I = set_hp("sigma_I", name, hpfn)
    I = GaussianTimeSeries("I", dynamic=dp_I, static=[],
                           fwd_model=_f_I, sigma=sigma_I, t=t)
