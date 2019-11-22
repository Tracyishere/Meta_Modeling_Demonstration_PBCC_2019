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

def _f_S(S_0, G_in_0, k, Npatch, Nisg, Ninsulin, Disg, Rpbc):
    w1 = 0.1
    w2 = 18.0
    w3 = 0.5
    w4 = 0.5
    w5 = 0.02
    w6 = 0.5
    w7 = 1e-5
    w8 = -5.0
    return w1*S_0 + w2*G_in_0 + w3*k + w4*Npatch + w5*Nisg + w6*Ninsulin + \
    w7*Disg + w8*Rpbc


def _f_I(I_0, S_0):
    w9 = 0.5
    w10 = 25./(2. * 34.)
    return w9*I_0 + w10*S_0


@jit
def SPTModel(inputs={}, evidence={}, start={}, t=2,
             hpfn=None, name="spt"):
    """
    (SP)tio-(T)emporal model representing a simplified version of a Brownian
    Dynamics simulation in a toy model of the beta cell. Rate of secretion of
    insulin granlues (ISG)s from the cell per unit time is simply a linear
    gaussian combination of the model parameters. Hence, this is a toy
    realization of what is actually a many-body particle solver. 

    The basic equations (time discretized ODEs) are:
    S(t+1) = w1*S(t) + w2*G_in(t) + w3*k + w4*Npatch + w5*Nisg + w6*Ninsulin +
             w7*Disg + w8*Rpbc

    I(t+1) = w9*I(t) + w10*S(t)

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
    b) actin force constant (k)
    c) number of patches on insulin secretory granules (Npatch)
    d) number of insulin secretory granules (Nisg)
    e) number of insulin molecules in a granule (Ninsulin)
    f) diffusion coefficient of granules (Disg)
    e) radius of the beta cell (Rpbc)
    NOTE: If not given, arbitrary values will be assigned.

    :param evidence: dict containing evidence (specify at compile time)

    :param start: dict containing starting (t=0) values of timeseries that
    depend on their own past timeslices

    :param t: # of timeslices

    :param hpfn: json file containing dictionary of hyperpriors. Will raise
    error if not supplied

    :param name: name of model (specify at compile time, suggested: "spt")
    """

    # S CPD
    # -----------------------------------
    # input glucose profile
    # (set a dummy prior in case this is treated as an evidence)
    G_in_prior = pm.GaussianRandomWalk("G_in_prior", sigma=1.0, shape=t)
    G_in = set_input("G_in", inputs, prior=G_in_prior)

    # input actin force constant
    # (set a dummy prior in case this is treated as evidence)
    k_default = set_hp("k", name, hpfn)
    k_prior = pm.Normal("k_prior", mu=k_default,
                        sigma=set_default_sigma(k_default))
    k = set_input("k", inputs, prior=k_prior)

    # input patch density on insulin granules
    # (set a dummy prior in case this is treated as evidence)
    Npatch_default = set_hp("Npatch", name, hpfn)
    Npatch_prior = pm.Normal("Npatch_prior", mu=Npatch_default,
                             sigma=set_default_sigma(Npatch_default))
    Npatch = set_input("Npatch", inputs, prior=Npatch_prior)

    # input number of insulin granules
    # (set a dummy prior in case this is treated as evidence)
    Nisg_default = set_hp("Nisg", name, hpfn)
    Nisg_prior = pm.Normal("Nisg_prior", mu=Nisg_default,
                           sigma=set_default_sigma(Nisg_default))
    Nisg = set_input("Nisg", inputs, prior=Nisg_prior)

    # input insulin molecule density inside a granule
    # (set a dummy prior in case this is treated as evidence)
    Ninsulin_default = set_hp("Ninsulin", name, hpfn)
    Ninsulin_prior = pm.Normal("Ninsulin_prior", mu=Ninsulin_default,
                               sigma=set_default_sigma(Ninsulin_default))
    Ninsulin = set_input("Ninsulin", inputs, prior=Ninsulin_prior)

    # input granule diffusion coefficient
    # (set a dummy prior in case this is treated as evidence)
    Disg_default = set_hp("Disg", name, hpfn)
    Disg_prior = pm.Normal("Disg_prior", mu=Disg_default,
                           sigma=set_default_sigma(Disg_default))
    Disg = set_input("Disg", inputs, prior=Disg_prior)

    # input beta cell radius
    # (set a dummy prior in case this is treated as evidence)
    Rpbc_default = set_hp("Rpbc", name, hpfn)
    Rpbc_prior = pm.Normal("Rpbc_prior", mu=Rpbc_default,
                           sigma=set_default_sigma(Rpbc_default))
    Rpbc = set_input("Rpbc", inputs, prior=Rpbc_prior)

    # ATP
    # dynamic parents
    dp_S = [{"node": "me", "timeslices": 0},
            {"node": G_in, "timeslices": 0}]
    # static parents
    sp_S = [k, Npatch, Nisg, Ninsulin, Disg, Rpbc]
    sigma_S = set_hp("sigma_S", name, hpfn)
    S = GaussianTimeSeries("S", dynamic=dp_S, static=sp_S,
                             fwd_model=_f_S, sigma=sigma_S, t=t)

    # I CPD
    # -----------------------------------
    # dynamic parents (no static parents)
    dp_I = [{"node": "me", "timeslices": 0},
            {"node": S, "timeslices": 0}]
    sigma_I = set_hp("sigma_I", name, hpfn)
    I = GaussianTimeSeries("I", dynamic=dp_I, static=[],
                           fwd_model=_f_I, sigma=sigma_I, t=t)