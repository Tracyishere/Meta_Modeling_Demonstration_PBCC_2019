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

def _f_analog_conc(analog_conc_0):
    w1 = 1.0
    return w1*analog_conc_0

def _f_GLP1R(analog_conc_1, analog_z_score):
    w2 = 1.2
    w3 = 1.2
    return w2*analog_conc_1 + w3*analog_z_score


@jit
def VirtualScreenModel(inputs={}, evidence={}, start={}, t=2,
                       hpfn=None, name="vs"):
    """
    Simple linear regression model for GLP1R concentration dependent on
    binding affinity to GLP1 analogs and the analog concentration. The binding
    affinity is assumed to be presented as a z_score derived from docking and
    free energy calculations.

    The basic equation (time discretized ODEs) is:
    conc(t+1) = w1*analog_conc(t)
    GLP1R(t) = w2*analog_z_score + w3*analog_conc(t)

    where, w's are weights that are supplied by the user. In principle these
    should be variable and amenable to Bayesian estimation, but for now they
    are constant (i.e. "baked" into the model definition)

    Random variable (RV)s that depend on previous timeslices are represented
    using the custom GaussianTimeSeries CPD while those without such
    dependencies can simply use the usual PyMC3 RV algebra. All functions
    governing time evolution of RVs are given as private functions of the form
    _f_X() where X = RV in question.

    :param inputs: dictionary containing possible inputs to the system.
    This should at least contain:
    a) analog_z_score (assumed to be obtained from a docking calculation
    between the analog and GLP1R). This score is relative to GLP1. Higher scores
    indicate higher binding affinity than GLP1 and a score of 0 means the
    incretin is effectively GLP1.
    NOTE: If not given, arbitrary values will be assigned.

    :param evidence: dict containing evidence (specify at compile time)

    :param start: dict containing starting (t=0) values of timeseries that
    depend on their own past timeslices

    :param t: # of timeslices

    :param hpfn: json file containing dictionary of hyperpriors. Will raise
    error if not supplied

    :param name: name of model (specify at compile time, suggested: "vs")
    """

    # analog conc
    # -----------------------------------
    # dynamic parents (no static parents)
    dp = {"node": "me", "timeslices": 0}
    sigma_analog_conc = set_hp("sigma_analog_conc", name, hpfn)
    analog_conc_prior = GaussianTimeSeries("analog_conc_prior", dynamic=dp,
                                    static=[], fwd_model=_f_analog_conc,
                                    sigma=sigma_analog_conc, t=t)
    analog_conc = set_input("analog_conc", inputs, prior=analog_conc_prior)

    # GLP1R
    # -----------------------------------
    # input analog z_score
    # (set a dummy prior in case this is treated as evidence)
    analog_z_score_default = set_hp("analog_z_score", name, hpfn)
    analog_z_score_prior = pm.Normal("analog_z_score_prior",
                                   mu=analog_z_score_default,
                                   sigma=set_default_sigma(
                                       analog_z_score_default))
    analog_z_score = set_input("analog_z_score", inputs,
                               prior=analog_z_score_prior)

    # GLP1R
    # (use standard PyMC3 RV algebra since no dependence on past timesteps)
    sigma_GLP1R = set_hp("sigma_GLP1R", name, hpfn)
    GLP1R = pm.Normal("GLP1R", mu=_f_GLP1R(analog_conc, analog_z_score),
                      sigma=sigma_GLP1R, shape=t)

