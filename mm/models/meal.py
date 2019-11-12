import sys
import numpy as np
import pymc3 as pm
from pymc3 import memoize
import theano.tensor as tt

from mm.core import jit, GaussianTimeSeries
from mm.utils import set_hp, set_input, set_default_sigma

# timestep (in mins)
dt = 2.0


# Time evolution functions
# -----------------------------------
# These are the "forward models" for the corresponding GaussianTimeSeries CPDs
# The weights (w_i)s are numbered exactly as defined in the doc-string of the
# model builder function.

# In these functions, x_0 and x_1 respectively refer to previous and present
# timeslice values of x, i.e. x[t-1] and x[t]. The returned value is always
# for the present timeslice.

# @memoize
def _f_State(State_0, DGintake_0, DGintake_1,
             alpha, beta, gamma, K, Gb, Sb, k1, k2):
    # unpack state variables
    G_0 = State_0[:, 0]
    Y_0 = State_0[:, 1]
    I_0 = State_0[:, 2]

    # apply time evolution equations
    G_1 = (1 - k2 * dt) * G_0 - k1 * dt * I_0 + dt * DGintake_0
    Y_1 = (1 - alpha * dt) * Y_0 + alpha * beta * dt * (G_0 - Gb)
    S_0 = Y_1 + K * DGintake_1 + Sb
    I_1 = (1 - gamma * dt) * I_0 + dt * S_0

    # pack outputs
    out = tt.stack([G_1, Y_1, I_1], axis=1)
    return out


# @memoize
def _f_S(Y_1, DGintake_1, K, Sb):
    return Y_1 + K * DGintake_1 + Sb


@jit
def MealModel(inputs={}, evidence={}, start={}, t=2,
              hpfn=None, name="meal_normal"):
    """
    Pharmacokinetic model for glucose induced postprandial inuslin secretion.
    Further details can be found in:
    Meal simulation model of the glucose-insulin system, C. Dalla Man,
    R.A. Rizza and C. Cobelli, IEEE Trans. Biomed., 2007, 54(10), 1740-9

    The basic equations (time discretized ODEs) were taken from this paper
    and modified:
    Y(t+1) = (1-alpha*dt)*Y(t) + alpha*beta*dt*(G(t)-Gb)
    S(t) = Y(t) + K*DGintake + Sb
    I(t+1) = (1-gamma*dt)*I(t) + dt*S(t)
    G(t+1) = (1-k2*dt)*G(t) -k1*dt*I(t) + dt*DGintake(t)

    The last equation is our own addition (although based on material from
    the paper), where the plasma glucose conc (G) is given a feedback from
    plasma insulin (I) and the derivative of glucose ingested during a meal (
    DGintake)

    :param inputs: dictionary containing possible inputs to the system

    :param evidence: dict containing evidence (specify at compile time)
    This should at least contain:
    a) derivative of ingested glucose conc. (DGintake)
    NOTE: If not given, arbitrary values will be assigned.

    :param start: dict containing starting (t=0) values of timeseries that
    depend on their own past timeslices

    :param t: # of timeslices

    :param hpfn: json file containing dictionary of hyperpriors. Will raise
    error if not supplied.

    :param name: name of model (specify at compile time, suggested:
    "meal_normal" or "meal_t2d")
    """

    # input derivative of ingested glucose conc.
    # (set a dummy prior in case this is treated as an evidence)
    DGintake_prior = pm.GaussianRandomWalk("DGintake_prior", sigma=1.0, shape=t)
    DGintake = set_input("DGintake", inputs, prior=DGintake_prior)

    # "State" CPD
    # -----------------------------------
    # NOTE: Because of the feedback loop in the model equations, G,Y,I have
    # to be solved *simultaneously* as a vector called "State", i.e.
    # State = [G, Y, I]

    # input alpha
    # (set a dummy prior in case this is treated as evidence)
    alpha_default = set_hp("alpha", name, hpfn)
    alpha_prior = pm.Normal("alpha_prior", mu=alpha_default,
                            sigma=set_default_sigma(alpha_default))
    alpha = set_input("alpha", inputs, prior=alpha_prior)

    # input beta
    # (set a dummy prior in case this is treated as evidence)
    beta_default = set_hp("beta", name, hpfn)
    beta_prior = pm.Normal("beta_prior", mu=beta_default,
                           sigma=set_default_sigma(beta_default))
    beta = set_input("beta", inputs, prior=beta_prior)

    # input gamma
    # (set a dummy prior in case this is treated as evidence)
    gamma_default = set_hp("gamma", name, hpfn)
    gamma_prior = pm.Normal("gamma_prior", mu=gamma_default,
                            sigma=set_default_sigma(gamma_default))
    gamma = set_input("gamma", inputs, prior=gamma_prior)

    # input K
    # (set a dummy prior in case this is treated as evidence)
    K_default = set_hp("K", name, hpfn)
    K_prior = pm.Normal("K_prior", mu=K_default,
                        sigma=set_default_sigma(K_default))
    K = set_input("K", inputs, prior=K_prior)

    # input Gb
    # (set a dummy prior in case this is treated as evidence)
    Gb_default = set_hp("Gb", name, hpfn)
    Gb_prior = pm.Normal("Gb_prior", mu=Gb_default,
                         sigma=set_default_sigma(Gb_default))
    Gb = set_input("Gb", inputs, prior=Gb_prior)

    # input Sb
    # (set a dummy prior in case this is treated as evidence)
    Sb_default = set_hp("Sb", name, hpfn)
    Sb_prior = pm.Normal("Sb_prior", mu=Sb_default,
                         sigma=set_default_sigma(Gb_default))
    Sb = set_input("Sb", inputs, prior=Sb_prior)

    # input k1
    # (set a dummy prior in case this is treated as evidence)
    k1_default = set_hp("k1", name, hpfn)
    k1_prior = pm.Normal("k1_prior", mu=k1_default,
                         sigma=set_default_sigma(k1_default))
    k1 = set_input("k1", inputs, prior=k1_prior)

    # input k2
    # (set a dummy prior in case this is treated as evidence)
    k2_default = set_hp("k2", name, hpfn)
    k2_prior = pm.Normal("k2_prior", mu=k2_default,
                         sigma=set_default_sigma(k2_default))
    k2 = set_input("k2", inputs, prior=k2_prior)

    # State
    # dynamic parents
    dp_State = [{"node": "me", "timeslices": 0},
                {"node": DGintake, "timeslices": [0, 1]}]
    # static parents
    sp_State = [alpha, beta, gamma, K, Gb, Sb, k1, k2]
    sigma_State = set_hp("sigma_State", name, hpfn)  # use spherical covariances
    State = GaussianTimeSeries("State", dynamic=dp_State, static=sp_State,
                               fwd_model=_f_State, sigma=sigma_State,
                               t=t, dim=3)

    # output G, I, S (derived from elements of state)
    # these should essentially be deterministic but give them (dummy)
    # distributions with small sigma anyway
    # -----------------------------------------------------------------
    # G
    G = pm.Normal("G", mu=State[:, 0], sigma=0.001, shape=t)

    # I
    I = pm.Normal("I", mu=State[:, 2], sigma=0.001, shape=t)

    # S
    Y = State[:, 1]
    S = pm.Normal("S", mu=_f_S(Y, DGintake, K, Sb), sigma=0.001, shape=t)
