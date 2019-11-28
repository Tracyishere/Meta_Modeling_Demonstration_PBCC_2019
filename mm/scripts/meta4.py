import sys
import numpy as np
import pymc3 as pm
import theano.tensor as tt

from mm.core import jit, GaussianTimeSeries
from mm.utils import set_hp, set_input, set_default_sigma
from mm.models import MealModel, SPTModel, NetworkModel, VirtualScreenModel

# timestep (in mins)
dt = 2.0

@jit
def MetaModel(inputs={}, evidence={}, start={}, t=2,
              hpfn=None, name="meta", mealmodel_type="normal"):
    """
    TODO: Add description

    :param inputs: dictionary containing possible inputs to the system

    :param evidence: dict containing evidence (specify at compile time)
    This should at least contain:
    a) derivative of ingested glucose conc. (DGintake)
    b) Concentration of incretin hormone (GLP1 analog) (analog_conc)
    c) Binding affinity score of incretin hormone relative to GLP1
       (analog_z_score)

    NOTE: If not given, arbitrary values will be assigned.

    :param start: dict containing starting (t=0) values of timeseries that
    depend on their own past timeslices

    :param t: # of timeslices

    :param hpfn: json file containing dictionary of hyperpriors. Will raise
    error if not supplied.

    :param name: name of model (specify at compile time, suggested: "meta")

    :param mealmodel_type: "normal" or "t2d". The meal-model specific parameter
    set will be loaded accordingly from the hyper-parameter file
    """

    # call the meal model MBF
    meal = MealModel(inputs=inputs, evidence=evidence, start=start,
                     hpfn=hpfn, name="meal_%s" % mealmodel_type, t=t)

    # write a coupling function to couple all models together
    # This basically acts as a time evolution function for the coupled variables
    # G (plasma glucose), Y (glucose level dependent insulin response),
    # I (plasma insulin), but at the meta-model level. Hence this is written
    # *inside* the MBF to be able to access all inputs passed to the MBF

    def _f_Meta_State(State_0, DGintake_0, DGintake_1):
        # unpack state variables
        G_0 = State_0[:, 0]
        Y_0 = State_0[:, 1]
        I_0 = State_0[:, 2]

        # get plasma glucose from meal model
        G_1 = (1 - meal.k2 * dt) * G_0 - meal.k1 * dt * I_0 + dt * DGintake_0

        # initialize a virtual screen model
        # with GLP1 analog information as input
        vs = VirtualScreenModel(inputs=inputs, evidence=evidence, start=start,
                                hpfn=hpfn, name="vs", t=t)

        # initialize a network model that is coupled to the virtual screen
        # and meal models
        net = NetworkModel(inputs={"G_in": G_0, "GLP1R_ext": vs.GLP1R},
                           evidence=evidence, start=start,
                           name="net", hpfn=hpfn, t=t)

        # initialize a spt model that is coupled to the meal model
        spt = SPTModel(inputs={"G_in": G_1}, evidence=evidence, start=start,
                       name="spt", hpfn=hpfn, t=t)

        # calculate insulin secretion from meal model alone
        Y_1 = (1 - meal.alpha * dt) * Y_0 + \
              meal.alpha * beta * dt * (G_0 - meal.Gb)
        S_meal_0 = Y_1 + meal.K * DGintake_1 + meal.Sb

        # couple all models through insulin secretion output
        S_0 = (S_meal_0 + spt.S + net.S) / 3.0

        # calculate plasma insulin
        I_1 = (1 - meal.gamma * dt) * I_0 + dt * S_0

        # pack outputs
        out = tt.stack([G_1, Y_1, I_1], axis=1)
        return out

    # "State" CPD
    # -----------------------------------
    # NOTE: Because of the feedback loop in the model equations, G,Y,I have
    # to be solved *simultaneously* as a vector called "State", i.e.
    # State = [G, Y, I]

    # State
    # dynamic parents
    dp_State = [{"node": "me", "timeslices": 0},
                {"node": inputs["DGintake"], "timeslices": [0, 1]}]

    sigma_State = 1.00
    State = GaussianTimeSeries("State", dynamic=dp_State, static=[],
                               fwd_model=_f_Meta_State, sigma=sigma_State,
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
    S = pm.Normal("S", mu=Y+inputs["DGintake"]+meal.Sb, sigma=0.001, shape=t)
