#!/usr/bin/env python

import numpy as np
import copy
from tqdm import trange

import theano.tensor as tt
import pymc3 as pm
from pymc3.distributions import distribution, draw_values, Continuous, Flat


class GaussianTimeSeries(Continuous):
    """
    A generalized version of PyMC3's GaussianRandomWalk timeseries
    distribution. This allows CPDs of the form:
    p(x(t+1) | x(t), y(t), y(t-1), z) = f(x(t), y(t), y(t-1), z) + N(0, sigma)
    i.e. the timeseries x(t+1) can depend on past values of itself (x(t)),
    past or present values of other PyMC3 random variables (RVs) (y(t),
    y(t-1)) and non-timeseries PyMC3 RVs z. Gaussian noise with
    std. dev. sigma will added. The function f is referred to as the
    "fwd_model".

    USAGE: x(t+1) = x(t) + 2*y1(t-1) + sin(y2(t)) + a + b
    where, y1 and y2 are timeseries (dynamic) and a and b are
    non-timeseries (static) RVs and t = 0 to 9. This can be specified as:

    1) first create a function that encapsulates the forward model. This can
    be an entire function or a lambda.
    def myfunc(x_0, y1_0, y2_1, a, b):
        return x_0 + y1

    #2) instantiate this RV (shown here within a model context):
    with pm.Model() as model:
        # define a, b, y1, y2
        a = ..
        b = ..
        y1 = ..
        y2 = ..

        dynamic_parents = [{"node": "me", "timeslice": 0},
                           {"node": y1,   "timeslice": 0},
                           {"node": y2,   "timeslice": 1}
        x = GaussianTimeSeries(dynamic=dynamic_parents,
                               static=[a, b],
                               fwd_model=myfunc,
                               sigma=0.5,
                               shape = 10)

    The order of variables defined in myfunc is VERY IMPORTANT. The order is:
    1) previous time point value of x (i.e. the variable on which this
    distribution is defined)

    2) all other dynamic RVs that are parents of this RV *in the order* in which
    they are defined in the argument dynamic.

    3) all static RVs that are parents of this RV *in the order* in which
    they are defined in the argument static.

    The dynamic argument is a list of dictionaries, each of the form
    {"node": <var>, timeslice: <timelist>},

    where <timelist> can be one of 0,
    1, [0], [1] or [0,1]. (0 is previous timepoint and 1 is current
    timepoint).

    <var> refers to a PyMC3 timeseries RV defined previously.

    "me" is a special dynamic RV that refers to the previous
    timepoint of x, i.e. the RV on which this distribution is applied,
    and any timeslice provided here will be ignored and default to 0.
    Instead of "me", "myself" or "self" are also accepted.

    Note: All vars, static or dynamic can be python variables
    (i.e. not PyMC3 RVs as well)
    """

    def __init__(self, dynamic=[], static=[], fwd_model=None,
                 sigma=1., lower=None, upper=None, initval=0.,
                 init=Flat.dist(), t=2, dim=1, *args, **kwargs):

        """
        Constructor for the GaussianTimeSeries distribution

        :param dynamic: single (or list) of PyMC3 timeseries RVs
        (or numpy arrays)

        :param static: single (or list) of PyMC3 non-timeseries RVs
        (or numpy floats, ints, etc)

        :param fwd_model: function or lambda defining the forward model. See
        USAGE documentation.

        :param sigma: std. dev for the Gaussian noise model

        :param lower: lower bound. Used to create truncated Gaussians.

        :param upper: upper bound: Used to create truncated Gaussians.

        :param initval: initial or starting value at t=0. Defaults to 0.

        :param init: initial distribution at t=0. Defaults to Flat()

        :param args: other arguments

        :param kwargs: other keyword arguments. Must specify at-least shape,
        here. Where shape = (n,) for a 1D RV with n timeslices or (n,d) for a
        d-dim RV with n timeslices.
        """

        # shape handling
        shape = None
        if isinstance(dim, int):
            if dim == 1:
                shape = t
            elif dim > 1:
                shape = (t, dim)
        elif isinstance(dim, tuple):
            shape = (t,) + dim
        else:
            raise ValueError("dim %s can only be int or tuple" % str(dim))
        kwargs["shape"] = shape

        # init a continuous distribution
        super(GaussianTimeSeries, self).__init__(*args, **kwargs)

        # associate attributes
        self.dynamic = dynamic
        self.static = static
        self.fwd_model = fwd_model

        # set defaults for initial value (broadcast if necessary)
        self._shape = shape
        self.set_initval(initval)

        # get initial distribution
        self.init = init

        # check dynamic parents
        if not isinstance(self.dynamic, list):
            self.dynamic = [self.dynamic]
        n_dynamic = len(self.dynamic)
        if n_dynamic > 0:
            for i, p in enumerate(self.dynamic):
                var = p["node"]
                timeslices = p["timeslices"]
                if not isinstance(timeslices, list):
                    timeslices = [timeslices]
                if len(timeslices) > 2 or any(t > 1 for t in timeslices):
                    raise ValueError("Only current or previous timeslice \
                                    allowed")
                p["timeslices"] = timeslices
                self.dynamic[i] = p

        # check static parents
        if not isinstance(self.static, list):
            self.static = [self.static]

        # set default fwd_model which is f(parents) = sum(parents)
        if self.fwd_model is None and (self.dynamic or self.static):
            parents = self.dynamic + self.static
            self.fwd_model = lambda *parents: tt.sum(parents)

        # set params for gaussian noise model
        self.sigma = sigma
        self.lower = lower
        self.upper = upper

    def set_initval(self, value):
        """
        Sets an initial (t=0) value for this RV

        :param value: value to set as initial value
        """

        self.testval = np.broadcast_to(value, self._shape)

    def _is_dynamic_parent(self, p):
        """
        Check if the parent is dynamic (i.e. timseries)

        :param p: parent RV (or python var)

        :return: True if p belongs to the list of dynamics
        """

        # return p in self.dynamic

        # written without using elemwise comparison from self.dynamic
        # because of possible bug in numpy which might break it in future
        # https://github.com/numpy/numpy/issues/6784
        return isinstance(p, dict) and ("node" in p and "timeslices" in p)

    def _is_static_parent(self, p):
        """
        Check if the parent is static (i.e. non-timeseries)

        :param p: parent RV (or python var)

        :return: True if p belongs to the list of statics
        """

        # return p in self.static

        # written without using elemwise comparison from self.dynamic
        # because of possible bug in numpy which might break it in future
        # https://github.com/numpy/numpy/issues/6784

        return not self._is_dynamic_parent(p)

    def _is_me(self, p):
        """
        Check if the parent is previous timepoint value of the RV on which
        this distribution is applied.

        :param p: parent RV

        :return: True if p belongs to the list of dynamics
        and the "node" key of p is "me", "myself" or "self"
        """

        return self._is_dynamic_parent(p) and \
               isinstance(p["node"], str) and \
               p["node"] in ["me", "myself", "self"]

    def _apply_fwd_model(self, x):
        """
        Parses the parents (dynamic, static, "me", etc) and uses them to
        evaluate the fwd model at the next timepoint.

        :param x: Theano-tensor containing current timepoint values of the RV
        to which this distribution is applied.

        :return: Theano-tensor containing the output of fwd_model applied to
        x and its parents.
        """

        # parse args for the fwd model
        fwd_args = []
        # add all dynamic parents
        for p in self.dynamic:
            var = p["node"]
            if self._is_me(p):
                var = x
            # add past values
            timeslices = p["timeslices"]
            if 0 in timeslices:
                fwd_args.append(var[:-1])
            # add present values except for the special dynamic parent "me"
            # ignore if current timeslice supplied for "me"
            if 1 in timeslices and not self._is_me(p):
                fwd_args.append(var[1:])
        # add all static parents
        [fwd_args.append(p) for p in self.static]
        # evaluate the fwd model with these args
        return self.fwd_model(*fwd_args)

    def logp(self, x):
        """
        Log-posterior calculator for this RV.

        :param x: Theano-tensor specifying the value of this RV

        return: Theano-tensor that gives the log-posterior.
        """

        # Gaussian pdf
        pdf = pm.Normal
        # decide whether to add bounds
        if (self.upper is None and self.lower is None):
            transformed_pdf = pdf
        else:
            transformed_pdf = pm.Bound(pdf, lower=self.lower, upper=self.upper)
        # get the likelihood for each time point
        fwd_i_minus_1 = self._apply_fwd_model(x)
        x_i = x[1:]
        innov_like = transformed_pdf.dist(mu=fwd_i_minus_1, sigma=self.sigma,
                                          shape=self._shape).logp(x_i)
        # evaluate the total log-likelihood for present values (i.e. ignore
        # the first value of the sequence)
        return self.init.logp(x[0]) + tt.sum(innov_like)

    def _draw_from_parents(self, parent, point=None, size=None):
        """
        A wrapper over PyMC3's distribution.draw_values() method that
        generalizes drawing random samples from the parents of this
        distribution to cases where the parents can be PyMC3 RVs or python
        vars (ints, floats, numpy arrays, etc).

        :param parent: parent RV (or python var)

        :param point: value of this RV at which to draw samples.
                      (Recommendation: Use default None)

        :param size: number of samples requested (defaults to 1)

        :return: a set of random samples from each parent (RV or python var)
        of this RV
        """

        if size is None:
            size = 1

        # static or dynamic?
        if self._is_dynamic_parent(parent):
            p = parent["node"]
        elif self._is_static_parent(parent):
            p = parent

        # if this is just the previous timeslice value of this node,
        # then fill in with self.testval (which is also the initial value)
        # and tile to get the correct dimension
        if self._is_me(parent):
            val = self.testval[np.newaxis, :]
            val = np.repeat(val, size, axis=0)

        # if this is a numpy array, return the array broadcasted to one
        # dimension higher. The new dimension is now the 0th dimension and is
        # the number of samples requested, i.e., size
        elif isinstance(p, np.ndarray):
            val = p[np.newaxis, :]
            val = np.repeat(val, size, axis=0)

        # for all other cases, uses pymc3's internal draw_values API
        else:
            val = draw_values([p], point=point, size=size)[0]

        return val

    def _random(self, noise_shape, point):
        """
        Draw a single random sample the shape for the (Gaussian) noise model

        :param noise_shape: shape to be used for the Gaussian noise model

        :param point: value of this RV at which to draw samples

        :return: a single sample
        """
        # assemble parent vals (this logic is a lot like apply_fwd_args)
        fwd_args = []

        # assemble dynamic parents
        for p in self.dynamic:
            # draw a single parent value
            var = self._draw_from_parents(p, point=point, size=2)[0]
            # add past values
            timeslices = p["timeslices"]
            if 0 in timeslices:
                fwd_args.append(var[:-1])
            # add present values except for the special dynamic parent "me"
            # ignore if current timeslice supplied for "me"
            if 1 in timeslices and not self._is_me(p):
                fwd_args.append(var[1:])

        # assemble static parents
        for p in self.static:
            var = self._draw_from_parents(p, point=point, size=1)
            fwd_args.append(var)

        # evaluate the fwd model with these args
        # use eval since if this turns out to be a theano tensor
        fwd_out = self.fwd_model(*fwd_args)
        if isinstance(fwd_out, tt.TensorVariable):
            fwd_i_minus_1 = fwd_out.eval()
        else:
            fwd_i_minus_1 = fwd_out

        # add noise model
        x_i_minus_1 = fwd_i_minus_1 + np.random.random(noise_shape) * \
                      self.sigma

        # add the initial value to make the correct final shape
        x_0 = np.array([self.testval[0]])
        x_i = np.concatenate([x_0, x_i_minus_1], axis=0)
        return x_i

    def random(self, point=None, size=None, quiet=True):
        """
        Draw a random sample from this RV given its parents.

        :param point: value of this RV at which to draw samples.

        :param size: number of samples requested.

        :param quiet: Set to true to prevent displaying progressbar on screen

        :return: set of random samples of shape (size,) for 1D RVs or (size,
        d) for d-dim RVs.
        """

        if size is None:
            size = 1

        # figure out the correct noise model shape
        if isinstance(self._shape, tuple):
            nt = self._shape[0]
            rest = self._shape[1:]
            noise_shape = (nt - 1,) + rest
        else:
            noise_shape = self.shape - 1

        # form a list that holds the samples
        # remember that each sample is a timeseries
        # Note: adding to a list makes it easier to cast the final output as a
        # multidim array with the number of samples (i.e. size) as the 0th
        # dimension.
        x = []

        # prepare args for the slave function _random
        args = [(noise_shape, point) for n in range(size)]

        # run in serial
        # NOTE: non-eager traversing of the computation graph is the fastest
        # i.e. parent vals are drawn only when required inside the _random
        # method

        if quiet:
            sampling_range = range(size)
        else:
            sampling_range = trange(size)

        [x.append(self._random(point=point, noise_shape=noise_shape)) for _
         in sampling_range]

        return np.asarray(x)
