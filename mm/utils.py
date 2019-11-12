import os
import json
import numpy as np
import scipy.stats as stats

def set_default_sigma(mu):
    """
    Sets a default sigma for a Gaussian distribution based on the mean mu
    If mu is very small, it returns 0.01 as default. Any smaller sigma can
    cause numerical artifacts during Bayesian inference with PyMC3. If you
    need a smaller sigma, consider reparameterizing your model.

    :param mu: mean of the Gaussian

    :return: either 0.1 * mu or 0.01 whichever is larger
    """
    return max(mu/10., 0.01)


def set_hp(hpname, modelname, hpfn, frac=0.1):
    """
    Extract hyperpriors either from the supplied dict of hyperpriors or if
    not present, read it from default json file

    :param hpname: Name of the hyperprior

    :param modelname: Name of input model to which this hyperprior belongs

    :param hpfn: Json file containing dict of hyperpriors.

    :param frac: If the requested hyperprior is a sigma and is not found
    either in the supplied dict, try and set it to one order of magnitude
    lower than the corresponding mean taking it either from the supplied dict

    :return: values of hyper-prior corresponding to key hpname in the
    supplied dict
    """

    with open(os.path.abspath(hpfn), "r") as of:
        master_hpdict = json.load(of)

    # get the dict for the given model
    if not modelname:
        hpdict = master_hpdict
    else:
        hpdict = master_hpdict[modelname]

    # query the supplied dict
    out = hpdict.get(hpname, None)

    # if not found, check if this hyperprior is a sigma and set it to
    # a frac of the corresponding mean value
    if out is None:
        if hpname.split("_")[0] == "sigma":
            hpname_mu = hpname.split("sigma_")[1]
            if hpname_mu in hpdict:
                out = frac * hpdict[hpname_mu]

    # if still not found, raise an error
    if out is None:
        raise ValueError("Hyperprior %s not supplied and cannot be estimated"
                         % hpname)

    return out

def set_input(name, inputdict, prior=None):
    """
    Extract an input random variable (RV) from the given input dict to a
    model. This helps to reuse the same model builder function for accepting
    both priors and trained point estimates or distributions (to be
    implemented soon) on parameters. Input RVs are essential for coupling
    different models.

    :param name: name of the RV (does not need to be prepended by the model
    name)

    :param inputdict: dict containing input RV (or python var)

    :param prior: a RV to fall back-on in case not supplied in the inputdict

    :return: RV with given name
    """

    rv = inputdict.get(name, None)

    # check if this is a set of samples drawn from some posterior
    # in which case convert it to an Empirical distribution
    #TODO: to be implemented later

    # if the rv is not provided, use the supplied prior
    if rv is None:
        if prior is None:
            raise TypeError("No input or prior supplied for RV %s" % name)
        rv = prior

    return rv

def set_start(rvname, modelname, hpfn):
    """
    Sets a starting point for random variable (RV)s of GaussianTimeSeries
    type. For all other RV types, supplying a starting point has no effect
    and is silently ignored.

    :param rvname: name of the RV

    :param modelname: name of the parent model it belongs to

    :param hpfn: Json file containing dict of starting values (which are
    essentially hyperpriors too). Starting values for RVs should use keys of
    the form <rvname>_0 in the file.

    :return: starting value for GaussianTimeSeries types of RVs. If not found
    in the file returns 0
    """

    with open(os.path.abspath(hpfn), "r") as of:
        master_hpdict = json.load(of)

    # get the dict for the given model
    if not modelname:
        hpdict = master_hpdict
    else:
        hpdict = master_hpdict[modelname]

    queryname = rvname + "_0"

    # query the supplied dict
    out = hpdict.get(queryname, 0.0)

    return out


def get_distribution(samples, vartype="static", smooth=True):
    """
    Convert samples into a distribution.

    :param samples: samples

    :param vartype: "static" (non-arrays) or "dynamic"

    :return: bin-centers and bin-values for the corresponding histogram
    """

    if vartype == "static":
        if not smooth:
            y, bin_edges = np.histogram(samples, bins=20, density=True)
            x = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(y))]
        else:
            smin, smax = np.min(samples), np.max(samples)
            width = smax - smin
            x = np.linspace(smin, smax, 100)
            y = stats.gaussian_kde(samples)(x)
            x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
            y = np.concatenate([[0], y, [0]])

        return x, y

    elif vartype == "dynamic":
        y = np.mean(samples, axis=0)
        yerr = np.std(samples, axis=0, ddof=1)
        return y, yerr

    else:
        raise TypeError("Unknown variable type. Use 'static' or 'dynamic' ")

