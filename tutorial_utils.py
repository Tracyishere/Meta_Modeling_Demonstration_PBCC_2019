import os
import numpy as np
import random
import scipy.stats as stats

from pdf2image import convert_from_path
from IPython.display import HTML, Image, display
from IPython.core.magic import register_cell_magic

import pymc3 as pm

OUTDIR = os.path.abspath("./.tutorial_dump")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)


def visualize_model(m, prefix=None, h=500, w=500):
    """
    Visualize the PGM for a model in plate notation

    :param m: Compiled PyMC3 model

    :param prefix: prefix for the png file (defaults to model name)
    """
    if prefix is None:
        prefix = m.name
    prefix = os.path.join(OUTDIR, prefix)
    pm.model_to_graphviz(m).render(prefix)
    os.remove(prefix)
    img = convert_from_path(prefix + ".pdf")
    img[0].save(prefix + ".png")
    os.remove(prefix + ".pdf")
    img = Image(prefix + ".png", height=h, width=w)
    display(img)


def get_prior_samples(m, rvname, samples=500):
    """
    Collect samples for a random variable from the model prior

    :param m: Compiled PyMC3 model

    :param rvname: name of the RV

    :param samples: number of samples to generate from the prior

    :return: prior samples for the RV
    """

    rvlist = m.basic_RVs
    rv = [x for x in rvlist if x.name == rvname][0]

    return rv.distribution.random(size=samples)


def sample_posterior(m, tune=2000, samples=500):
    """
    Run variational inference (VI) on the model

    :param m: compiled PyMC3 model

    :param tune: number of steps to run VI

    :param samples: number of samples to draw from the approximate posterior
    returned from VI

    :return: a trace of samples
    """

    approx = pm.fit(model=m, method="advi", n=tune)
    trace = approx.sample(draws=samples)
    return approx.hist, trace


def get_distribution(samples, vartype="static", smooth=False):
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
