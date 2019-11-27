"""
Just-in-time (jit) compilation of a PyMC3 model by associating observations
and starting values just when the model is compiled. Extends the "sampled"
decorator by Colin Caroll. (https://github.com/ColCarroll/sampled)
"""

from functools import wraps
from pymc3 import Model
from .gaussiantimeseries import GaussianTimeSeries


class CompiledModel(Model):
    """
    Compiles the model.
    a) Stores observed variables until the model is created. Observations are
    passed through the evidence dict.

    b) Sets starting values (testval) for GaussianTimeSeries distributions when
    the model is created. Starting values are passed through the start dict.

    Other keyword arguments can also be passed, that are useful for model
    construction through the model builder function that gets decorated
    using jit (see below).
    """

    def __init__(self, **kwargs):
        """
        Constructor for CompiledModel

        :param kwargs: keyword arguments for name of model, parent model (if
        any), evidence dict and input dict (any other keyword argument is
        ignored)
        """

        # instantiate model
        name = kwargs.get("name", None)
        model = kwargs.get("model", None)
        super(CompiledModel, self).__init__(name, model)

        # get evidence
        self.evidence = kwargs.get("evidence", {})

        # get RV (or python var) names supplied as input through the input dict
        self.input_names = list(kwargs.get("inputs", {}))

        # get starts
        self.start = kwargs.get("start", {})

    def _is_extraneous_RV(self, rvname):
        """
        Flags a random variable (RV) with name <rvname>_prior as extraneous
        if a RV (or python var) with name <rvname> has already been supplied
        with the inputs arg of the model builder function

        :param rvname: name of the RV

        :return: True if both RVs with names <rvname> and <rvname>_prior exist
        """

        has_prior = rvname != "_prior" and rvname.endswith("_prior")
        if not has_prior:
            return False
        else:
            basename = rvname.split("_prior")[0]
            return basename in self.input_names

    def Var(self, name, dist, data=None, total_size=None):
        """
        Overrides the Var method of Model() in PyMC3

        :param name: name of the random variable (RV) (Note: name of model gets
        prepended automatically)

        :param dist: PyMC3 distribution for this RV

        :param data: non-None values of data turns this into an observed
        RV. Data is taken from the evidence dict and should correspond to the
        key <modelname_varname> if modelname is not None else <varname>.

        :param total_size: total size of this RV

        :return: PyMC3 style FreeRV or ObservedRV depending on whether it is
        supplied in the evidence dict.
        """

        # check if this is an extraneous RV and in that case do nothing
        # since this is already supplied through the inputs argument to the
        # model builder function
        if self._is_extraneous_RV(name):
            return

        # if a RV passes the extraneous test, its time to chop off its
        # _prior suffix
        if name != "_prior" and name.endswith("_prior"):
            name = name.split("_prior")[0]

        varname = name
        if not self.name is None:
            varname = self.name + "_" + varname

        # set a starting point if available for GaussianTimeSeries dist
        if varname in self.start and self.start[varname] is not None and \
                isinstance(dist, GaussianTimeSeries):
            dist.set_initval(self.start[varname])

        return super(CompiledModel, self).Var(name, dist,
                                              data=self.evidence.get(varname,
                                                                     data),
                                              total_size=total_size)


def jit(f):
    """
    Decorator to delay initializing PyMC3 model until data is passed in.

    :param f: model builder function. This should be of the form
    f(name=<model_name>, **kwargs).
    
    <model_name>: any valid string (or "")

    kwargs: keyword arguments necessary for the model builder f. Recommended
    kwargs are a dict of evidence RVs, a dict of input RVs and a dict of
    starting points. In evidence and start dicts, RV names should be
    specified as <modelname_varname> if modelname is not None else <varname>.
    
    :return: PyMC3 model containing ObservedRVs if requested through the
    evidence dict
    """

    @wraps(f)
    def _wrapped_f(*args, **kwargs):
        """
        Wraps the model builder function

        :param args: positional arguments used by the model builder function.
        These can be used to supply input RVs (or python vars) to the model
        builder function.

        :param kwargs: keyword arguments for the model builder

        :return: PyMC3 model containing ObservedRVs if requested through the
        evidence dict
        """

        with CompiledModel(**kwargs) as model:
            f(*args, **kwargs)
        return model

    return _wrapped_f
