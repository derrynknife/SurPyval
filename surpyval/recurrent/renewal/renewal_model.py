from surpyval.recurrent.inference import LikelihoodInferenceMixin
from surpyval.recurrent.simulation import RecurrenceSimulationMixin


class RenewalModel(RecurrenceSimulationMixin, LikelihoodInferenceMixin):
    """
    A fitted renewal / imperfect-repair recurrence model.

    This is the model object returned by the renewal-family fitters
    (``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``), in the same
    way that the intensity fitters (``Crow``, ``Duane``, ...) return a
    ``ParametricRecurrenceModel``. It holds the fitted underlying lifetime
    distribution and the restoration parameter, and provides the simulation
    (``mcf``, ``plot``, ``count_terminated_simulation``,
    ``time_terminated_simulation``) and likelihood-inference
    (``log_likelihood``, ``aic``, ``bic``, ``standard_errors``) behaviour via
    the shared mixins.

    These processes have no closed-form intensity, so the mean cumulative
    function is obtained by simulation.

    Parameters
    ----------
    model : Parametric
        The fitted underlying lifetime distribution.
    restoration : float
        The fitted restoration / repair parameter (``q`` for the generalized
        and G1 renewal processes, ``rho`` for ARA).
    restoration_name : str
        The attribute/label name of the restoration parameter (e.g. ``"q"`` or
        ``"rho"``); it is also exposed as an attribute of that name.
    restoration_label : str
        Human-readable label used in ``__repr__`` (e.g. ``"Restoration
        Factor"`` or ``"Repair Efficiency"``).
    kind : str
        Display name of the process (e.g. ``"Generalized Renewal"``).
    sampler_factory : callable
        ``sampler_factory(model) -> sample`` returning a fresh per-sequence
        sampler ``sample(ui) -> interarrival`` for simulation.
    """

    def __init__(
        self,
        model,
        restoration,
        restoration_name,
        restoration_label,
        kind,
        sampler_factory,
    ):
        self.model = model
        self.restoration = restoration
        self._restoration_param_name = restoration_name
        self._restoration_label = restoration_label
        self.kind = kind
        self._sampler_factory = sampler_factory
        # Expose the restoration parameter under its conventional name
        # (``q``/``rho``) so existing usage keeps working.
        setattr(self, restoration_name, restoration)

    def _new_sequence_sampler(self):
        return self._sampler_factory(self)

    def __repr__(self):
        title = f"{self.kind} SurPyval Model"
        lines = [
            title,
            "=" * len(title),
            f"Distribution        : {self.model.dist.name}",
            "Fitted by           : MLE",
        ]
        if getattr(self, "kijima_type", None) is not None:
            lines.append(f"Kijima Type         : {self.kijima_type}")
        if getattr(self, "m", None) is not None:
            lines.append(f"Memory (m)          : {self.m}")
        lines.append(
            "{:<20}: {}".format(self._restoration_label, self.restoration)
        )

        param_string = "\n".join(
            "{:>10}".format(name) + ": " + str(p)
            for p, name in zip(self.model.params, self.model.dist.param_names)
        )
        return "\n".join(lines) + "\nParameters          :\n" + param_string
