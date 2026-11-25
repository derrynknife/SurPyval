"""The questions of an accelerated life test that the fitted model of it
answers beside its predictions: the activation energy of a thermally
activated life model in electronvolts, whether the shape is common to the
stress levels, and the life against stress drawn with its bounds.

The host is a
:class:`~surpyval.univariate.regression.ParametricRegressionModel`; every
method refuses a model that is not an accelerated life model, saying which
model each needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

#: Boltzmann's constant in electronvolts per kelvin: exactly
#: 1.380649e-23 J/K over 1.602176634e-19 J/eV (the SI since 2019).
BOLTZMANN_EV = 1.380649e-23 / 1.602176634e-19


class AcceleratedLifeAnalysisMixin:
    """``activation_energy``, ``common_shape_test`` and
    ``plot_life_stress`` for an accelerated life model."""

    model: Any
    distribution: Any
    data: Any
    params: Any
    kind: str
    _is_accelerated_life: Any
    param_cb: Any
    qf: Any
    quantile_cb: Any
    mean: Any
    mean_cb: Any
    life_parameter: Any

    def _life_model(self, what: str) -> Any:
        if not self._is_accelerated_life():
            raise ValueError(
                f"{what} is for an accelerated life model (AcceleratedLife("
                "distribution, life_model).fit(...)); this model is a "
                f"{self.kind} model."
            )
        return getattr(self.model, "life_model", None) or getattr(
            self, "reg_model", None
        )

    # -- activation energy ------------------------------------------------

    def activation_energy(
        self,
        alpha_ci: float = 0.05,
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> dict:
        r"""
        The activation energy :math:`E_a` of a thermally activated life
        model (Arrhenius, ``life_models.Arrhenius`` or ``Exponential``;
        ``Eyring``; and their inverses; and the temperature term of
        ``PowerExponential``, temperature with a non-thermal stress), in
        electronvolts, with its two-sided ``1 - alpha_ci`` interval.

        The life model's parameter is :math:`E_a / k_B` in kelvin (the
        temperatures in kelvin); :math:`E_a` is it times Boltzmann's
        constant, :math:`k_B = 8.617333262\times 10^{-5}` eV/K (exact in
        the SI). The interval is the parameter's :meth:`param_cb` by
        ``method`` (``"wald"``, ``"lr"`` or ``"bootstrap"``), scaled the
        same way.

        Parameters
        ----------
        alpha_ci : float, optional
            The significance level of the interval, by default 0.05.
        method : str, optional
            ``"wald"`` (the default), ``"lr"`` or ``"bootstrap"``, as for
            :meth:`param_cb`.
        n_boot : int, optional
            The refits of a bootstrap, by default 200.
        random_state : optional
            The seed of a bootstrap.

        Returns
        -------
        dict
            ``"activation_energy"`` (eV), ``"lower"`` and ``"upper"`` (eV),
            ``"kelvin"`` (:math:`E_a / k_B`, K, with ``"kelvin_lower"`` and
            ``"kelvin_upper"``), ``"parameter"`` (the life model's
            parameter it comes from), ``"alpha_ci"`` and ``"method"``.

        Raises
        ------
        ValueError
            If the model is not an accelerated life model, or its life
            model is not thermally activated.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> from surpyval import life_models
        >>> rng = np.random.default_rng(3)
        >>> T = np.repeat([358.15, 378.15, 398.15], 20)
        >>> life = 1e-6 * np.exp(0.7 / 8.617333262e-5 / T)
        >>> x = life * rng.weibull(2.0, T.size)
        >>> model = sp.AcceleratedLife(sp.Weibull, life_models.Arrhenius)
        >>> model = model.fit(x, Z=T)
        >>> energy = model.activation_energy()
        >>> round(energy["activation_energy"], 3), round(energy["kelvin"])
        (0.618, 7169)
        >>> round(energy["lower"], 3), round(energy["upper"], 3)
        (0.529, 0.707)
        """
        life_model = self._life_model("activation_energy")
        found = getattr(life_model, "activation_parameter", None)
        if found is None:
            name = getattr(life_model, "name", life_model)
            raise ValueError(
                f"The {name} life model has no activation energy: it is "
                "for a thermally activated life model, Arrhenius "
                "(life_models.Arrhenius), Eyring or their inverses, or "
                "PowerExponential, with the temperatures in kelvin."
            )
        name, sign = found
        value = sign * float(self.params[self._parameter_index(name)])
        bounds = sign * np.asarray(
            (
                self.param_cb(
                    name,
                    alpha_ci=alpha_ci,
                    method=method,
                    n_boot=n_boot,
                    random_state=random_state,
                )
                if method == "bootstrap"
                else self.param_cb(name, alpha_ci=alpha_ci, method=method)
            ),
            dtype=float,
        )
        low, high = float(np.min(bounds)), float(np.max(bounds))
        return {
            "activation_energy": value * BOLTZMANN_EV,
            "lower": low * BOLTZMANN_EV,
            "upper": high * BOLTZMANN_EV,
            "kelvin": value,
            "kelvin_lower": low,
            "kelvin_upper": high,
            "parameter": name,
            "alpha_ci": float(alpha_ci),
            "method": method,
        }

    def _parameter_index(self, name: str) -> int:
        names = list(getattr(self, "parameter_names"))
        return names.index(name)

    def _activation_line(self) -> str:
        """The printed model's activation energy, for a thermally
        activated life model; empty otherwise or where it has no
        interval."""
        if not self._is_accelerated_life():
            return ""
        life_model = self._life_model("activation_energy")
        if getattr(life_model, "activation_parameter", None) is None:
            return ""
        try:
            energy = self.activation_energy()
        except (ValueError, np.linalg.LinAlgError):
            return ""
        return (
            "\nActivation energy   : {:.4g} eV (Wald 95%: {:.4g} to {:.4g}"
            " eV), {} = E_a / k_B in kelvin".format(
                energy["activation_energy"],
                energy["lower"],
                energy["upper"],
                energy["parameter"],
            )
        )

    # -- common shape -----------------------------------------------------

    def common_shape_test(self) -> dict:
        r"""
        The likelihood-ratio test that the distribution's shape is the
        same at every stress level, as an accelerated life model assumes.

        At each distinct stress (each distinct row of ``Z``) the baseline
        distribution is fitted on its own, its life and shape free; and
        all levels together with one shape and a life of their own each.
        Twice the difference of the maximised log-likelihoods is, under a
        common shape, :math:`\chi^2` with ``k - 1`` degrees of freedom for
        ``k`` levels (Nelson, *Accelerated Testing*, 1990, ch. 9; Meeker
        and Escobar, 1998, section 19.4). Neither likelihood involves the
        life-stress relation, so the test is of the shape alone.

        A level with fewer than two failures has no shape of its own: it
        is left out of the test, with a warning naming it and its
        failures.

        Returns
        -------
        dict
            ``"statistic"``, ``"df"``, ``"p_value"``, ``"shapes"`` (each
            level's own, by its stress), ``"common_shape"``,
            ``"levels"`` (the stresses tested, as tuples),
            ``"log_likelihood_separate"``, ``"log_likelihood_common"``,
            ``"parameter"`` (the shape's name) and ``"left_out"`` (the
            levels left out, with their failures).

        Raises
        ------
        ValueError
            If the model is not an accelerated life model, has no data or
            was fitted with weights, its distribution has not exactly one
            parameter beside the life, or fewer than two levels have two
            failures.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> from surpyval import life_models
        >>> rng = np.random.default_rng(3)
        >>> T = np.repeat([358.15, 378.15, 398.15], 20)
        >>> life = 1e-6 * np.exp(0.7 / 8.617333262e-5 / T)
        >>> x = life * rng.weibull(2.0, T.size)
        >>> model = sp.AcceleratedLife(sp.Weibull, life_models.Arrhenius)
        >>> test = model.fit(x, Z=T).common_shape_test()
        >>> test["df"], round(test["statistic"], 3), round(test["p_value"], 3)
        (2, 0.333, 0.847)
        """
        import warnings

        from scipy import optimize, stats

        from surpyval.utils.warnings import (
            caller_stacklevel,
            ignoring_warnings,
        )

        self._life_model("common_shape_test")
        data = self._raw_data("common_shape_test")
        if getattr(data, "w", None) is not None:
            raise ValueError(
                "common_shape_test does not take a fit with weights w: the "
                "test is of likelihoods, and a weighted fit maximises a "
                "pseudo-likelihood. Refit without weights."
            )
        names = list(self.distribution.parameter_names)
        others = [n for n in names if n != self.life_parameter]
        if len(others) != 1:
            raise ValueError(
                f"common_shape_test needs a distribution with one shape "
                f"parameter beside its life ({self.life_parameter}); "
                f"{self.distribution.name} has {others or 'none'}."
            )
        shape_name = others[0]
        Z = np.asarray(data.Z, dtype=float)
        levels, which = np.unique(Z, axis=0, return_inverse=True)
        which = np.ravel(which)
        failed = np.asarray(data.c) != 1
        counts = np.asarray(data.n)
        kept, left_out = [], {}
        for j, level in enumerate(levels):
            failures = int(np.sum(counts[(which == j) & failed]))
            key = tuple(float(v) for v in level)
            if failures < 2:
                left_out[key] = failures
            else:
                kept.append(j)
        if left_out:
            listed = ", ".join(
                (
                    f"{k if len(k) > 1 else k[0]:g}: {v}"
                    if len(k) == 1
                    else f"{k}: {v}"
                )
                for k, v in left_out.items()
            )
            warnings.warn(
                f"common_shape_test left out {len(left_out)} stress "
                f"level(s) with fewer than 2 failures ({listed}): a level "
                "needs two failures for a shape of its own. The test is of "
                "the other levels; test longer at those stresses to "
                "include them.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        if len(kept) < 2:
            raise ValueError(
                "common_shape_test needs at least two stress levels with "
                f"two failures or more; {len(kept)} have them."
            )

        def rows(j: int) -> dict:
            mask = which == j
            return {
                "x": np.asarray(data.x)[mask],
                "c": np.asarray(data.c)[mask],
                "n": counts[mask],
                "t": np.asarray(data.t)[mask],
            }

        dist = self.distribution
        with ignoring_warnings():
            separate = [dist.fit(**rows(j)) for j in kept]
        ll_separate = -float(sum(m.neg_ll() for m in separate))
        shapes = [float(m.params[names.index(shape_name)]) for m in separate]

        def neg_common(log_shape: float) -> float:
            fixed = {shape_name: float(np.exp(log_shape))}
            with ignoring_warnings():
                total = 0.0
                for j in kept:
                    total += dist.fit(**rows(j), fixed=fixed).neg_ll()
            return float(total)

        logs = np.log(shapes)
        found = optimize.minimize_scalar(
            neg_common,
            bounds=(logs.min() - 1.0, logs.max() + 1.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        ll_common = -float(found.fun)
        statistic = max(2.0 * (ll_separate - ll_common), 0.0)
        df = len(kept) - 1
        tested = [tuple(float(v) for v in levels[j]) for j in kept]
        return {
            "statistic": statistic,
            "df": df,
            "p_value": float(stats.chi2.sf(statistic, df)),
            "shapes": dict(zip(tested, shapes)),
            "common_shape": float(np.exp(found.x)),
            "levels": tested,
            "log_likelihood_separate": ll_separate,
            "log_likelihood_common": ll_common,
            "parameter": shape_name,
            "left_out": left_out,
        }

    def _raw_data(self, what: str) -> Any:
        """The fitted rows, with the covariates as given (not centred)."""
        centring = getattr(self, "_fit_centring", None)
        raw = getattr(centring, "raw", None)
        data = raw if raw is not None else getattr(self, "data", None)
        if data is None or getattr(data, "Z", None) is None:
            raise ValueError(
                f"{what} needs the data the model was fitted to, which "
                "this model does not keep (one built by from_params or "
                "restored with from_dict): refit it."
            )
        return data

    # -- life against stress ----------------------------------------------

    def plot_life_stress(
        self,
        p: float | None = 0.5,
        stresses: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
        column: int = 0,
        Z: npt.ArrayLike | None = None,
        points: bool = True,
        ax: Any = None,
    ) -> Any:
        r"""
        Plot a life against stress, with its bounds and the test levels
        marked: the life model's relation, as an engineer reads an
        accelerated test.

        The life is the quantile ``qf(p, Z)`` (``p=0.5`` the median, the
        default; ``0.1`` the B10 life) with its :meth:`quantile_cb`
        bounds, or, with ``p=None``, the mean life with :meth:`mean_cb`
        bounds, at each stress of a grid over ``stresses`` (by default
        from the lowest test stress, or the use stress if lower, to the
        highest). The stress varies in column ``column`` of ``Z``, the
        other columns held at ``Z`` (by default the first row fitted).
        The test stresses are marked, with the failures at each
        (``points``) as dots and the units censored as open marks; the
        time axis is logarithmic.

        Parameters
        ----------
        p : float or None, optional
            The probability of the quantile plotted, by default 0.5; None
            plots the mean.
        stresses : array_like, optional
            The stresses to draw over: their range is gridded (give the
            use stress here to extrapolate to it).
        alpha_ci : float, optional
            The significance level of the bounds, by default 0.05.
        column : int, optional
            The covariate column that is the stress, by default 0.
        Z : array_like, optional
            The covariate row whose other columns are held; by default the
            first row fitted.
        points : bool, optional
            Whether to draw the data at each test stress, by default
            True.
        ax : matplotlib Axes, optional
            The axes to draw on; a new figure's by default.

        Returns
        -------
        matplotlib Axes

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> from surpyval import life_models
        >>> rng = np.random.default_rng(3)
        >>> T = np.repeat([358.15, 378.15, 398.15], 20)
        >>> life = 1e-6 * np.exp(0.7 / 8.617333262e-5 / T)
        >>> x = life * rng.weibull(2.0, T.size)
        >>> model = sp.AcceleratedLife(sp.Weibull, life_models.Arrhenius)
        >>> ax = model.fit(x, Z=T).plot_life_stress(
        ...     p=0.1, stresses=[313.15, 398.15])
        >>> ax.get_ylabel()
        'B10 life'
        """
        import matplotlib.pyplot as plt

        self._life_model("plot_life_stress")
        data = self._raw_data("plot_life_stress")
        Zd = np.asarray(data.Z, dtype=float)
        if not 0 <= int(column) < Zd.shape[1]:
            raise ValueError(
                f"column must be one of the model's {Zd.shape[1]} covariate "
                f"columns (0 to {Zd.shape[1] - 1}), got {column!r}."
            )
        base = (
            Zd[0].copy()
            if Z is None
            else np.asarray(Z, dtype=float).reshape(-1).copy()
        )
        levels = np.unique(Zd[:, column])
        span = (
            levels if stresses is None else np.r_[levels, np.ravel(stresses)]
        )
        grid = np.linspace(float(np.min(span)), float(np.max(span)), 60)
        rows = np.tile(base, (grid.size, 1))
        rows[:, column] = grid
        if p is None:
            life = np.array([float(self.mean(r)) for r in rows])
            band = np.vstack(
                [self.mean_cb(r, alpha_ci=alpha_ci) for r in rows]
            )
            label = "Mean life"
        else:
            life = np.asarray(self.qf(np.full(grid.size, p), rows), float)
            band = np.asarray(
                self.quantile_cb(
                    np.full(grid.size, p), rows, alpha_ci=alpha_ci
                ),
                dtype=float,
            )
            label = "Median life" if p == 0.5 else f"B{100 * p:.3g} life"
        if ax is None:
            ax = plt.gcf().gca()
        level = f"{100 * (1 - alpha_ci):.3g}%"
        ax.plot(grid, life, color="C0", label=label)
        ax.fill_between(
            grid,
            band[:, 0],
            band[:, 1],
            color="C0",
            alpha=0.2,
            label=f"{level} bounds",
        )
        for k, stress in enumerate(levels):
            ax.axvline(stress, color="0.6", lw=0.8, ls=":")
            if not points:
                continue
            at = Zd[:, column] == stress
            x = np.asarray(data.x, dtype=float)
            x = x if x.ndim == 1 else x.max(axis=1)
            c = np.asarray(data.c)
            ax.plot(
                np.full(int(np.sum(at & (c != 1))), stress),
                x[at & (c != 1)],
                "o",
                color="C1",
                ms=4,
                label="failures" if k == 0 else None,
            )
            ax.plot(
                np.full(int(np.sum(at & (c == 1))), stress),
                x[at & (c == 1)],
                "o",
                mfc="none",
                color="C1",
                ms=4,
                label="censored" if k == 0 else None,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Stress" if column == 0 else f"Stress (column {column})")
        ax.set_ylabel(label)
        ax.legend()
        return ax
