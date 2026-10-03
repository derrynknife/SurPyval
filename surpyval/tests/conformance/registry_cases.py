"""Every case of the conformance registry (see ``registry.py``).

The univariate, competing-risks, recurrent, degradation and copula
cases, the DataFrame fit paths, the bound and option sweeps, and
how each case's fit is starved for the convergence test. The known
failures are applied in ``registry.py``.
"""

import functools
from dataclasses import replace

import numpy as np
import pandas as pd

import surpyval as sp
from surpyval import degradation as dg
from surpyval import multivariate as mv
from surpyval import recurrent as rc
from surpyval.beta import ml
from surpyval.tests.conformance.registry_families import (
    BIVARIATE,
    CAUSES,
    CAUSES_REGRESSION,
    COUNTING,
    COUNTING_CAUSES,
    COUNTING_REGRESSION,
    REFIT_PROPERTIES,
    UNI_FUNCTIONS,
    UNIVARIATE,
    Bound,
    Case,
    _accelerated_life_family,
    _beta,
    _fit,
    _fitted,
    _frailty_family,
    _regression_family,
    _seeded,
    _semi_parametric,
    _trees,
    continuous,
    discrete,
)
from surpyval.tests.conformance.registry_fixtures import (
    X_COP,
    X_CR,
    X_DESTR,
    X_PATH,
    X_PROC,
    X_REC,
    X_UNI,
    Z_CR,
    Z_REC,
    beta_geometric_data,
    binary_data,
    copula_data,
    cr_data,
    destructive_data,
    exact_event_data,
    lfp_count_data,
    lfp_data,
    mixture_data,
    offset_data,
    path_data,
    process_data,
    quantiles,
    recurrent_data,
    scramble,
    uni_data,
    uni_exact_data,
    unit_interval_data,
    xcnt_data,
    zi_data,
)
from surpyval.univariate import competing_risks as cr


# ---------------------------------------------------------------------------
# Univariate cases
# ---------------------------------------------------------------------------
def _gompertz_fun(x, *params):
    from autograd import numpy as anp

    return params[0] * (anp.exp(params[1] * x) - 1)


GOMPERTZ = sp.CustomDistribution(
    "ConformanceGompertz",
    _gompertz_fun,
    ["nu", "b"],
    ((0, None), (0, None)),
    (0, np.inf),
)


def _nonparametric(name, data=uni_data, **kw):
    fitter = getattr(sp, name)

    def from_surpyval_data(d):
        data = sp.SurpyvalData(d["x"], d["c"], d["n"], tl=d.get("tl"))
        return fitter.fit(x=data.x, c=data.c, n=data.n, t=data.t)

    return Case(
        name=kw.pop("case_name", name),
        fitters=kw.pop("fitters", (f"surpyval.{name}",)),
        model_class="surpyval.NonParametric",
        interface=UNIVARIATE,
        data=data,
        fit=_fit(fitter),
        functions=UNI_FUNCTIONS + ("qf",),
        x=X_UNI,
        rows=kw.pop("rows", ("x", "c", "n")),
        times=kw.pop("times", ("x",)),
        paths={"surpyval_data": from_surpyval_data},
        draw=lambda m, s: m.random(15, random_state=s),
        explicit_seed=True,
        jump_functions=("hf", "df"),
        rtol=1e-9,
        exclude={
            "df_hf_sf": "a step function: hf and df are the jumps between "
            "the query points (Conventions, 'Function Conventions')",
            "qf_ff": "a step function: qf is a generalised inverse, so "
            "qf(ff(x)) is the step at or below x, not x",
            **kw.pop("exclude", {}),
        },
        **kw,
    )


def _univariate():
    # Gauss and Galton are other names for Normal and LogNormal.
    alias = {"Normal": "Gauss", "LogNormal": "Galton"}
    out = []
    for name in (
        "Weibull",
        "Exponential",
        "Gamma",
        "LogNormal",
        "LogLogistic",
        "ExpoWeibull",
        "Rayleigh",
        "Normal",
        "Gumbel",
        "GumbelLEV",
        "Logistic",
    ):
        fitters = (f"surpyval.{name}",)
        if name in alias:
            fitters += (f"surpyval.{alias[name]}",)
        out.append(continuous(name, fitters=fitters))
    out.append(continuous("Uniform", data=uni_exact_data))
    out.append(
        continuous(
            "Beta",
            data=unit_interval_data,
            x=np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]),
            exclude={"units": "supported on [0, 1], not a scale family"},
        )
    )
    out.append(
        continuous(
            "Beta4",
            data=unit_interval_data,
            x=np.array([0.15, 0.2, 0.35, 0.5, 0.65, 0.75]),
            slow=frozenset({"*"}),
            exclude={
                "units": "its likelihood is unbounded (a shape below 1 "
                "puts infinite density at a support end), so the MLE has "
                "no maximum and stops where its search gave up, which "
                "depends on the units: shapes 1.00, 1.19 on the fixture, "
                "0.18, 0.18 on it times 7.3. The fit warns (No finite "
                "maximum, or not a verified maximum) and recommends "
                "how='MPS', which is unit-free: test_beta4_no_maximum.py "
                "(#385)"
            },
        )
    )
    # offset / limited failure population / zero-inflation, for the
    # half-line families where each is supported
    for name in ("Weibull", "Exponential", "Gamma", "LogNormal"):
        for variant, data in (
            ("offset", offset_data),
            ("lfp", lfp_data),
            ("zi", zi_data),
        ):
            out.append(
                continuous(
                    name,
                    case_name=f"{name}[{variant}]",
                    fitters=(),
                    data=data,
                    fixed={variant: True},
                    x=X_UNI + (5.0 if variant == "offset" else 0.0),
                    slow=(
                        frozenset() if name == "Weibull" else REFIT_PROPERTIES
                    ),
                    # the survival data to refit (#403), drawn from the
                    # lifetimes random draws (inf for a unit that never
                    # fails); one call, so a Generator seed is used once
                    draw=lambda m, s: m.random_data(15, random_state=s),
                    explicit_seed=True,
                )
            )
    out.append(
        continuous(
            "Weibull",
            case_name="Weibull[xcnt]",
            fitters=(),
            data=xcnt_data,
            rows=("x", "c", "n", "tl"),
            times=("x", "tl"),
            paths={},
        )
    )
    # A limited failure population in interval-censored counts, whose
    # default start ran p to its bound of 1 (#579)
    out.append(
        continuous(
            "Weibull",
            case_name="Weibull[lfp-counts]",
            fitters=(),
            data=lfp_count_data,
            fixed={"lfp": True},
            rows=("x", "c", "n"),
            times=("x",),
            paths={},
            slow=REFIT_PROPERTIES,
            draw=lambda m, s: m.random_data(15, random_state=s),
            explicit_seed=True,
        )
    )
    out.append(
        continuous(
            "ConformanceGompertz",
            fitter=GOMPERTZ,
            fitters=("surpyval.CustomDistribution",),
            # autograd through a user function: each refit takes ~0.4 s
            slow=REFIT_PROPERTIES,
        )
    )
    # discrete
    out.append(discrete("Poisson"))
    for name in ("Geometric", "NegativeBinomial", "DiscreteWeibull"):
        out.append(discrete(name, start=1))
    out.append(
        discrete(
            "BetaGeometric",
            start=1,
            data=beta_geometric_data,
        )
    )
    discretized = sp.Discretize(sp.Weibull)
    out.append(
        discrete(
            "Discretize(Weibull)",
            fitter=discretized,
            start=1,
            fitters=(
                "surpyval.Discretize",
                "surpyval.DiscretizedFitter",
            ),
        )
    )
    out.append(
        Case(
            name="Binomial",
            fitters=("surpyval.Binomial",),
            model_class="surpyval.Parametric",
            interface=UNIVARIATE,
            data=lambda: {"x": np.array([2, 3, 1, 4, 3, 2])},
            fit=lambda d: sp.Binomial.fit(**d, n_trials=5),
            functions=UNI_FUNCTIONS + ("qf",),
            x=np.arange(0.0, 6.0),
            continuous=False,
            rows=("x",),
            paths={
                "from_params": lambda d: sp.Binomial.from_params(
                    sp.Binomial.fit(**d, n_trials=5).params
                )
            },
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={
                "units": "counts of successes out of n_trials",
                "counts": "Binomial.fit takes no counts",
            },
        )
    )
    for name in ("Bernoulli", "FixedEventProbability"):
        fitter = getattr(sp, name)
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.{name}",),
                model_class="surpyval.Parametric",
                interface=UNIVARIATE,
                data=binary_data,
                fit=_fit(fitter),
                # Bernoulli's qf inverts its ff since #344; the flat
                # FixedEventProbability has no quantile to test.
                functions=(
                    ("sf", "ff", "Hf", "qf")
                    if name == "Bernoulli"
                    else ("sf", "ff", "Hf")
                ),
                # Bernoulli is defined at the outcomes 0 and 1 only.
                x=(
                    np.array([0.0, 1.0])
                    if name == "Bernoulli"
                    else np.array([0.0, 1.0, 2.0, 3.0, 5.0])
                ),
                continuous=False,
                rows=("x", "n"),
                paths={
                    "from_params": lambda d, f=fitter: f.from_params(
                        f.fit(**d).params
                    )
                },
                draw=lambda m, s: m.random(15, random_state=s),
                explicit_seed=True,
                exclude={
                    "units": "the outcomes are 0 and 1, not times",
                    "df_hf_sf": "only sf, ff and Hf are part of its model",
                },
            )
        )
    out.append(
        Case(
            name="ExactEventTime",
            fitters=("surpyval.ExactEventTime",),
            model_class="surpyval.Parametric",
            interface=UNIVARIATE,
            data=exact_event_data,
            fit=_fit(sp.ExactEventTime),
            functions=("sf", "ff", "Hf", "qf"),
            x=np.array([1.0, 3.0, 3.4, 3.6, 4.5, 7.0]),
            paths={
                "from_params": lambda d: sp.ExactEventTime.from_params(
                    sp.ExactEventTime.fit(**d).params
                )
            },
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={
                "df_hf_sf": "a point mass: no density or hazard rate",
                "qf_ff": "a point mass: ff takes only the values 0 and 1",
                "seed_global": "a point mass: every draw is T, whatever "
                "the seed",
            },
        )
    )
    no_data = "built from its parameters, not fitted to data"
    out.append(
        Case(
            name="Hypoexponential",
            fitters=("surpyval.Hypoexponential",),
            model_class="surpyval.Parametric",
            interface=UNIVARIATE,
            data=dict,
            fit=lambda d: sp.Hypoexponential.from_params([0.5, 1.5, 3.0]),
            functions=UNI_FUNCTIONS + ("qf",),
            x=np.array([0.1, 0.5, 1.0, 2.0, 4.0, 8.0]),
            rows=(),
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={p: no_data for p in REFIT_PROPERTIES},
        )
    )
    for name in ("NeverOccurs", "InstantlyOccurs"):
        cls = getattr(sp, name)
        out.append(
            Case(
                name=name,
                fitters=(),
                model_class=f"surpyval.{name}",
                interface=UNIVARIATE,
                data=dict,
                fit=lambda d, cls=cls: cls,
                functions=UNI_FUNCTIONS + ("qf",),
                x=np.array([0.0, 1.0, 5.0, 100.0]),
                rows=(),
                draw=lambda m, s: m.random(5, random_state=s),
                explicit_seed=True,
                exclude={
                    **{p: no_data for p in REFIT_PROPERTIES},
                    "df_hf_sf": "a point mass at 0 or infinity",
                    "qf_ff": "a point mass at 0 or infinity",
                    "seed_global": "a point mass: every draw is the same",
                },
            )
        )
    # non-parametric
    out.append(_nonparametric("KaplanMeier"))
    out.append(_nonparametric("NelsonAalen"))
    out.append(_nonparametric("FlemingHarrington"))
    out.append(
        _nonparametric(
            "Turnbull",
            data=xcnt_data,
            rows=("x", "c", "n", "tl"),
            times=("x", "tl"),
        )
    )
    # mixtures and splines
    out.append(
        Case(
            name="MixtureModel",
            fitters=("surpyval.MixtureModel",),
            model_class="surpyval.MixtureModel",
            interface=UNIVARIATE,
            data=mixture_data,
            fit=lambda d: _fit_mixture(d),
            functions=("sf", "ff", "df", "Hf"),
            x=np.array([1.0, 3.0, 5.0, 10.0, 22.0, 30.0, 45.0]),
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={
                "df_hf_sf": "MixtureModel has no hf (Conventions)",
                "qf_ff": "MixtureModel has no qf (Conventions)",
            },
        )
    )
    out.append(
        Case(
            name="RoystonParmar",
            fitters=("surpyval.RoystonParmar",),
            model_class="surpyval.RoystonParmarModel",
            interface=UNIVARIATE,
            data=uni_data,
            fit=_fit(sp.RoystonParmar),
            functions=UNI_FUNCTIONS + ("qf",),
            x=X_UNI,
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
        )
    )
    return out


def _fit_mixture(d):
    model = sp.MixtureModel(dist=sp.Weibull, m=2)
    model.fit(**d)
    return model


# ---------------------------------------------------------------------------
# Competing risks
# ---------------------------------------------------------------------------
def _competing_risks():
    def cr_from_df(d):
        df = pd.DataFrame({"x": d["x"], "e": d["e"], "n": d["n"]})
        return cr.CompetingRisks.fit_from_df(
            df, x_col="x", e_col="e", n_col="n"
        )

    def pcr_from_df(d):
        df = pd.DataFrame({"x": d["x"], "e": d["e"], "n": d["n"]})
        return cr.ParametricCompetingRisks.fit_from_df(
            df, x_col="x", e_col="e", n_col="n"
        )

    def crph_from_df(d, how="Cox"):
        df = pd.DataFrame(
            {"x": d["x"], "e": d["e"], "n": d["n"], "z0": d["Z"][:, 0]}
        )
        return cr.CompetingRisksProportionalHazards.fit_from_df(
            df, x_col="x", e_col="e", Z_cols=["z0"], n_col="n", model=how
        )

    out = [
        Case(
            name=f"CompetingRisks[{method}]",
            fitters=(
                ("surpyval.univariate.competing_risks.CompetingRisks",)
                if method == "Nelson-Aalen"
                else ()
            ),
            model_class="surpyval.univariate.competing_risks.CompetingRisks",
            interface=CAUSES,
            data=cr_data,
            fit=_fit(cr.CompetingRisks, how=method),
            functions=("sf", "ff", "Hf"),
            event_functions=("cif",),
            events=("a", "b"),
            x=X_CR,
            rows=("x", "e", "n"),
            paths=(
                {"fit_from_df": cr_from_df} if method == "Nelson-Aalen" else {}
            ),
            exclude=(
                {
                    "cif_sum": "documented: sf and ff report exp(-H) under "
                    "the Nelson-Aalen method, while cif is always "
                    "Aalen-Johansen, which sums to 1 - Kaplan-Meier (the "
                    "Kaplan-Meier case checks that)"
                }
                if method == "Nelson-Aalen"
                else {}
            ),
            rtol=1e-9,
        )
        for method in ("Nelson-Aalen", "Kaplan-Meier")
    ]
    out.append(
        Case(
            name="ParametricCompetingRisks",
            fitters=(
                "surpyval.univariate.competing_risks.ParametricCompetingRisks",
            ),
            model_class=(
                "surpyval.univariate.competing_risks.ParametricCompetingRisks"
            ),
            interface=CAUSES,
            data=cr_data,
            fit=_fit(cr.ParametricCompetingRisks),
            functions=("sf", "ff", "Hf"),
            event_functions=("cif",),
            events=("a", "b"),
            x=X_CR,
            rows=("x", "e", "n"),
            paths={"fit_from_df": pcr_from_df},
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
        )
    )
    for how in ("Cox", "Fine-Gray"):
        out.append(
            Case(
                name=f"CompetingRisksProportionalHazards[{how}]",
                fitters=(
                    (
                        "surpyval.univariate.competing_risks"
                        ".CompetingRisksProportionalHazards",
                    )
                    if how == "Cox"
                    else ()
                ),
                model_class=(
                    "surpyval.univariate.competing_risks"
                    ".CompetingRisksProportionalHazards"
                ),
                interface=CAUSES_REGRESSION,
                data=functools.partial(cr_data, True),
                fit=_fit(cr.CompetingRisksProportionalHazards, model=how),
                # The Fine-Gray form has no all-cause survival; its sf is a
                # cause's 1 - cif, reached through cif below.
                functions=("sf", "ff") if how == "Cox" else (),
                event_functions=("cif",),
                events=("a", "b"),
                x=X_CR,
                Z=Z_CR,
                rows=("x", "Z", "e", "n"),
                covariates="Z",
                paths={
                    "fit_from_df": functools.partial(crph_from_df, how=how)
                },
                rtol=1e-6,
                coefficients=lambda m: m.betas,
                intercept=True,
                exclude=(
                    {
                        "cif_sum": "Fine-Gray models each cause's "
                        "subdistribution separately; their CIFs need not "
                        "sum to one minus a survival"
                    }
                    if how == "Fine-Gray"
                    else {}
                ),
            )
        )
    out.append(
        Case(
            name="FineGray",
            fitters=("surpyval.univariate.competing_risks.FineGray",),
            model_class=(
                "surpyval.univariate.competing_risks.regression.fine_gray"
                ".FineGrayModel"
            ),
            interface=CAUSES_REGRESSION,
            data=functools.partial(cr_data, True),
            fit=_fit(cr.FineGray, event="a"),
            functions=("sf", "cif"),
            x=X_CR,
            Z=Z_CR,
            rows=("x", "Z", "e", "n"),
            covariates="Z",
            rtol=1e-6,
            coefficients=_beta,
            intercept=True,
            exclude={
                "cif_sum": "one cause of interest: sf is 1 - cif by "
                "definition, checked by sf_ff instead",
            },
        )
    )
    return out


# ---------------------------------------------------------------------------
# Recurrent events
# ---------------------------------------------------------------------------
def _recurrent_data_path(fitter, **fixed):
    def run(d):
        data = sp.handle_xicn(d["x"], d["i"], d["c"], d["n"])
        return fitter.fit_from_recurrent_data(data, **fixed)

    return run


def _counting_draw(m, s):
    return m.count_terminated_simulation(3, items=2, random_state=s)


def _timed_draw(m, s):
    # For a falling intensity (the CoxLewis fixture: beta = -0.021, so
    # cif(inf) = 6.04), which count termination refuses (#386).
    return m.time_terminated_simulation(60.0, items=2, random_state=s)


def _recurrent():
    out = []
    for name in ("HPP", "CrowAMSAA", "Duane", "CoxLewis"):
        fitter = getattr(rc, name)
        paths = {"fit_from_recurrent_data": _recurrent_data_path(fitter)}
        paths["from_params"] = lambda d, f=fitter: f.from_params(
            f.fit(**d).params
        )
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.recurrent.{name}",),
                model_class=(
                    "surpyval.recurrent.parametric.parametric_recurrence"
                    ".ParametricRecurrenceModel"
                ),
                interface=COUNTING,
                data=recurrent_data,
                fit=_fit(fitter),
                functions=("cif", "iif"),
                x=X_REC,
                rows=("x", "i", "c", "n"),
                paths=paths,
                draw=_timed_draw if name == "CoxLewis" else _counting_draw,
                explicit_seed=True,
            )
        )
    out.append(
        Case(
            name="NonParametricCounting",
            fitters=("surpyval.recurrent.NonParametricCounting",),
            model_class="surpyval.recurrent.NonParametricCounting",
            interface=COUNTING,
            data=recurrent_data,
            fit=_fit(rc.NonParametricCounting),
            functions=("mcf",),
            x=X_REC,
            rows=("x", "i", "c", "n"),
            paths={
                "fit_from_recurrent_data": _recurrent_data_path(
                    rc.NonParametricCounting
                )
            },
            rtol=1e-9,
        )
    )
    for name in ("ProportionalIntensityHPP", "ProportionalIntensityNHPP"):
        fitter = getattr(rc, name)
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.recurrent.{name}",),
                model_class=(
                    "surpyval.recurrent.regression.proportional_intensity"
                    ".ProportionalIntensityModel"
                ),
                interface=COUNTING_REGRESSION,
                data=functools.partial(recurrent_data, True),
                fit=_fit(fitter),
                functions=("cif", "iif"),
                x=X_REC,
                Z=Z_REC,
                rows=("x", "Z", "i", "c", "n"),
                covariates="Z",
                drops_missing_covariate=False,
                coefficients=lambda m: m.coeffs,
                # The baseline rate (HPP) or the Duane scale b absorbs a
                # constant.
                intercept=True,
                draw=lambda m, s: m.count_terminated_simulation(
                    3, items=2, random_state=s, Z=[0.5]
                ),
                explicit_seed=True,
            )
        )
    renewals = (
        ("GeneralizedRenewal", {}),
        ("GeneralizedRenewal", {"kijima": "ii"}),
        ("GeneralizedOneRenewal", {}),
        ("ARA", {"m": 1}),
        ("ARI", {"m": 1}),
    )
    for name, kw in renewals:
        fitter = getattr(rc, name)
        label = name + ("[kijima ii]" if kw.get("kijima") == "ii" else "")
        out.append(
            Case(
                name=label,
                fitters=(
                    (f"surpyval.recurrent.{name}",) if "[" not in label else ()
                ),
                model_class="surpyval.recurrent.RenewalModel",
                interface=COUNTING,
                data=recurrent_data,
                fit=_fit(fitter, **kw),
                # The MCF is simulated; a fixed seed makes it a function.
                functions=("mcf",),
                call_kwargs={"items": 30, "random_state": 1},
                x=X_REC,
                rows=("x", "i", "c", "n"),
                paths={
                    "fit_from_recurrent_data": _recurrent_data_path(
                        fitter, **kw
                    )
                },
                draw=lambda m, s: m.mcf(X_REC, items=5, random_state=s),
                explicit_seed=True,
                slow=REFIT_PROPERTIES,
            )
        )
    out.append(
        Case(
            name="CauseSpecificMCF",
            fitters=("surpyval.recurrent.CauseSpecificMCF",),
            model_class="surpyval.recurrent.CauseSpecificMCF",
            interface=COUNTING_CAUSES,
            data=functools.partial(recurrent_data, False, True),
            fit=_fit(rc.CauseSpecificMCF),
            functions=(),
            event_functions=("mcf",),
            events=("a", "b"),
            x=X_REC,
            rows=("x", "i", "c", "n", "e"),
            rtol=1e-9,
        )
    )
    out.append(
        Case(
            name="CauseSpecificNHPP",
            fitters=("surpyval.recurrent.CauseSpecificNHPP",),
            model_class="surpyval.recurrent.CauseSpecificNHPP",
            interface=COUNTING_CAUSES,
            data=functools.partial(recurrent_data, False, True),
            fit=_fit(rc.CauseSpecificNHPP),
            functions=(),
            event_functions=("cif", "iif"),
            events=("a", "b"),
            x=X_REC,
            rows=("x", "i", "c", "n", "e"),
        )
    )
    # A count above one is refused on an observed recurrent event (the
    # xicn convention: several events at one instant are not a count).
    no_counts = {
        "counts": "the xicn format refuses a count above 1 on an event row"
    }
    return [replace(c, exclude={**no_counts, **c.exclude}) for c in out]


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------
def _degradation():
    def da(path):
        return Case(
            name=f"DegradationAnalysis[{path}]",
            fitters=(
                ("surpyval.degradation.DegradationAnalysis",)
                if path == "linear"
                else ()
            ),
            model_class="surpyval.degradation.DegradationModel",
            interface=UNIVARIATE,
            data=path_data,
            fit=_fit(dg.DegradationAnalysis, threshold=150.0, path=path),
            functions=UNI_FUNCTIONS + ("qf",),
            x=X_PATH,
            rows=("x", "y", "i"),
            paths={
                "fit_from_df": lambda d: dg.DegradationAnalysis.fit_from_df(
                    pd.DataFrame(d), threshold=150.0, path=path
                )
            },
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={"counts": "degradation readings carry no counts"},
        )

    out = [da("linear"), da("power")]
    for name in ("WienerProcess", "GammaProcess"):
        fitter = getattr(dg, name)
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.degradation.{name}",),
                model_class=f"surpyval.degradation.{name}Model",
                interface=UNIVARIATE,
                data=process_data,
                fit=_fit(fitter, threshold=100.0),
                functions=UNI_FUNCTIONS + ("qf",),
                x=X_PROC,
                rows=("x", "y", "i"),
                paths={
                    "fit_from_df": lambda d, f=fitter: f.fit_from_df(
                        pd.DataFrame(d), threshold=100.0
                    )
                },
                draw=lambda m, s: m.random(15, random_state=s),
                explicit_seed=True,
                exclude={"counts": "degradation readings carry no counts"},
            )
        )
    out.append(
        Case(
            name="DestructiveDegradation",
            fitters=("surpyval.degradation.DestructiveDegradation",),
            model_class="surpyval.degradation.DestructiveDegradationModel",
            interface=UNIVARIATE,
            data=destructive_data,
            fit=_fit(dg.DestructiveDegradation, threshold=20.0),
            functions=("sf", "ff", "df", "Hf"),
            x=X_DESTR,
            rows=("x", "y"),
            exclude={
                "counts": "destructive readings carry no counts",
                "df_hf_sf": "DestructiveDegradationModel has no hf",
                "qf_ff": "DestructiveDegradationModel has no qf",
            },
        )
    )
    out.append(
        Case(
            name="InducedFailureDistribution",
            fitters=(),
            model_class="surpyval.degradation.InducedFailureDistribution",
            interface=UNIVARIATE,
            data=path_data,
            fit=lambda d: dg.DegradationAnalysis.fit(
                **d, threshold=150.0
            ).induced_life(n_samples=400, random_state=0),
            functions=("sf", "ff"),
            x=X_PATH,
            rows=("x", "y", "i"),
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={
                "counts": "degradation readings carry no counts",
                "df_hf_sf": "a Monte Carlo distribution: no hf or df",
                "qf_ff": "an empirical distribution of simulated lives",
            },
        )
    )
    return out


# ---------------------------------------------------------------------------
# Copulas
# ---------------------------------------------------------------------------
_COPULA_FAMILIES = (
    "Independence",
    "Clayton",
    "Gumbel",
    "Frank",
    "Gaussian",
    "Joe",
    "AMH",
    "StudentT",
)


# The rotation option (#157), on one family: (case name, family, rotation)
_ROTATED_COPULAS = (("ClaytonCopula[rotation=180]", "Clayton", 180),)


def _copulas():
    out = []
    plain = [(f"{name}Copula", name, 0) for name in _COPULA_FAMILIES]
    for case_name, name, rotation in plain + list(_ROTATED_COPULAS):
        fitter = getattr(mv, name)
        rotated = {"rotation": rotation} if rotation else {}

        def from_params(d, f=fitter, r=rotated):
            m = f.fit(**d, margins=[sp.Weibull, sp.Weibull], **r)
            return f.from_params(m.params, margins=m.margins, **r)

        out.append(
            Case(
                name=case_name,
                fitters=(f"surpyval.multivariate.{name}",),
                model_class="surpyval.multivariate.CopulaModel",
                interface=BIVARIATE,
                data=copula_data,
                fit=_fit(fitter, margins=[sp.Weibull, sp.Weibull], **rotated),
                functions=("sf", "cdf", "pdf"),
                x=X_COP,
                rows=("x", "n"),
                paths={"from_params": from_params},
                draw=lambda m, s: m.random(15, random_state=s),
                explicit_seed=True,
            )
        )
    return out


CASES: list[Case] = (
    _univariate()
    + _regression_family()
    + _accelerated_life_family()
    + _frailty_family()
    + _semi_parametric()
    + _trees()
    + _competing_risks()
    + _recurrent()
    + _degradation()
    + _copulas()
)

# Fits whose optimum moves by more than 1e-4 (relative) when the data are
# rescaled or reordered: the optimiser stops at its own tolerance, and a
# flat likelihood turns that into a larger change in the predictions.
_LOOSE: tuple[str, ...] = ("LogNormalAH", "WeibullAL[InverseExponential]")
_LOOSE += ("CauseSpecificNHPP", "GammaProcess")
# LogNormalAFT agrees to 1e-4 with the numpy/scipy of the development
# environment but moved by 2.3e-4 in sf under the newer ones CI installs
# (numpy 2.5, scipy 1.18): the same optimiser-tolerance effect.
_LOOSE += ("LogNormalAFT",)
# LogisticAH: counts against repeated rows agree to 1e-6 here, but CI's
# Python 3.11 runner (the same numpy 2.4 / scipy 1.17) stops 1.7e-4 apart
# in ff at a value near 0 (-0.0194), on every run: the same effect.
_LOOSE += ("LogisticAH",)
CASES = [replace(c, rtol=1e-3) if c.name in _LOOSE else c for c in CASES]


# ---------------------------------------------------------------------------
# fit_from_df paths (#511): every fitter reads a DataFrame, and gives the
# model its fit gives on the same arrays. Each family names the columns as
# its fit_from_df does (utils/dataframe.py).
# ---------------------------------------------------------------------------
def _frame(d, keys):
    """The per-row entries ``keys`` of a fixture as DataFrame columns (a
    2-D entry as columns ``<key>0``, ``<key>1``, ...), and the rest."""
    cols, rest = {}, {}
    for key, value in d.items():
        if key not in keys:
            rest[key] = value
            continue
        value = np.asarray(value)
        if value.ndim == 2:
            for k in range(value.shape[1]):
                cols[f"{key}{k}"] = value[:, k]
        else:
            cols[key] = value
    return pd.DataFrame(cols), rest


def _column_names(df, key):
    return [k for k in df.columns if k[: len(key)] == key and k != key]


def _uni_df(fitter, **fixed):
    """``x``, ``c``, ``n``, ``tl`` (and ``xl`` / ``xr`` for interval
    rows), named as the univariate fit_from_df names them."""

    def run(d):
        df, rest = _frame(d, ("x", "c", "n", "tl", "tr"))
        if "x" in df:
            names = {"x": "x"}
        else:
            df = df.rename(columns={"x0": "xl", "x1": "xr"})
            names = {"xl": "xl", "xr": "xr"}
        names |= {k: k for k in ("c", "n", "tl", "tr") if k in df}
        return fitter.fit_from_df(df, **names, **fixed, **rest)

    return run


def _col_df(fitter, keys=("x", "i", "c", "n", "e"), **fixed):
    """The ``<key>_col`` names (and ``Z_cols``) of the recurrent,
    regression and competing-risks fit_from_df."""

    def run(d):
        df, rest = _frame(d, keys + ("Z",))
        names = {f"{k}_col": k for k in keys if k in df}
        if "Z" in d:
            names["Z_cols"] = _column_names(df, "Z")
        return fitter.fit_from_df(df, **names, **fixed, **rest)

    return run


def _copula_df(fitter, **fixed):
    def run(d):
        df, rest = _frame(d, ("x", "n"))
        return fitter.fit_from_df(
            df, x_cols=_column_names(df, "x"), n_col="n", **fixed, **rest
        )

    return run


def _df_paths():
    """Case name -> its fit_from_df path, for the cases that had none."""
    paths = {
        name: _uni_df(getattr(sp, name))
        for name in ("KaplanMeier", "NelsonAalen", "FlemingHarrington")
    }
    paths["Turnbull"] = _uni_df(sp.Turnbull)
    paths["Weibull[xcnt]"] = _uni_df(sp.Weibull)
    paths["Weibull[lfp-counts]"] = _uni_df(sp.Weibull, lfp=True)
    paths["RoystonParmar"] = _uni_df(sp.RoystonParmar)
    paths["MixtureModel"] = _uni_df(sp.MixtureModel, dist=sp.Weibull, m=2)
    paths["Binomial"] = _uni_df(sp.Binomial, n_trials=5)
    for name in ("Bernoulli", "FixedEventProbability", "ExactEventTime"):
        paths[name] = _uni_df(getattr(sp, name))
    for kind in ("weibull", "exponential", "non-parametric"):
        paths[f"SurvivalTree[{kind}]"] = _seeded(
            _col_df(ml.SurvivalTree, ("x", "c", "n"), kind=kind)
        )
    paths["RandomSurvivalForest"] = _seeded(
        _col_df(ml.RandomSurvivalForest, ("x", "c", "n"), n_trees=3)
    )
    paths["CoxPH[strata]"] = _col_df(
        sp.CoxPH, ("x", "c", "n", "strata"), tie_method="efron"
    )
    paths["CompetingRisks[Kaplan-Meier]"] = _col_df(
        cr.CompetingRisks, ("x", "e", "n"), how="Kaplan-Meier"
    )
    paths["FineGray"] = _col_df(cr.FineGray, ("x", "e", "n"), event="a")
    for name in ("HPP", "CrowAMSAA", "Duane", "CoxLewis"):
        paths[name] = _col_df(getattr(rc, name))
    paths["NonParametricCounting"] = _col_df(rc.NonParametricCounting)
    for name in ("ProportionalIntensityHPP", "ProportionalIntensityNHPP"):
        paths[name] = _col_df(getattr(rc, name))
    paths["GeneralizedRenewal"] = _col_df(rc.GeneralizedRenewal)
    paths["GeneralizedRenewal[kijima ii]"] = _col_df(
        rc.GeneralizedRenewal, kijima="ii"
    )
    paths["GeneralizedOneRenewal"] = _col_df(rc.GeneralizedOneRenewal)
    paths["ARA"] = _col_df(rc.ARA, m=1)
    paths["ARI"] = _col_df(rc.ARI, m=1)
    paths["CauseSpecificMCF"] = _col_df(rc.CauseSpecificMCF)
    paths["CauseSpecificNHPP"] = _col_df(rc.CauseSpecificNHPP)
    paths["DestructiveDegradation"] = lambda d: (
        dg.DestructiveDegradation.fit_from_df(
            pd.DataFrame(d), x_col="x", y_col="y", threshold=20.0
        )
    )
    for name in _COPULA_FAMILIES:
        paths[f"{name}Copula"] = _copula_df(
            getattr(mv, name), margins=[sp.Weibull, sp.Weibull]
        )
    for case_name, name, rotation in _ROTATED_COPULAS:
        paths[case_name] = _copula_df(
            getattr(mv, name),
            margins=[sp.Weibull, sp.Weibull],
            rotation=rotation,
        )
    return paths


def _add_df_path(case, paths=_df_paths()):
    if case.name not in paths:
        return case
    assert "fit_from_df" not in case.paths, case.name
    return replace(case, paths={**case.paths, "fit_from_df": paths[case.name]})


CASES = [_add_df_path(c) for c in CASES]

# Cases with no fit_from_df path, and why (checked by test_fit_paths.py).
NO_DF_PATH: dict[str, str] = {
    "Hypoexponential": "built from its parameters: its fit refuses data",
    "NeverOccurs": "a fixed model, not fitted",
    "InstantlyOccurs": "a fixed model, not fitted",
    "InducedFailureDistribution": "derived from a fitted degradation model",
}


# ---------------------------------------------------------------------------
# Options (test_options.py): the uncertainty methods of each case, the
# values its interp= takes and the estimation options of its fit
# ---------------------------------------------------------------------------
_ON_ALL = ("sf", "ff", "Hf", "hf", "df")
_ON_SURVIVAL = ("sf", "ff", "Hf")

# Parametric fits with no covariance to bound with: cb and param_cb
# raise ValueError, as documented ("Only MLE has confidence bounds"; a
# closed-form estimate or a model built from its parameters carries none).
_NO_COVARIANCE = (
    "ExactEventTime",
    "Hypoexponential",
)
# The probability models: param_cb bounds p from the counts of events and
# trials (exact Clopper-Pearson by default, Wald and likelihood ratio as
# options); cb, quantile_cb and mean_cb raise, pointing to it (#580).
_PROBABILITY_MODELS = ("Binomial", "Bernoulli", "FixedEventProbability")
# The likelihood-ratio search runs pointwise, so it is swept at three
# times, and only in the full suite. Rayleigh, Geometric and Uniform
# joined in #421 (a df bound stalled on the far side of the estimate;
# the Uniform's search stalled at the support's edge), then
# NegativeBinomial and ExpoWeibull (profiles that did not follow their
# valleys, bounds that were rounding noise where they are infinite, and
# bands that were not nested). Beta4's MLE has no maximum (#385).
_LR_X = {
    "Weibull": np.array([4.0, 8.0, 13.0]),
    "Rayleigh": np.array([3.2, 8.0, 14.6]),
    "Geometric": np.array([2.0, 5.0, 8.0]),
    "Uniform": np.array([3.2, 8.0, 14.6]),
    "NegativeBinomial": np.array([2.0, 5.0, 8.0]),
    # One time for ExpoWeibull: its searches in a three-parameter valley
    # take seconds each, and three times tripled a 17-minute sweep. The
    # tail (13) is where its bands were hardest (the hf nesting in #421).
    "ExpoWeibull": np.array([13.0]),
}
_LR_NIGHTLY = {"NegativeBinomial", "ExpoWeibull"}
# The regression models' likelihood-ratio bounds (#583), swept on one
# case of each kind of covariate link, at two times (with the case's
# first two covariate rows): a multiplier on the hazard, on the time, on
# the odds, an additive hazard, and the accelerated life model of #583.
_WHOLE_LINE = ("Normal", "Gumbel", "Logistic")
_REGRESSION_LR_X = {
    "WeibullPH": (2.0, 8.0),
    "LogNormalAFT": (2.0, 8.0),
    "WeibullPO": (2.0, 8.0),
    "WeibullAH": (2.0, 8.0),
    "WeibullAL[PowerExponential]": (3.0, 10.0),
}


# Fits with no parameter covariance by design, so no Wald bounds: the
# Uniform's MLE sits on the sample extremes, where the likelihood's
# curvature says nothing about its uncertainty (#460).
_NO_WALD = {"Uniform"}


def _parametric_bounds(case):
    on = tuple(f for f in _ON_ALL if f in case.functions)
    out = []
    if case.name not in _NO_WALD:
        out += [
            Bound("cb", on=on, kwargs={"method": "wald"}, label="cb[wald]"),
            Bound(
                "param_cb",
                kind="param",
                kwargs={"method": "wald"},
                label="param_cb[wald]",
            ),
        ]
        # Bounds on the B-lives and the mean (#494)
        out += [
            Bound(
                "quantile_cb",
                point="qf",
                kwargs={"method": "wald"},
                label="quantile_cb[wald]",
            ),
            Bound(
                "mean_cb",
                kind="summary",
                kwargs={"method": "wald"},
                query=((),),
                # Documented: with support ends among the parameters the
                # Wald bound is on the mean's own scale
                in_range=case.name != "Beta4",
                label="mean_cb[wald]",
            ),
        ]
    # The likelihood-ratio search is swept on the cases in _LR_X only.
    # The ExpoWeibull's and NegativeBinomial's sweeps (searches in
    # multi-parameter valleys, seconds each) took about ten minutes on
    # four cores, so they run nightly, with the calibration studies; the
    # other four sweep in every run, and test_likelihood_ratio_edges.py
    # checks those two families' edges and valleys directly (#421).
    # (Documented: it is not available for offset, limited-failure or
    # zero-inflated models.)
    if case.name not in _LR_X:
        return tuple(out)
    x = _LR_X[case.name]
    lr = dict(
        wald=False,
        nan_ok=True,
        rtol=1e-3,
        slow=True,
        nightly=case.name in _LR_NIGHTLY,
    )
    out.append(
        Bound(
            "cb",
            on=on,
            kwargs={"method": "lr"},
            query=tuple(x),
            label="cb[lr]",
            **lr,
        )
    )
    out.append(
        Bound(
            "param_cb",
            kind="param",
            kwargs={"method": "lr"},
            label="param_cb[lr]",
            **lr,
        )
    )
    if case.continuous:
        # (a discrete quantile's bound inverts the band on ff at every
        # count up to it: a likelihood-ratio search at each)
        out.append(
            Bound(
                "quantile_cb",
                point="qf",
                kwargs={"method": "lr"},
                query=(0.1, 0.5, 0.9),
                label="quantile_cb[lr]",
                **lr,
            )
        )
        out.append(
            Bound(
                "mean_cb",
                kind="summary",
                kwargs={"method": "lr"},
                query=((),),
                label="mean_cb[lr]",
                **lr,
            )
        )
    return tuple(out)


def _nonparametric_bounds(case):
    out = []
    for bound_type in ("exp", "normal"):
        for interp in ("step", "linear", "cubic"):
            out.append(
                Bound(
                    "cb",
                    on=_ON_SURVIVAL,
                    kwargs={"bound_type": bound_type, "interp": interp},
                    in_range=bound_type == "exp",
                    nan_ok=True,
                    label=f"cb[{bound_type},{interp}]",
                )
            )
        for method in ("hall-wellner", "nair"):
            out.append(
                Bound(
                    "band",
                    kwargs={"method": method, "bound_type": bound_type},
                    sides=False,
                    # Not a Wald interval: a whole path rarely stays inside
                    # a narrow strip, so the critical value does not go to
                    # 0 as alpha_ci -> 1 (0.27 at 1 - 1e-6, Hall-Wellner
                    # over the whole range), and the band does not close
                    # onto the estimate. (It used to be left out because
                    # the search at alpha_ci -> 1 did not end, #420.)
                    wald=False,
                    in_range=bound_type == "exp",
                    nan_ok=True,
                    label=f"band[{method},{bound_type}]",
                )
            )
        out.append(
            Bound(
                "quantile_cb",
                point="qf",
                kwargs={"bound_type": bound_type},
                sides=False,
                # Documented: the upper end is NaN where the interval is
                # open to the right.
                nan_ok=True,
                label=f"quantile_cb[{bound_type}]",
            )
        )
    # The band's default scale (#390), which the pointwise bounds lack.
    for method in ("hall-wellner", "nair"):
        out.append(
            Bound(
                "band",
                kwargs={"method": method, "bound_type": "arcsine"},
                sides=False,
                wald=False,
                nan_ok=True,
                label=f"band[{method},arcsine]",
            )
        )
    out.append(
        Bound(
            "bootstrap_cb",
            kwargs={"n_boot": 40, "random_state": 1},
            wald=False,
            nan_ok=True,
            # Each resample reruns the Turnbull EM.
            slow=case.name == "Turnbull",
        )
    )
    out.append(Bound("mean_cb", kind="summary", sides=False, query=((),)))
    out.append(
        Bound("rmst", kind="summary", sides=False, query=((6.0,), (12.0,)))
    )
    return tuple(out)


def _mcf_bounds(per_cause=False):
    return tuple(
        Bound(
            "mcf_cb",
            point="mcf",
            kwargs={"bound_type": bound_type, "interp": interp},
            in_range=bound_type == "exp",
            nan_ok=True,
            per_cause=per_cause,
            label=f"mcf_cb[{bound_type},{interp}]",
        )
        for bound_type in ("exp", "normal")
        for interp in ("step", "linear")
    )


_PARAM_CB = Bound("param_cb", kind="param")
_BOOT = {"n_boot": 20, "random_state": 1}


def _bounds(case):
    """The uncertainty methods of ``case``'s model (see :class:`Bound`)."""
    cls = case.model_class.rsplit(".", 1)[-1]
    if cls == "Parametric":
        if case.name in _NO_COVARIANCE:
            return ()
        if case.name in _PROBABILITY_MODELS:
            return tuple(
                Bound(
                    "param_cb",
                    kind="param",
                    kwargs={"method": method},
                    label=f"param_cb[{method}]",
                    # Clopper-Pearson's and the likelihood-ratio interval
                    # do not close onto the estimate as alpha_ci -> 1.
                    wald=method == "wald",
                )
                for method in ("exact", "wald", "lr")
            )
        return _parametric_bounds(case)
    if cls == "NonParametric":
        return _nonparametric_bounds(case)
    if cls == "RoystonParmarModel":
        return (Bound("cb", on=_ON_SURVIVAL),)
    if cls == "ParametricRegressionModel":
        wald = (
            Bound("cb", on=_ON_ALL),
            _PARAM_CB,
            Bound(
                "quantile_cb",
                point="qf",
                kwargs={"method": "wald"},
                label="quantile_cb[wald]",
                # A baseline on the whole line has its quantile bounded on
                # its own scale, where an interval at alpha_ci -> 1 is
                # the estimate +- 1.25e-6 standard errors: more than 1e-5
                # of a quantile near 0 (GumbelAFT's qf(0.05), -0.147).
                rtol=1e-4 if case.name.startswith(_WHOLE_LINE) else 1e-8,
            ),
        )
        if case.name not in _REGRESSION_LR_X:
            return wald
        lr = dict(
            kwargs={"method": "lr"},
            wald=False,
            nan_ok=True,
            rtol=1e-3,
            slow=True,
        )
        return (
            *wald,
            Bound(
                "cb",
                on=_ON_ALL,
                query=_REGRESSION_LR_X[case.name],
                label="cb[lr]",
                **lr,
            ),
            Bound("param_cb", kind="param", label="param_cb[lr]", **lr),
            Bound("quantile_cb", point="qf", label="quantile_cb[lr]", **lr),
        )
    if cls in ("FrailtyModel", "ProportionalOddsModel"):
        # The profile-likelihood interval (#617), slow as the regression
        # models' likelihood-ratio bounds are.
        lr = Bound(
            "param_cb",
            kind="param",
            kwargs={"method": "lr"},
            label="param_cb[lr]",
            wald=False,
            nan_ok=True,
            rtol=1e-3,
            slow=True,
        )
        return (_PARAM_CB, lr)
    if cls == "CoxFrailtyModel":
        return (_PARAM_CB,)
    if cls == "BuckleyJamesModel":
        return (
            Bound(
                "bootstrap_ci",
                kind="coef",
                kwargs=_BOOT,
                sides=False,
                wald=False,
            ),
        )
    if cls == "ParametricRecurrenceModel":
        return (
            Bound("cif_cb", point="cif"),
            Bound("iif_cb", point="iif"),
            Bound("mtbf_cb", point="mtbf"),
            _PARAM_CB,
        )
    if cls == "ProportionalIntensityModel":
        return (
            Bound("cif_cb", point="cif"),
            Bound("iif_cb", point="iif"),
            _PARAM_CB,
        )
    if cls == "RenewalModel":
        return (_PARAM_CB,)
    if cls == "NonParametricCounting":
        return _mcf_bounds()
    if cls == "CauseSpecificMCF":
        return _mcf_bounds(per_cause=True)
    if cls == "DegradationModel":
        # A new unit's first three readings, still below the threshold.
        d = path_data()
        unit = (d["x"][:3], d["y"][:3])
        return (
            Bound(
                "cb",
                on=_ON_SURVIVAL,
                kwargs={"method": "analytic"},
                label="cb[analytic]",
            ),
            Bound(
                "cb",
                on=_ON_SURVIVAL,
                kwargs={
                    "method": "bootstrap",
                    "n_boot": 10,
                    "random_state": 1,
                },
                wald=False,
                slow=True,
                label="cb[bootstrap]",
            ),
            Bound(
                "predict_rul",
                kind="rul",
                kwargs={"n_samples": 4000, "random_state": 1},
                sides=False,
                # Documented: a remaining life is negative once the unit
                # has most likely crossed the threshold.
                in_range=False,
                query=(unit,),
                rtol=1e-2,
            ),
        )
    if cls == "DestructiveDegradationModel":
        # Every call refits the model n_boot times.
        return (
            Bound(
                "cb",
                on=("sf", "ff"),
                kwargs={"n_boot": 10, "random_state": 1},
                wald=False,
                slow=True,
            ),
        )
    if cls in ("WienerProcessModel", "GammaProcessModel"):
        return (
            Bound(
                "predict_rul",
                kind="rul",
                sides=False,
                query=((0.0,), (40.0,), (80.0,)),
                rtol=1e-6,
            ),
        )
    if cls == "CopulaModel":
        # The joint sf and the joint CDF are not complements: one sweep
        # each, so that cb_transform does not read one as 1 - the other
        # (#540). The fixture's AMH estimate is on its bound, theta = 1
        # (the data's Kendall's tau is past the family's 1/3), where no
        # Wald bound exists: NaN, with a warning, as documented.
        nan_ok = case.name == "AMHCopula"
        return (
            Bound("cb", on=("sf",), nan_ok=nan_ok),
            Bound("cb", on=("ff",), nan_ok=nan_ok, label="cb[ff]"),
            replace(_PARAM_CB, nan_ok=nan_ok),
        )
    return ()


# interp= values: the documented ones, and the other scipy interp1d
# kinds the non-parametric functions are documented to accept.
_NP_INTERP: tuple[str, ...] = ("step", "linear", "cubic")
_NP_INTERP += ("nearest", "zero", "slinear", "quadratic", "previous", "next")
_INTERP = {
    "KaplanMeier": _NP_INTERP,
    "NelsonAalen": _NP_INTERP,
    "FlemingHarrington": _NP_INTERP,
    "Turnbull": _NP_INTERP,
    "NonParametricCounting": ("step", "linear"),
    "CauseSpecificMCF": ("step", "linear"),
    # Its baselines are steps: interp= takes "step" only (#416).
    "CompetingRisksProportionalHazards[Cox]": ("step",),
}

# Estimation options. ``estimators`` are swept on the case's fixture;
# the agreement sweep adds the values in ``_LARGE_ONLY`` (the method of
# moments needs uncensored data, so it is refused on the fixtures) and
# refits the large sample of ``large`` (a function of the fitted model).
N_LARGE = 1000
_U_LARGE = (np.arange(1, N_LARGE + 1) - 0.5) / N_LARGE


def _quantile_sample(model):
    """A deterministic 'sample' of the model: its quantiles."""
    return {"x": np.asarray(model.qf(_U_LARGE), float)}


def _parametric_estimators(case):
    fitter = getattr(sp, case.name)
    how = ["MLE", "MPS", "MSE", "MPP"]
    if fitter.discrete:
        how.remove("MPS")  # documented: MPS needs a continuous CDF
    if not fitter.supports_mpp:
        how.remove("MPP")  # documented: not fitted by probability plotting
    return {"how": tuple(how)}


def _censored_sample(model):
    # Weibull(10, 2) quantiles, every fifth one right censored.
    x = quantiles(N_LARGE)
    return {"x": x, "c": (np.arange(N_LARGE) % 5 == 4).astype(int)}


def _cox_sample(model):
    size = 400
    z0 = np.tile([0.0, 1.0], size // 2)
    z1 = np.round(np.linspace(-1.0, 1.0, size), 3)
    life = quantiles(size, 1.0, 2.0)[scramble(size)]
    # Rounded to one decimal, so there are ties for the methods to handle.
    x = np.round(10.0 * np.exp(-0.5 * z0 + 0.3 * z1) * life, 1)
    c = (np.arange(size) % 7 == 0).astype(int)
    return {"x": x, "Z": np.column_stack([z0, z1]), "c": c}


def _cr_sample(model, with_Z=True):
    size = 480
    x = np.round(quantiles(size, 10.0, 1.5), 1)
    e = np.array(["a", "b", "a", None] * (size // 4), dtype=object)
    d = {"x": x, "e": e[scramble(size)]}
    if with_Z:
        Z = np.tile([0.0, 1.0, 1.0], size // 3)[:, None]
        d["Z"] = Z[scramble(size)]
    return d


def _recurrent_sample(model):
    data = model.time_terminated_simulation_data(
        60.0, items=40, random_state=1
    )
    return {"x": data.x, "i": data.i, "c": data.c, "n": data.n}


def _copula_sample(model):
    return {"x": model.random(800, random_state=1)}


_PLAIN_CONTINUOUS: tuple[str, ...] = (
    "Weibull",
    "Exponential",
    "Gamma",
    "LogNormal",
)
_PLAIN_CONTINUOUS += ("LogLogistic", "ExpoWeibull", "Rayleigh", "Normal")
_PLAIN_CONTINUOUS += ("Gumbel", "GumbelLEV", "Logistic", "Uniform")
_PLAIN_DISCRETE: tuple[str, ...] = ("Poisson", "Geometric", "NegativeBinomial")
_PLAIN_DISCRETE += ("DiscreteWeibull",)
# The agreement sweeps run on a pull request for these; the rest are slow.
_FAST_ESTIMATORS: tuple[str, ...] = (
    "Weibull",
    "LogNormal",
    "Poisson",
    "Turnbull",
    "CoxPH",
)
_FAST_ESTIMATORS += ("GumbelCopula", "CrowAMSAA")


def _estimators(case):
    """(estimators, large-only values, large sample) of ``case``."""
    name = case.name
    if name in _PLAIN_CONTINUOUS + _PLAIN_DISCRETE:
        return (
            _parametric_estimators(case),
            {"how": ("MOM",)},
            _quantile_sample,
        )
    if name == "Turnbull":
        est = ("Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington")
        return {"turnbull_estimator": est}, {}, _censored_sample
    if name == "CoxPH":
        methods = ("breslow", "efron", "exact", "kalbfleisch-prentice")
        return {"tie_method": methods}, {}, _cox_sample
    if name == "CompetingRisksProportionalHazards[Cox]":
        return {"tie_method": ("efron", "breslow")}, {}, _cr_sample
    if name == "CompetingRisks[Nelson-Aalen]":
        methods = ("Nelson-Aalen", "Kaplan-Meier")
        return (
            {"how": methods},
            {},
            functools.partial(_cr_sample, with_Z=False),
        )
    if name == "ParametricCompetingRisks":
        return {"how": ("MLE", "MPS", "MSE", "MPP")}, {}, None
    if name in ("CrowAMSAA", "Duane", "CoxLewis"):
        return {"how": ("MLE", "MSE")}, {}, _recurrent_sample
    if name == "CauseSpecificNHPP":
        return {"how": ("MLE", "MSE")}, {}, None
    if case.model_class == "surpyval.multivariate.CopulaModel":
        return {"how": ("IFM", "MLE")}, {}, _copula_sample
    if name == "DegradationAnalysis[linear]":
        est = {
            "how": ("MLE", "MPS", "MSE", "MPP"),
            "population_method": ("moments", "reml"),
        }
        return est, {}, None
    return {}, {}, None


# The bound sweeps run on a pull request for one or two cases of each
# family (each regression bound recomputes a numerical Hessian); the rest
# are slow.
_FAST_BOUNDS: tuple[str, ...] = (
    "Weibull",
    "Poisson",
    "Weibull[lfp]",
    "KaplanMeier",
)
_FAST_BOUNDS += ("Turnbull", "RoystonParmar", "WeibullPH", "WeibullFrailty")
_FAST_BOUNDS += ("HPP", "CrowAMSAA", "ProportionalIntensityHPP")
_FAST_BOUNDS += ("GeneralizedRenewal", "NonParametricCounting")
_FAST_BOUNDS += ("CauseSpecificMCF", "DegradationAnalysis[linear]")
_FAST_BOUNDS += ("WienerProcess", "ClaytonCopula")


def _with_options(case):
    estimators, large_only, large = _estimators(case)
    bounds = _bounds(case)
    if case.name not in _FAST_BOUNDS:
        bounds = tuple(replace(b, slow=True) for b in bounds)
    slow = case.slow
    if large is not None and case.name not in _FAST_ESTIMATORS:
        slow = slow | {"estimators_agree"}
    exclude = case.exclude
    if case.name in _NO_COVARIANCE:
        exclude = {
            **exclude,
            "cb_declared": "no covariance: cb and param_cb raise "
            "ValueError, as documented",
        }
    if case.name in _PROBABILITY_MODELS:
        exclude = {
            **exclude,
            "cb_declared": "#580: cb, quantile_cb and mean_cb raise "
            "ValueError, as documented: the bounds are on p (param_cb, "
            "swept)",
        }
    return replace(
        case,
        exclude=exclude,
        bounds=bounds,
        interp=_INTERP.get(case.name, ()),
        estimators=estimators,
        estimators_large=large_only,
        large=large,
        slow=frozenset(slow),
    )


CASES = [_with_options(c) for c in CASES]


# ---------------------------------------------------------------------------
# Convergence (test_convergence.py): how each case's fit is starved, or why
# it cannot be
# ---------------------------------------------------------------------------
# A starved start is the fitted value times FAR (a millionth of the way
# into a bounded range): far enough that a search stopping where the
# gradient first looks flat stops short of the maximum.
FAR = 1e6


def _far(value, bound):
    lo, hi = bound
    return value * FAR if hi is None else lo + (hi - lo) / FAR


def _parametric_start(model):
    """The fitted parameters with the first one unbounded above (else the
    first) moved :data:`FAR` away, in ``init``'s order."""
    bounds = model.dist.bounds
    k = next((i for i, b in enumerate(bounds) if b[1] is None), 0)
    params = np.array(model.params, dtype=float)
    params[k] = _far(params[k], bounds[k])
    start = ([model.gamma] if model.offset else []) + list(params)
    start += [model.p] if model.lfp else []
    return start + ([model.f0] if model.zi else [])


def _scaled_start(params, k=0):
    start = np.array(params, dtype=float)
    start[k] *= FAR
    return start


def _far_start(case, start):
    """Refit ``case`` from ``init=start(model)``, ``model`` its fit to the
    fixture."""
    return lambda d: case.fit({**d, "init": start(_fitted(case.name))})


def _no_event_level(d):
    """The first covariate is 1 on exactly the censored rows: a group with
    no events, whose coefficient the likelihood drives to infinity."""
    out = dict(d)
    Z = np.array(d["Z"], dtype=float)
    if "e" in d:
        Z[:, 0] = [e is None for e in d["e"]]
    else:
        Z[:, 0] = np.asarray(d["c"]) == 1
    out["Z"] = Z
    return out


def _comonotone(d):
    """The second coordinate half the first: dependence at its limit."""
    x = np.array(d["x"], dtype=float)
    x[:, 1] = x[:, 0] / 2
    return {**d, "x": x}


def _noise_free(y):
    """Degradation readings exactly on a path: no noise to estimate."""
    return lambda d: {**d, "y": y(np.asarray(d["x"], dtype=float))}


def _starve(case):
    """How ``case``'s fit is starved (see ``Case.starve``), or ``None``."""
    name, fit = case.name, case.fit
    cls = case.model_class.rsplit(".", 1)[-1]
    if name in ("Turnbull", "BuckleyJames"):
        return lambda d: fit({**d, "max_iter": 1})
    if name.startswith("WeibullAL"):
        # The life model's first parameter (the first is a fixed
        # placeholder, the second the Weibull shape).
        return _far_start(case, lambda m: _scaled_start(m.params, 2))
    if cls in ("ParametricRegressionModel", "FrailtyModel"):
        return lambda d: fit(_no_event_level(d))
    if cls in (
        "SemiParametricRegressionModel",
        "FineGrayModel",
        "ProportionalOddsModel",
        "CoxFrailtyModel",
    ):
        return lambda d: fit(_no_event_level(d))
    if cls == "CompetingRisksProportionalHazards":
        return lambda d: fit(_no_event_level(d))
    if name == "Logistic":
        # The scale, not the location: from a far location the fit
        # recovers with scipy 1.17 but stops short with 1.18, so only a
        # far scale fails the same way everywhere.
        return _far_start(case, lambda m: _scaled_start(m.params, 1))
    if cls == "Parametric":
        return _far_start(case, _parametric_start)
    if cls == "MixtureModel":
        # One component's data a point mass: its shape runs to infinity.
        return lambda d: fit({**d, "x": np.r_[np.full(10, 3.0), d["x"][10:]]})
    if cls == "ParametricCompetingRisks":
        # Every cause-b failure at one time: no maximum for its Weibull.
        return lambda d: fit({**d, "x": np.where(d["e"] == "b", 5.0, d["x"])})
    if cls == "ParametricRecurrenceModel":
        return _far_start(case, lambda m: _scaled_start(m.params))
    if cls == "ProportionalIntensityModel":
        return _far_start(
            case, lambda m: _scaled_start(np.r_[m.params, m.coeffs])
        )
    if cls == "CauseSpecificNHPP":
        return _far_start(case, lambda m: _scaled_start(m.models["a"].params))
    if cls == "RenewalModel":
        # [restoration, *distribution parameters]: the scale moved.
        return _far_start(
            case,
            lambda m: _scaled_start(np.r_[m.restoration, m.model.params], 1),
        )
    if cls == "CopulaModel" and name != "IndependenceCopula":
        return lambda d: fit(_comonotone(d))
    if cls == "DegradationModel":
        return lambda d: fit(_noise_free(lambda x: 10.0 + 0.35 * x)(d))
    if cls in ("WienerProcessModel", "GammaProcessModel"):
        return lambda d: fit(_noise_free(lambda x: 0.5 * x)(d))
    if cls == "DestructiveDegradationModel":
        return lambda d: fit(_noise_free(lambda x: np.exp(4.0 - 0.02 * x))(d))
    return None


# The fits with nothing to starve.
_CLOSED_FORM = "a closed-form estimate: no iteration to fail"
_EXACT = "an exact (product-limit or Nelson-Aalen type) estimator"
_NO_STARVE: dict[str, str] = {
    "Exponential": _CLOSED_FORM + " (failures / total time; init is unused)",
    "Uniform": _CLOSED_FORM + " (the sample extremes; init is unused)",
    "Binomial": _CLOSED_FORM,
    "Bernoulli": _CLOSED_FORM,
    "FixedEventProbability": _CLOSED_FORM,
    "ExactEventTime": _CLOSED_FORM,
    "AdditiveHazards": _CLOSED_FORM + " (Lin-Ying: a linear system)",
    "KaplanMeier": _EXACT,
    "NelsonAalen": _EXACT,
    "FlemingHarrington": _EXACT,
    "CompetingRisks[Nelson-Aalen]": _EXACT,
    "CompetingRisks[Kaplan-Meier]": _EXACT,
    "NonParametricCounting": _EXACT,
    "CauseSpecificMCF": _EXACT,
    "IndependenceCopula": "no dependence parameter: the margins are "
    "univariate fits, starved in their own cases",
    "AMHCopula": "the comonotone starve is no failure for it: the AMH "
    "reaches neither Frechet bound, and on such data it goes to its bound "
    "theta = 1, a valid copula, without a word (as Clayton does on "
    "countermonotone data, test_no_finite_maximum.py)",
    "InducedFailureDistribution": "a Monte Carlo of the DegradationAnalysis "
    "fit, which is starved in its own case",
    "RoystonParmar": "no public iteration limit or starting point, and no "
    "data found whose fit fails (its Nelder-Mead result is not checked "
    "for convergence, only for a finite likelihood)",
}
for _kind in ("weibull", "exponential", "non-parametric"):
    _NO_STARVE[f"SurvivalTree[{_kind}]"] = (
        "no public iteration limit or starting point: the splits are "
        "bounded searches, and a leaf is fitted when first used"
    )
_NO_STARVE["RandomSurvivalForest"] = _NO_STARVE["SurvivalTree[weibull]"]


# Fits whose initial guess is already the maximum, so returning it is right.
_START_IS_MAXIMUM = (
    "the initial guess is the maximum: the Normal MLE of log x (and the "
    "share of zeros for f0)"
)


def _with_convergence(case):
    if "convergence" in case.exclude:  # not fitted to data
        return case
    if case.name in _NO_STARVE:
        reason = _NO_STARVE[case.name]
        return replace(case, exclude={**case.exclude, "convergence": reason})
    exclude = case.exclude
    if case.name in ("LogNormal", "LogNormal[zi]"):
        exclude = {**exclude, "convergence[initial guess]": _START_IS_MAXIMUM}
    return replace(case, starve=_starve(case), exclude=exclude)


CASES = [_with_convergence(c) for c in CASES]


# Cases whose fit is not a likelihood maximisation, though other fits of
# their model class are (the "maximum" property, test_maximum.py; whole
# classes are in registry_families.NOT_A_LIKELIHOOD_FIT).
_NOT_MAXIMISED: dict[str, str] = {
    "Hypoexponential": "built from its parameters: its fit refuses data",
}


def _with_maximum(case):
    if case.name not in _NOT_MAXIMISED:
        return case
    reason = _NOT_MAXIMISED[case.name]
    return replace(case, exclude={**case.exclude, "maximum": reason})


CASES = [_with_maximum(c) for c in CASES]
