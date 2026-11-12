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
from surpyval import ml
from surpyval import multistate as ms
from surpyval import multivariate as mv
from surpyval import recurrent as rc
from surpyval.tests.conformance.registry_families import (
    BASELINES,
    BIVARIATE,
    CAUSES,
    CAUSES_REGRESSION,
    COUNTING,
    COUNTING_CAUSES,
    COUNTING_REGRESSION,
    REFIT_PROPERTIES,
    STATES,
    STATES_REGRESSION,
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
    regression,
)
from surpyval.tests.conformance.registry_fixtures import (
    JOINT_ITEM,
    X_COP,
    X_CR,
    X_DESTR,
    X_JOINT,
    X_MS,
    X_MSR,
    X_PANEL,
    X_PATH,
    X_PROC,
    X_REC,
    X_UNI,
    Z_CR,
    Z_MS,
    Z_REC,
    beta_geometric_data,
    binary_data,
    copula_data,
    cr_data,
    cure_data,
    destructive_data,
    equivalence_data,
    exact_event_data,
    joint_data,
    lfp_count_data,
    lfp_data,
    mixture_data,
    ms_data,
    offset_data,
    panel_data,
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
    zi_reg_data,
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


#: The probability-plotting fits registered, with their pivotal bounds.
_MPP_FAMILIES = ("Weibull", "LogNormal")


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
    # Probability-plotting fits, whose bounds are the pivotal ones
    for name in _MPP_FAMILIES:
        out.append(
            continuous(
                name,
                case_name=f"{name}[MPP]",
                fitters=(),
                fixed={"how": "MPP"},
                exclude={
                    "convergence[initial guess]": "a closed-form "
                    "regression: no search, so no initial guess to leave",
                    "maximum": "not a maximum-likelihood fit: its "
                    "maximum is 'not applicable'",
                    "maximum[starved]": "not a maximum-likelihood fit",
                    "derivatives": "no likelihood is maximised, so no "
                    "gradient or Hessian is taken",
                    "cb_declared": "mean_cb, mrl_cb and rmst_cb refuse, as "
                    "documented: a probability-plotting fit's bounds are "
                    "the pivotal ones on sf/ff/Hf, the quantiles and the "
                    "parameters (swept)",
                },
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
                # Bernoulli's qf inverts its ff since #344, and the flat
                # FixedEventProbability's is 0 or inf since #626.
                functions=("sf", "ff", "Hf", "qf"),
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
            functions=("sf", "ff", "df", "hf", "Hf", "qf"),
            x=np.array([1.0, 3.0, 5.0, 10.0, 22.0, 30.0, 45.0]),
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
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
        entry = {}
        if "tl" in d:
            df["tl"] = d["tl"]
            entry["tl_col"] = "tl"
        return cr.CompetingRisksProportionalHazards.fit_from_df(
            df,
            x_col="x",
            e_col="e",
            Z_cols=["z0"],
            n_col="n",
            model=how,
            **entry,
        )

    def crph_entry_data():
        # The cause-specific Cox model takes delayed entry (WP5b): every
        # third row enters at a third of its time, between event times.
        d = cr_data(True)
        d["tl"] = np.where(np.arange(d["x"].size) % 3 == 1, d["x"] / 3, 0.0)
        return d

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

    def cr_entry_data():
        # Delayed entry: a third of the units from 0, the rest entering
        # part of the way to their exit, many after others have left.
        d = cr_data()
        share = np.tile([0.0, 0.3, 0.6], 8)[scramble(24)]
        d["tl"] = np.round(share * d["x"], 3)
        return d

    def cr_entry_from_df(d):
        df = pd.DataFrame(
            {"x": d["x"], "e": d["e"], "n": d["n"], "tl": d["tl"]}
        )
        return cr.CompetingRisks.fit_from_df(
            df,
            x_col="x",
            e_col="e",
            n_col="n",
            tl_col="tl",
            how="Kaplan-Meier",
        )

    out.append(
        Case(
            name="CompetingRisks[tl]",
            fitters=(),
            model_class="surpyval.univariate.competing_risks.CompetingRisks",
            interface=CAUSES,
            data=cr_entry_data,
            fit=_fit(cr.CompetingRisks, how="Kaplan-Meier"),
            functions=("sf", "ff", "Hf"),
            event_functions=("cif",),
            events=("a", "b"),
            x=X_CR,
            rows=("x", "e", "n", "tl"),
            times=("x", "tl"),
            paths={"fit_from_df": cr_entry_from_df},
            rtol=1e-9,
        )
    )
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
            # All-cause df and qf (WP2c): df == hf * sf, qf(ff(x)) == x.
            functions=("sf", "ff", "Hf", "hf", "df", "qf"),
            event_functions=("cif",),
            events=("a", "b"),
            x=X_CR,
            rows=("x", "e", "n"),
            paths={"fit_from_df": pcr_from_df},
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
        )
    )

    def pcr_interval_data():
        # The same units, every third failure found at an inspection: in
        # the window (x - 1.5, x + 0.5], left censored where it would start
        # at or below 0. The causes are then fitted jointly.
        d = cr_data()
        x, e = d["x"], d["e"]
        found = (np.arange(x.size) % 3 == 0) & np.array(
            [v is not None for v in e]
        )
        lo, hi = x - 1.5, x + 0.5
        left = found & (lo <= 0)
        x2 = np.column_stack(
            [
                np.where(found & ~left, lo, np.where(left, hi, x)),
                np.where(found, hi, x),
            ]
        )
        c = np.where(
            found,
            np.where(left, -1, 2),
            np.where([v is None for v in e], 1, 0),
        )
        return {"x": x2, "c": c, "e": e, "n": d["n"]}

    def pcr_interval_from_df(d):
        df = pd.DataFrame(
            {
                "lo": d["x"][:, 0],
                "hi": d["x"][:, 1],
                "c": d["c"],
                "e": d["e"],
                "n": d["n"],
            }
        )
        return cr.ParametricCompetingRisks.fit_from_df(
            df, None, "e", c_col="c", n_col="n", xl_col="lo", xr_col="hi"
        )

    out.append(
        Case(
            name="ParametricCompetingRisks[interval]",
            fitters=(),
            model_class=(
                "surpyval.univariate.competing_risks.ParametricCompetingRisks"
            ),
            interface=CAUSES,
            data=pcr_interval_data,
            fit=_fit(cr.ParametricCompetingRisks),
            functions=("sf", "ff", "Hf", "hf", "df", "qf"),
            event_functions=("cif",),
            events=("a", "b"),
            x=X_CR,
            rows=("x", "c", "e", "n"),
            paths={"fit_from_df": pcr_interval_from_df},
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={
                "derivatives": "the checks rebuild each cause's own "
                "likelihood, which the joint fit is not; the joint "
                "likelihood's derivatives are checked against finite "
                "differences in test_parametric_competing_risks_interval",
                "maximum": "the search there is a cause at a time; the "
                "joint fit's own verification (zero gradient, positive "
                "definite Hessian) is checked in "
                "test_parametric_competing_risks_interval",
            },
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
                data=(
                    crph_entry_data
                    if how == "Cox"
                    else functools.partial(cr_data, True)
                ),
                fit=_fit(cr.CompetingRisksProportionalHazards, model=how),
                # The Fine-Gray form has no all-cause survival; its sf is a
                # cause's 1 - cif, reached through cif below.
                functions=("sf", "ff") if how == "Cox" else (),
                event_functions=("cif",),
                events=("a", "b"),
                x=X_CR,
                Z=Z_CR,
                rows=(
                    ("x", "Z", "e", "n", "tl")
                    if how == "Cox"
                    else ("x", "Z", "e", "n")
                ),
                times=("x", "tl") if how == "Cox" else ("x",),
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

    def pcrr_from_df(d, formula=False):
        df = pd.DataFrame(
            {"x": d["x"], "e": d["e"], "n": d["n"], "z0": d["Z"][:, 0]}
        )
        covariates = {"formula": "z0"} if formula else {"Z_cols": ["z0"]}
        return cr.ParametricCompetingRisksRegression.fit_from_df(
            df, x_col="x", e_col="e", n_col="n", **covariates
        )

    out.append(
        Case(
            name="ParametricCompetingRisksRegression",
            fitters=(
                "surpyval.univariate.competing_risks"
                ".ParametricCompetingRisksRegression",
            ),
            model_class=(
                "surpyval.univariate.competing_risks"
                ".ParametricCompetingRisksRegression"
            ),
            interface=CAUSES_REGRESSION,
            data=functools.partial(cr_data, True),
            fit=_fit(cr.ParametricCompetingRisksRegression),
            # (qf(p, Z) is checked in the model's own tests: the suite's
            # quantile query has no covariate rows to pair with)
            functions=("sf", "ff", "Hf", "hf", "df"),
            event_functions=("cif",),
            events=("a", "b"),
            x=X_CR,
            Z=Z_CR,
            rows=("x", "Z", "e", "n"),
            covariates="Z",
            paths={
                "fit_from_df": pcrr_from_df,
                "formula": functools.partial(pcrr_from_df, formula=True),
            },
            draw=lambda m, s: m.random(15, Z_CR[:2], random_state=s)[0]["x"],
            explicit_seed=True,
            # One row of coefficients per cause, as the cause-specific Cox
            # model's betas
            # (after each cause's Weibull alpha and beta)
            coefficients=lambda m: np.array(
                [
                    np.asarray(m.models[k].params, dtype=float)[2:]
                    for k in m.causes
                ]
            ),
            intercept=True,
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
# Multi-state
# ---------------------------------------------------------------------------
_MS_STATES = ("healthy", "ill", "dead")


def _multistate():
    def frame(d):
        df = pd.DataFrame(
            {
                "x": d["x"],
                "e": d["e"],
                "state": d["state"],
                "tl": d["tl"],
                "n": d["n"],
            }
        )
        if "Z" in d:
            df["z0"] = d["Z"][:, 0]
        return df

    def ms_from_df(d):
        return ms.MultiState.fit_from_df(
            frame(d),
            "x",
            "e",
            "state",
            tl_col="tl",
            n_col="n",
            transitions=d["transitions"],
        )

    def msph_from_df(d):
        return ms.MultiStateProportionalHazards.fit_from_df(
            frame(d),
            "x",
            ["z0"],
            "e",
            "state",
            tl_col="tl",
            n_col="n",
            transitions=d["transitions"],
        )

    # The covariate's effect shared by the two transitions into "dead",
    # whose baselines are also taken as proportional (one stratum, with an
    # indicator). The transitions are read from the data, so the case
    # follows its states when they are relabelled.
    def shared_options(d):
        into_dead = [tuple(d["transitions"][1]), tuple(d["transitions"][3])]
        return {"shared": {0: into_dead}, "proportional_baselines": into_dead}

    def msph_shared(d):
        return ms.MultiStateProportionalHazards.fit(
            **{**shared_options(d), **d}
        )

    def msph_shared_from_df(d):
        return ms.MultiStateProportionalHazards.fit_from_df(
            frame(d),
            "x",
            ["z0"],
            "e",
            "state",
            tl_col="tl",
            n_col="n",
            transitions=d["transitions"],
            **shared_options(d),
        )

    def pms_from_df(d):
        return ms.ParametricMultiState.fit_from_df(
            frame(d),
            "x",
            "e",
            "state",
            tl_col="tl",
            n_col="n",
            transitions=d["transitions"],
        )

    def pmsr_from_df(d):
        return ms.ParametricMultiStateRegression.fit_from_df(
            frame(d),
            "x",
            ["z0"],
            "e",
            "state",
            tl_col="tl",
            n_col="n",
            transitions=d["transitions"],
        )

    exact = "an exact (Aalen-Johansen) estimate of the spells"
    return [
        Case(
            name="MultiState",
            fitters=("surpyval.multistate.MultiState",),
            model_class="surpyval.multistate.MultiState",
            interface=STATES,
            data=ms_data,
            fit=_fit(ms.MultiState),
            functions=(),
            event_functions=("occupancy",),
            events=_MS_STATES,
            x=X_MS,
            rows=("x", "e", "state", "tl", "n"),
            times=("x", "tl"),
            paths={"fit_from_df": ms_from_df},
            rtol=1e-9,
            exclude={"convergence": exact},
        ),
        Case(
            name="MultiStateProportionalHazards",
            fitters=("surpyval.multistate.MultiStateProportionalHazards",),
            model_class="surpyval.multistate.MultiStateProportionalHazards",
            interface=STATES_REGRESSION,
            data=functools.partial(ms_data, True),
            fit=_fit(ms.MultiStateProportionalHazards),
            functions=(),
            event_functions=("occupancy",),
            events=_MS_STATES,
            x=X_MSR,
            Z=Z_MS,
            rows=("x", "Z", "e", "state", "tl", "n"),
            times=("x", "tl"),
            covariates="Z",
            drops_missing_covariate=False,
            paths={"fit_from_df": msph_from_df},
            rtol=1e-6,
            coefficients=lambda m: m.betas,
            intercept=True,
        ),
        Case(
            name="MultiStateProportionalHazards[shared]",
            fitters=(),
            model_class="surpyval.multistate.MultiStateProportionalHazards",
            interface=STATES_REGRESSION,
            data=functools.partial(ms_data, True),
            fit=msph_shared,
            functions=(),
            event_functions=("occupancy",),
            events=_MS_STATES,
            x=X_MSR,
            Z=Z_MS,
            rows=("x", "Z", "e", "state", "tl", "n"),
            times=("x", "tl"),
            covariates="Z",
            drops_missing_covariate=False,
            paths={"fit_from_df": msph_shared_from_df},
            rtol=1e-6,
            coefficients=lambda m: m.betas,
            intercept=True,
            exclude=dict.fromkeys(
                ("aliasing", "aliasing_constant"),
                "a copy of a covariate shared by some transitions only is "
                "aliased in one combination of its per-transition "
                "coefficients (the shared column is their sum), not in all "
                "of them; a constant on the two proportional baselines is "
                "aliased with their indicator. The unshared case checks the "
                "aliasing itself",
            ),
        ),
        Case(
            name="ParametricMultiState",
            fitters=("surpyval.multistate.ParametricMultiState",),
            model_class="surpyval.multistate.ParametricMultiState",
            interface=STATES,
            data=ms_data,
            fit=_fit(ms.ParametricMultiState),
            functions=(),
            event_functions=("occupancy",),
            events=_MS_STATES,
            x=X_MS,
            rows=("x", "e", "state", "tl", "n"),
            times=("x", "tl"),
            paths={"fit_from_df": pms_from_df},
            exclude={
                "convergence": "each transition is fitted by its "
                "distribution's own univariate fit, whose starved fits the "
                "Parametric cases check; the model adds no search of its "
                "own and takes no init"
            },
        ),
        Case(
            name="ParametricMultiStateRegression",
            fitters=("surpyval.multistate.ParametricMultiStateRegression",),
            model_class="surpyval.multistate.ParametricMultiStateRegression",
            interface=STATES_REGRESSION,
            data=functools.partial(ms_data, True),
            fit=_fit(ms.ParametricMultiStateRegression),
            functions=(),
            event_functions=("occupancy",),
            events=_MS_STATES,
            x=X_MSR,
            Z=Z_MS,
            rows=("x", "Z", "e", "state", "tl", "n"),
            times=("x", "tl"),
            covariates="Z",
            drops_missing_covariate=False,
            paths={"fit_from_df": pmsr_from_df},
            coefficients=lambda m: m.betas,
            intercept=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Multi-state, states seen at visits (panel data)
# ---------------------------------------------------------------------------
_PANEL_EXCLUDE = {
    "counts": "the rows are visits of units, and a visit repeated is a "
    "unit seen twice at one time (refused); the model takes no counts",
    "convergence": "the search starts from the data's crude rates and "
    "takes no init or iteration limit to starve; a search that ends "
    "short is checked by 'maximum'",
}


def _panel_multistate():
    def frame(d):
        df = pd.DataFrame({"x": d["x"], "s": d["s"], "i": d["i"]})
        if "Z" in d:
            df["z0"] = d["Z"][:, 0]
        return df

    def options(d):
        return {
            "transitions": d["transitions"],
            "censored": d["censored"],
            "exact_entry": d["exact_entry"],
        }

    def from_df(d):
        return ms.PanelMultiState.fit_from_df(
            frame(d), "x", "s", "i", **options(d)
        )

    def reg_from_df(d):
        return ms.PanelMultiStateRegression.fit_from_df(
            frame(d), "x", ["z0"], "s", "i", **options(d)
        )

    return [
        Case(
            name="PanelMultiState",
            fitters=("surpyval.multistate.PanelMultiState",),
            model_class="surpyval.multistate.PanelMultiState",
            interface=STATES,
            data=panel_data,
            fit=_fit(ms.PanelMultiState),
            functions=(),
            event_functions=("occupancy",),
            events=_MS_STATES,
            x=X_PANEL,
            rows=("x", "s", "i"),
            times=("x",),
            paths={"fit_from_df": from_df},
            exclude=_PANEL_EXCLUDE,
        ),
        Case(
            name="PanelMultiStateRegression",
            fitters=("surpyval.multistate.PanelMultiStateRegression",),
            model_class="surpyval.multistate.PanelMultiStateRegression",
            interface=STATES_REGRESSION,
            data=functools.partial(panel_data, True),
            fit=_fit(ms.PanelMultiStateRegression),
            functions=(),
            event_functions=("occupancy",),
            events=_MS_STATES,
            x=X_PANEL,
            Z=Z_MS[: X_PANEL.size],
            rows=("x", "Z", "s", "i"),
            times=("x",),
            covariates="Z",
            drops_missing_covariate=False,
            paths={"fit_from_df": reg_from_df},
            coefficients=lambda m: m.betas,
            intercept=True,
            exclude=_PANEL_EXCLUDE,
        ),
    ]


# ---------------------------------------------------------------------------
# Recurrent events
# ---------------------------------------------------------------------------
def _recurrent_data_path(fitter, **fixed):
    def run(d):
        data = sp.handle_xicn(d["x"], d["i"], d["c"], d["n"])
        return fitter.fit_from_recurrent_data(data, **fixed)

    return run


def _recurrent_tvc_path(fitter, marks=False):
    """The fixture's event rows as start-stop rows, each from the item's
    previous row (0 for its first) with the item's covariates, fitted by
    ``fit_tvc``: the time-fixed data in the time-varying layout (with the
    event type of each row where ``marks``)."""

    def run(d):
        x, i = np.asarray(d["x"], dtype=float), np.asarray(d["i"])
        xl = np.zeros_like(x)
        for item in np.unique(i):
            rows = np.flatnonzero(i == item)
            rows = rows[np.argsort(x[rows], kind="stable")]
            xl[rows[1:]] = x[rows[:-1]]
        extra = {"e": d["e"]} if marks else {}
        return fitter.fit_tvc(i, xl, x, d["c"], d["Z"], d["n"], **extra)

    return run


def _recurrent_cox_tvc_path(fitter, **kw):
    """The fixture's rows as start-stop rows, each from the item's
    previous row (0 for its first), fitted by ``fit_tvc``."""

    def run(d):
        x, i = np.asarray(d["x"], dtype=float), np.asarray(d["i"])
        xl = np.zeros_like(x)
        for item in np.unique(i):
            rows = np.flatnonzero(i == item)
            rows = rows[np.argsort(x[rows], kind="stable")]
            xl[rows[1:]] = x[rows[:-1]]
        return fitter.fit_tvc(i, xl, x, d["c"], d["Z"], **kw)

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
                # fit_tvc with one covariate value per item is fit
                paths={"fit_tvc": _recurrent_tvc_path(fitter)},
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
    # The semi-parametric recurrent-event regressions: Cox fits of a
    # layout of each item's rows, with a robust covariance by item.
    recurrent_cox = (
        ("AndersenGill", "AndersenGill", {}, {}),
        ("PrenticeWilliamsPeterson", "PrenticeWilliamsPeterson", {}, {}),
        (
            "PrenticeWilliamsPeterson[gap]",
            "PrenticeWilliamsPeterson",
            {"time_scale": "gap"},
            {},
        ),
        ("WeiLinWeissfeld", "WeiLinWeissfeld", {}, {}),
    )
    for label, name, kw, _ in recurrent_cox:
        fitter = getattr(rc, name)
        stratified = name != "AndersenGill"
        paths = {}
        if name != "WeiLinWeissfeld":
            # fit_tvc with the item's covariates on every row is fit
            paths["fit_tvc"] = _recurrent_cox_tvc_path(fitter, **kw)
        out.append(
            Case(
                name=label,
                fitters=(
                    (f"surpyval.recurrent.{name}",) if "[" not in label else ()
                ),
                model_class=(
                    "surpyval.recurrent.regression.recurrent_cox"
                    ".RecurrentCoxModel"
                ),
                interface=COUNTING_REGRESSION,
                data=functools.partial(recurrent_data, True),
                fit=_fit(fitter, **kw),
                functions=("cif",),
                call_kwargs={"stratum": 2} if stratified else {},
                x=X_REC,
                Z=Z_REC,
                paths=paths,
                rows=("x", "Z", "i", "c", "n"),
                covariates="Z",
                drops_missing_covariate=False,
                coefficients=lambda m: m.params,
                # The unspecified baseline absorbs a constant.
                intercept=True,
                rtol=1e-6,
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
    out.append(
        Case(
            name="CauseSpecificProportionalIntensity",
            fitters=("surpyval.recurrent.CauseSpecificProportionalIntensity",),
            model_class=(
                "surpyval.recurrent.CauseSpecificProportionalIntensity"
            ),
            # cif(x, Z, event) per cause; total_cif(x, Z) over the causes
            interface=COUNTING_REGRESSION,
            data=functools.partial(recurrent_data, True, True),
            fit=_fit(rc.CauseSpecificProportionalIntensity),
            functions=("total_cif", "total_iif"),
            event_functions=("cif", "iif"),
            events=("a", "b"),
            x=X_REC,
            Z=Z_REC,
            # fit_tvc with one covariate value per item is fit
            paths={
                "fit_tvc": _recurrent_tvc_path(
                    rc.CauseSpecificProportionalIntensity, marks=True
                )
            },
            rows=("x", "Z", "i", "c", "n", "e"),
            covariates="Z",
            drops_missing_covariate=False,
            coefficients=lambda m: np.array(
                [m.models[k].coeffs for k in m.event_types]
            ),
            # Each cause's Duane scale b absorbs a constant.
            intercept=True,
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


# ---------------------------------------------------------------------------
# Joint models of a marker and a lifetime (#891)
# ---------------------------------------------------------------------------
def _joint_coefficients(model):
    """A joint model's covariate coefficients: the hazard's, then each path
    parameter's shifts of its mean (a row each)."""
    names, params = model.parameter_names, np.asarray(model.params, float)
    rows = [model.feature_names] + [
        [f"{a}_{f}" for f in model.feature_names]
        for a in model.path_model.parameter_names
    ]
    return np.array([[params[names.index(n)] for n in row] for row in rows])


def _joint():
    """The joint models, predicting for one item read four times and still
    running at 400 (``JOINT_ITEM``) through the array form ``sf(x, Z,
    markers=..., given=...)``: one covariate vector a call. The hazard
    follows the marker's current value (``WeibullJointPH``), its slope,
    the area under it, or its value and slope together."""
    fitters = {
        "WeibullJointPH": sp.WeibullJointPH,
        "WeibullJointPH[slope]": sp.JointPH(sp.Weibull, link="slope"),
        "WeibullJointPH[cumulative]": sp.JointPH(
            sp.Weibull, link="cumulative"
        ),
        "WeibullJointPH[value+slope]": sp.JointPH(
            sp.Weibull, link=["value", "slope"]
        ),
    }
    out = []
    for name, fitter in fitters.items():
        case = regression(
            name,
            fitter,
            # the links are ways of calling the one public factory
            fitters=("surpyval.WeibullJointPH",) if "[" not in name else (),
            model_class="surpyval.JointModel",
            data=joint_data,
            x=X_JOINT,
            Z=np.ones((X_JOINT.size, 1)),
            functions=("sf", "ff"),
            z_style="single",
            paths={
                "fit_from_df": lambda d, fitter=fitter: fitter.fit_from_df(
                    pd.DataFrame(
                        {
                            "x": d["x"],
                            "c": d["c"],
                            "n": d["n"],
                            "z0": np.asarray(d["Z"])[:, 0],
                            "markers": list(d["markers"]),
                        }
                    ),
                    x_col="x",
                    markers_col="markers",
                    c_col="c",
                    n_col="n",
                    Z_cols=["z0"],
                )
            },
            rows=("x", "Z", "c", "n", "markers"),
            coefficients=_joint_coefficients,
            intercept=True,
            draw=None,
            exclude={
                "random_sf": "an item's draws depend on its readings: "
                "JointModel.predict(...).random(size) draws for one item",
                "row_independence": "one item a call (Z is its covariate "
                "vector), so there are no rows to mix up",
            },
        )
        out.append(
            replace(
                case,
                times=("x", "markers"),
                call_kwargs=dict(JOINT_ITEM),
                call_times=("markers", "given"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Royston-Parmar (spline baseline) regression: PH and PO on the spline
# ---------------------------------------------------------------------------
def _spline_regression():
    out = []
    for name, extra in (
        ("RoystonParmarPH", ("surpyval.RoystonParmarRegression",)),
        ("RoystonParmarPO", ()),
    ):
        out.append(
            regression(
                name,
                getattr(sp, name),
                fitters=(f"surpyval.{name}",) + extra,
                intercept=True,
                slow=frozenset() if name.endswith("PH") else REFIT_PROPERTIES,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Distributional regression: covariates on every parameter
# ---------------------------------------------------------------------------
def _distributional_coefficients(model):
    """A distributional regression's coefficients, a row per parameter
    (every covariate is on every parameter in the fixtures)."""
    params = np.asarray(model.params, dtype=float)
    rows, start = [], model.k_dist
    for cols in model.covariate_positions:
        rows.append(params[start : start + len(cols)])
        start += len(cols)
    return np.array(rows)


DISTRIBUTIONAL_BASELINES = BASELINES + ("LogLogistic",)
FAST_DISTRIBUTIONAL = ("WeibullDistributional", "LogNormalDistributional")


def _distributional():
    out = []
    for base in DISTRIBUTIONAL_BASELINES:
        name = base + "Distributional"
        out.append(
            regression(
                name,
                getattr(sp, name),
                model_class="surpyval.DistributionalRegressionModel",
                fitters=(f"surpyval.{name}",)
                + (
                    (
                        "surpyval.Distributional",
                        "surpyval.DistributionalFitter",
                    )
                    if base == "Weibull"
                    else ()
                ),
                slow=(
                    frozenset()
                    if name in FAST_DISTRIBUTIONAL
                    else REFIT_PROPERTIES
                ),
                coefficients=_distributional_coefficients,
                intercept=True,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Mixture-cure regression: the distributional regression of a lifetime
# with a fraction that never fails
# ---------------------------------------------------------------------------
def _cure_coefficients(model):
    """A mixture-cure regression's coefficients, a row for each
    parameter that has covariates (in the fixtures the latency's life
    parameter and the incidence, both on every column)."""
    params = np.asarray(model.params, dtype=float)
    rows, start = [], model.k_dist
    for cols in model.covariate_positions:
        if len(cols):
            rows.append(params[start : start + len(cols)])
        start += len(cols)
    return np.array(rows)


CURE_LATENCIES = (
    "Weibull",
    "LogNormal",
    "LogLogistic",
    "Gamma",
    "Exponential",
)
FAST_CURE = ("WeibullMixtureCure",)


def _mixture_cure():
    out = []
    for base in CURE_LATENCIES:
        name = base + "MixtureCure"
        out.append(
            regression(
                name,
                getattr(sp, name),
                data=cure_data,
                # A MixtureCureModel is a DistributionalRegressionModel:
                # declared as one, so every table keyed by the model's
                # class applies to it.
                model_class="surpyval.DistributionalRegressionModel",
                fitters=(f"surpyval.{name}",)
                + (
                    (
                        "surpyval.MixtureCure",
                        "surpyval.MixtureCureFitter",
                        "surpyval.MixtureCureModel",
                    )
                    if base == "Weibull"
                    else ()
                ),
                slow=(frozenset() if name in FAST_CURE else REFIT_PROPERTIES),
                coefficients=_cure_coefficients,
                intercept=True,
            )
        )
    return out


FAST_ZI = ("WeibullZeroInflated",)


def _zero_inflated():
    out = []
    for base in CURE_LATENCIES:
        name = base + "ZeroInflated"
        out.append(
            regression(
                name,
                getattr(sp, name),
                data=zi_reg_data,
                # A ZeroInflatedModel is a DistributionalRegressionModel:
                # declared as one, so every table keyed by the model's
                # class applies to it.
                model_class="surpyval.DistributionalRegressionModel",
                fitters=(f"surpyval.{name}",)
                + (
                    (
                        "surpyval.ZeroInflated",
                        "surpyval.ZeroInflatedFitter",
                        "surpyval.ZeroInflatedModel",
                    )
                    if base == "Weibull"
                    else ()
                ),
                slow=(frozenset() if name in FAST_ZI else REFIT_PROPERTIES),
                coefficients=_cure_coefficients,
                intercept=True,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Equivalent time scales (surpyval.Equivalence): one life on two clocks
# ---------------------------------------------------------------------------
FAST_EQUIVALENCE = ("WeibullEquivalence",)


def _equivalence_fit(fitter):
    def fit(d):
        return fitter.fit(
            d["x"], d["x2"], c=d.get("c"), n=d.get("n"), init=d.get("init")
        )

    return fit


def _equivalence_df(fitter):
    def fit(d):
        df = pd.DataFrame(
            {"hours": d["x"], "starts": d["x2"], "c": d["c"], "n": d["n"]}
        )
        return fitter.fit_from_df(
            df, x_cols=["hours", "starts"], c_col="c", n_col="n"
        )

    return fit


def _equivalence():
    out = []
    for base in CURE_LATENCIES:
        name = base + "Equivalence"
        fitter = getattr(sp, name)
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.{name}",)
                + (
                    (
                        "surpyval.Equivalence",
                        "surpyval.EquivalenceFitter",
                    )
                    if base == "Weibull"
                    else ()
                ),
                model_class="surpyval.EquivalenceModel",
                interface=UNIVARIATE,
                data=equivalence_data,
                fit=_equivalence_fit(fitter),
                functions=UNI_FUNCTIONS + ("qf",),
                x=X_UNI,
                paths={"fit_from_df": _equivalence_df(fitter)},
                # the second clock (starts) is a count, not a time: the
                # units property rescales the first, and k with it
                rows=("x", "x2", "c", "n"),
                times=("x",),
                draw=lambda m, s: m.random(15, rate=0.2, random_state=s),
                explicit_seed=True,
                slow=(
                    frozenset()
                    if name in FAST_EQUIVALENCE
                    else REFIT_PROPERTIES
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Bayesian fits (surpyval.bayes)
# ---------------------------------------------------------------------------
def _bayes_weibull(d):
    """The grid engine: a prior on the shape (unitless) and none (flat on
    the log scale) on the scale, so a change of time unit leaves it as it
    is."""
    return sp.Weibull.fit_bayes(
        d["x"],
        d.get("c"),
        d.get("n"),
        priors={"beta": sp.prior.around(2, between=(1.2, 4))},
    )


def _bayes_exponential(d):
    """The closed form: a Gamma prior on the rate whose second parameter
    is a time (``prior_time``, rescaled with the data)."""
    return sp.Exponential.fit_bayes(
        d["x"],
        d.get("c"),
        d.get("n"),
        priors={
            "failure_rate": sp.Gamma.from_params(
                [2.0, float(np.asarray(d["prior_time"]))]
            )
        },
    )


def _bayes_paths(fit, fitter, priors):
    def from_df(d):
        df = pd.DataFrame({"x": d["x"], "c": d["c"], "n": d["n"]})
        return fitter.fit_bayes_from_df(
            df, x_col="x", c_col="c", n_col="n", priors=priors(d)
        )

    def from_data(d):
        return fitter.fit_bayes(
            sp.SurpyvalData(x=d["x"], c=d.get("c"), n=d.get("n")),
            priors=priors(d),
        )

    # (fit_from_df: the DataFrame path, fit_bayes_from_df)
    return {"fit_from_df": from_df, "fit_bayes(data)": from_data}


# A credible interval's own point is the posterior median (cri(x,
# point=True)); the estimate the sweeps compare with is the model's
# function, the posterior predictive, which is the posterior mean for sf,
# ff and df but not for hf: far past the data the predictive hazard falls
# below the hazards of most parameter values (the units that survive are
# those with low hazards), so containment does not hold for it, as it
# should not. cri(on="hf") is checked against its median in
# tests/bayes/test_fit_bayes.py. Not Wald intervals: they close onto the
# median, not the predictive (test_fit_bayes.py checks that).
_BAYES_BOUNDS = (
    Bound("cri", on=("sf", "ff", "Hf", "df"), wald=False),
    Bound("quantile_cri", point="qf", wald=False),
    Bound("param_cri", kind="param", wald=False),
    Bound("mean_cri", kind="summary", query=((),), wald=False),
)
_BAYES_EXCLUDE = {
    "convergence": "not an optimiser: the grid checks its own accuracy and "
    "refuses when it cannot reach it (tests/bayes/test_fit_bayes.py::"
    "test_starved_grid_refuses); a closed form has nothing to starve",
}


def _bayes():
    def weibull_priors(d):
        return {"beta": sp.prior.around(2, between=(1.2, 4))}

    def exponential_priors(d):
        return {
            "failure_rate": sp.Gamma.from_params(
                [2.0, float(np.asarray(d["prior_time"]))]
            )
        }

    def exponential_data():
        return {**uni_data(), "prior_time": 20.0}

    common = dict(
        model_class="surpyval.bayes.BayesianParametric",
        interface=UNIVARIATE,
        functions=UNI_FUNCTIONS + ("qf",),
        x=X_UNI,
        draw=lambda m, s: m.random(15, random_state=s),
        explicit_seed=True,
        exclude=dict(_BAYES_EXCLUDE),
    )
    return [
        Case(
            name="Weibull[bayes]",
            fitters=("surpyval.bayes.BayesianParametric",),
            data=uni_data,
            fit=_bayes_weibull,
            paths=_bayes_paths(_bayes_weibull, sp.Weibull, weibull_priors),
            rtol=1e-6,
            **common,
        ),
        Case(
            name="Exponential[bayes]",
            fitters=(),
            data=exponential_data,
            fit=_bayes_exponential,
            paths=_bayes_paths(
                _bayes_exponential, sp.Exponential, exponential_priors
            ),
            times=("x", "prior_time"),
            rtol=1e-9,
            **common,
        ),
    ]


CASES: list[Case] = (
    _univariate()
    + _regression_family()
    + _accelerated_life_family()
    + _frailty_family()
    + _semi_parametric()
    + _trees()
    + _competing_risks()
    + _multistate()
    + _recurrent()
    + _degradation()
    + _copulas()
    + _joint()
    + _spline_regression()
    + _distributional()
    + _mixture_cure()
    + _equivalence()
    + _zero_inflated()
    + _panel_multistate()
    + _bayes()
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
    """``x_col``, ``c_col``, ``n_col``, ``tl_col`` (and ``xl_col`` /
    ``xr_col`` for interval rows), the univariate fit_from_df's names."""

    def run(d):
        df, rest = _frame(d, ("x", "c", "n", "tl", "tr"))
        if "x" in df:
            names = {"x_col": "x"}
        else:
            df = df.rename(columns={"x0": "xl", "x1": "xr"})
            names = {"xl_col": "xl", "xr_col": "xr"}
        names |= {f"{k}_col": k for k in ("c", "n", "tl", "tr") if k in df}
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
    for label, name, kw in (
        ("AndersenGill", "AndersenGill", {}),
        ("PrenticeWilliamsPeterson", "PrenticeWilliamsPeterson", {}),
        (
            "PrenticeWilliamsPeterson[gap]",
            "PrenticeWilliamsPeterson",
            {"time_scale": "gap"},
        ),
        ("WeiLinWeissfeld", "WeiLinWeissfeld", {}),
    ):
        paths[label] = _col_df(getattr(rc, name), ("x", "i", "c", "n"), **kw)
    paths["CauseSpecificProportionalIntensity"] = _col_df(
        rc.CauseSpecificProportionalIntensity
    )
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
# trials (exact Clopper-Pearson by default, Wald, likelihood ratio and
# Jeffreys as options), and covariance() is p (1 - p) / N from the same
# counts; cb, quantile_cb and mean_cb raise, pointing to it (#580).
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
# The offset, limited-failure and zero-inflated fits' searches run over
# their extra parameters too: the Weibull's in every full run, the other
# families' with the calibration studies. An offset model's times include
# one before its estimated offset (6 on the Weibull's 6.56),
# where the survival is 1 at the estimate but not over the region.
for _family in ("Weibull", "Exponential", "Gamma", "LogNormal"):
    _LR_X[f"{_family}[offset]"] = np.array([6.0, 9.0, 13.0, 18.0])
    _LR_X[f"{_family}[lfp]"] = np.array([4.0, 8.0, 13.0])
    _LR_X[f"{_family}[zi]"] = np.array([4.0, 8.0, 13.0])
_LR_NIGHTLY = {"NegativeBinomial", "ExpoWeibull"} | {
    f"{family}[{variant}]"
    for family in ("Exponential", "Gamma", "LogNormal")
    for variant in ("offset", "lfp", "zi")
}
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
    "WeibullDistributional": (2.0, 8.0),
    "LogNormalDistributional": (2.0, 8.0),
    "WeibullMixtureCure": (2.0, 8.0),
    "WeibullZeroInflated": (2.0, 8.0),
}


# Fits with no parameter covariance by design, so no Wald bounds: the
# Uniform's MLE sits on the sample extremes, where the likelihood's
# curvature says nothing about its uncertainty (#460).
_NO_WALD = {"Uniform"}


#: The univariate cases whose parametric bootstrap is swept (#286), at
#: these times: the limited-failure and zero-inflated fits of two
#: families. The calls with one ``n_boot`` and seed share one set of
#: refits.
_BOOT_X = {
    f"{family}[{variant}]": np.array([4.0, 8.0, 13.0])
    for family in ("Weibull", "LogNormal")
    for variant in ("lfp", "zi")
}


def _bootstrap_bounds(case, on):
    boot = dict(kwargs={"method": "bootstrap", **_BOOT}, wald=False, slow=True)
    return [
        Bound(
            "cb",
            on=on,
            query=tuple(_BOOT_X[case.name]),
            label="cb[bootstrap]",
            **boot,
        ),
        Bound("param_cb", kind="param", label="param_cb[bootstrap]", **boot),
        Bound(
            "quantile_cb",
            point="qf",
            query=(0.1, 0.3),
            label="quantile_cb[bootstrap]",
            **boot,
        ),
    ]


def _pivotal_bounds(case):
    """The pivotal Monte Carlo bounds of a probability-plotting fit, its
    only bounds: on sf, ff and Hf, the quantiles and the parameters. The
    calls with one ``n_sim`` and seed share one simulation."""
    kw = dict(
        kwargs={"method": "pivotal", "n_sim": 999, "random_state": 1},
        wald=False,
    )
    on = tuple(f for f in ("sf", "ff", "Hf") if f in case.functions)
    return (
        Bound("cb", on=on, label="cb[pivotal]", **kw),
        Bound("param_cb", kind="param", label="param_cb[pivotal]", **kw),
        Bound(
            "quantile_cb",
            point="qf",
            query=(0.1, 0.3),
            label="quantile_cb[pivotal]",
            **kw,
        ),
    )


def _parametric_bounds(case):
    if case.name.endswith("[MPP]"):
        return _pivotal_bounds(case)
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
            # The mean residual life at the case's query times (#825)
            Bound(
                "mrl_cb",
                point="mrl",
                kwargs={"method": "wald"},
                in_range=case.name != "Beta4",
                nan_ok=True,
                label="mrl_cb[wald]",
            ),
        ]
        if case.continuous:
            # The restricted mean to the last query time
            out.append(
                Bound(
                    "rmst_cb",
                    kind="summary",
                    kwargs={"method": "wald"},
                    query=((_rmst_tau(case),),),
                    label="rmst_cb[wald]",
                )
            )
    # The likelihood-ratio search is swept on the cases in _LR_X only.
    # The ExpoWeibull's and NegativeBinomial's sweeps (searches in
    # multi-parameter valleys, seconds each) took about ten minutes on
    # four cores, so they run with the calibration studies; the
    # other four sweep in every run, and test_likelihood_ratio_edges.py
    # checks those two families' edges and valleys directly (#421). The
    # offset, limited-failure and zero-inflated fits sweep theirs too.
    if case.name in _BOOT_X:
        out += _bootstrap_bounds(case, on)
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
        if case.name in _LR_NIGHTLY and (
            "[" in case.name or case.name == "ExpoWeibull"
        ):
            # (an offset, LFP or ZI variant's mean residual life is an
            # integral at every point the searches try: 24 minutes for
            # the offset LogNormal's sweep alone; the Weibull's, in
            # closed form, sweeps the same path. The ExpoWeibull's is an
            # integral too, in a three-parameter valley: one two-sided
            # bound at one time took 414 s on four cores, so its sweep,
            # dozens of them, ran for hours on the 17 October 2026
            # release run.)
            return tuple(out)
        out.append(
            Bound(
                "mrl_cb",
                point="mrl",
                kwargs={"method": "lr"},
                query=tuple(x[:2]),
                label="mrl_cb[lr]",
                **lr,
            )
        )
        out.append(
            Bound(
                "rmst_cb",
                kind="summary",
                kwargs={"method": "lr"},
                query=((_rmst_tau(case),),),
                label="rmst_cb[lr]",
                **lr,
            )
        )
    return tuple(out)


def _rmst_tau(case):
    """The horizon of a case's restricted mean: its last query time."""
    return float(np.max(np.asarray(case.x, dtype=float)))


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


def _competing_risks_bounds():
    # The cumulative incidence of each cause (#728), and the all-cause and
    # each cause's net sf / ff / Hf (the single-event estimates' bounds).
    out = []
    for bound_type in ("exp", "normal"):
        common = dict(
            kwargs={"bound_type": bound_type},
            in_range=bound_type == "exp",
            nan_ok=True,
        )
        out += [
            Bound(
                "cb",
                on=("cif",),
                per_cause=True,
                label=f"cb[cif,{bound_type}]",
                **common,
            ),
            Bound("cb", on=_ON_SURVIVAL, label=f"cb[{bound_type}]", **common),
            Bound(
                "cb",
                on=_ON_SURVIVAL,
                per_cause=True,
                label=f"cb[net,{bound_type}]",
                **common,
            ),
        ]
    return tuple(out)


_PARAM_CB = Bound("param_cb", kind="param")
_BOOT = {"n_boot": 20, "random_state": 1}


def _parameter_bounds(case):
    """``parameter_cb`` of a distributional regression, at the first
    covariate row, on its second parameter (an Exponential's only one):
    positive in every family, so above 0 as a "summary" is."""
    names = list(_DISTRIBUTIONAL_PARAMETERS[case.name])
    row = np.asarray(case.Z, dtype=float)[:1]
    query = ((names[min(1, len(names) - 1)], row),)
    out = [Bound("parameter_cb", kind="summary", query=query)]
    if case.name in _REGRESSION_LR_X:
        out += [
            Bound(
                "parameter_cb",
                kind="summary",
                query=query,
                kwargs={"method": "lr"},
                label="parameter_cb[lr]",
                wald=False,
                nan_ok=True,
                rtol=1e-3,
                slow=True,
            ),
            Bound(
                "parameter_cb",
                kind="summary",
                query=query,
                kwargs={"method": "bootstrap", **_BOOT},
                label="parameter_cb[bootstrap]",
                wald=False,
                slow=True,
            ),
        ]
    return tuple(out)


#: Each distributional case's distribution's parameters.
_DISTRIBUTIONAL_PARAMETERS = {
    base + "Distributional": getattr(sp, base).parameter_names
    for base in DISTRIBUTIONAL_BASELINES
}
_DISTRIBUTIONAL_PARAMETERS.update(
    {
        base + "MixtureCure": [*getattr(sp, base).parameter_names, "lfp_p"]
        for base in CURE_LATENCIES
    }
)
_DISTRIBUTIONAL_PARAMETERS.update(
    {
        base + "ZeroInflated": [*getattr(sp, base).parameter_names, "f0"]
        for base in CURE_LATENCIES
    }
)


def _multistate_bounds(variances):
    """The bounds on each state's occupation probability, for each
    variance and bound type (a 'normal' interval can leave [0, 1])."""
    return tuple(
        Bound(
            "cb",
            point="occupancy",
            per_cause=True,
            kwargs=(
                {"bound_type": t}
                if v is None
                else {"variance": v, "bound_type": t}
            ),
            in_range=t == "exp",
            label=f"cb[{t}]" if v is None else f"cb[{v}, {t}]",
        )
        for v in variances
        for t in ("exp", "normal")
    )


def _bounds(case):
    """The uncertainty methods of ``case``'s model (see :class:`Bound`)."""
    cls = case.model_class.rsplit(".", 1)[-1]
    # Models whose bounds are a fixed set, whatever the case
    fixed = {
        "BayesianParametric": _BAYES_BOUNDS,
        # The Greenwood-type, Aalen-type (mstate) and jackknife (survfit)
        # variances; the Cox model's include its coefficients'.
        "MultiState": _multistate_bounds(("greenwood", "aalen", "robust")),
        "MultiStateProportionalHazards": _multistate_bounds(
            ("aalen", "greenwood")
        ),
        # The delta method over every transition's parameters.
        "ParametricMultiState": _multistate_bounds((None,)),
        "ParametricMultiStateRegression": _multistate_bounds((None,)),
        # The delta method over the intensities (and coefficients), Wald
        # bounds on each parameter, and on each transient state's mean
        # stay.
        "PanelMultiState": _multistate_bounds((None,))
        + (
            _PARAM_CB,
            Bound(
                "sojourn_cb",
                kind="summary",
                query=(("healthy",), ("ill",)),
            ),
        ),
        "PanelMultiStateRegression": _multistate_bounds((None,))
        + (
            _PARAM_CB,
            Bound(
                "sojourn_cb",
                kind="summary",
                query=(("healthy", [0.5]), ("ill", [1.0])),
            ),
        ),
        "RoystonParmarModel": (Bound("cb", on=_ON_SURVIVAL),),
        # Wald bounds over every parameter, k included; the profile
        # likelihood bound is on k alone; and the life along a usage path
        # (hours at a constant number of starts per hour).
        "EquivalenceModel": (
            Bound("cb", on=_ON_SURVIVAL),
            Bound("param_cb", kind="param", label="param_cb[wald]"),
            Bound(
                "param_cb",
                kind="summary",
                kwargs={"method": "lr"},
                query=(("k",),),
                wald=False,
                rtol=1e-6,
                label="param_cb[lr, k]",
            ),
            Bound(
                "cb_at_usage",
                kind="summary",
                query=((4.0, 0.0), (4.0, 0.3), (8.0, 1.0)),
                label="cb_at_usage",
            ),
            Bound(
                "quantile_cb_at_usage",
                kind="summary",
                query=((0.1, 0.0), (0.5, 0.3), (0.9, 1.0)),
                label="quantile_cb_at_usage",
            ),
        ),
        # Wald from the profile likelihood's curvature, the profile
        # likelihood-ratio interval and the case bootstrap; the curves
        # have no bounds (cb refuses, pointing to these and WeibullPH).
        "IntervalCoxPHModel": (
            _PARAM_CB,
            Bound(
                "param_cb",
                kind="param",
                kwargs={"method": "lr"},
                label="param_cb[lr]",
                wald=False,
                nan_ok=True,
                rtol=1e-3,
                slow=True,
            ),
            Bound(
                "param_cb",
                kind="param",
                kwargs={"method": "bootstrap", **_BOOT},
                label="param_cb[bootstrap]",
                wald=False,
                slow=True,
            ),
        ),
        "AalenAdditiveModel": (Bound("cb", on=_ON_SURVIVAL),),
        # The Cox model's predicted curves: R's survfit conf.types
        "SemiParametricRegressionModel": tuple(
            Bound(
                "cb",
                on=_ON_SURVIVAL,
                kwargs={"bound_type": bound_type},
                label=f"cb[{bound_type}]",
            )
            for bound_type in ("log", "log-log", "plain")
        ),
        # Each cause's own model's bounds, and the totals over the causes
        # (the causes' delta-method variances added, on the log scale).
        "CauseSpecificNHPP": (
            Bound("cif_cb", point="cif", per_cause=True),
            Bound("iif_cb", point="iif", per_cause=True),
            Bound("mtbf_cb", point="mtbf", per_cause=True),
            Bound("total_cif_cb", point="total_cif"),
            Bound("total_iif_cb", point="total_iif"),
            Bound("total_mtbf_cb", point="total_mtbf"),
        ),
        # Each cause's own model's bounds at a covariate row, and the
        # totals over the causes (variances added, on the log scale).
        "CauseSpecificProportionalIntensity": (
            Bound("cif_cb", point="cif", per_cause=True),
            Bound("iif_cb", point="iif", per_cause=True),
            Bound("total_cif_cb", point="total_cif"),
            Bound("total_iif_cb", point="total_iif"),
        ),
    }
    if cls in fixed:
        return fixed[cls]
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
                    # Clopper-Pearson's, Jeffreys' (onto the posterior
                    # median) and the likelihood-ratio interval do not
                    # close onto the estimate as alpha_ci -> 1.
                    wald=method == "wald",
                )
                for method in ("exact", "wald", "lr", "jeffreys")
            )
        return _parametric_bounds(case)
    if cls == "NonParametric":
        return _nonparametric_bounds(case)
    if cls == "CompetingRisks":
        return _competing_risks_bounds()
    if cls in (
        "ParametricCompetingRisks",
        "ParametricCompetingRisksRegression",  # (at covariate rows)
    ):
        # Wald bounds on each cause's cumulative incidence (logit) and the
        # all-cause sf / ff / Hf (log H); a cause's net sf / ff / Hf is
        # its own model's cb. (Its method="lr" and "bootstrap", the
        # cause's own, are swept on the Parametric cases: the cb_transform
        # sweep calls a per-cause bound without its cause.)
        return (
            Bound("cb", on=("cif",), per_cause=True, label="cb[cif]"),
            Bound("cb", on=_ON_SURVIVAL, label="cb[all-cause]"),
            Bound("cb", on=_ON_SURVIVAL, per_cause=True, label="cb[net]"),
        )
    if cls == "MixtureModel":
        # Wald bounds only (#651)
        return (
            Bound("cb", on=_ON_ALL),
            _PARAM_CB,
            # The median falls in the gap between the components, where
            # the density is small and the quantile's standard error 24
            # times the quantile: at alpha_ci -> 1 the interval is the
            # estimate +- 3e-5 of it.
            Bound("quantile_cb", point="qf", rtol=1e-3),
        )
    if cls == "DistributionalRegressionModel":
        # The parametric regressions' bounds, and each row's parameter
        regression = replace(
            case, model_class="surpyval.ParametricRegressionModel"
        )
        return _bounds(regression) + _parameter_bounds(case)
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
            # The restricted mean at the first covariate row
            Bound(
                "rmst_cb",
                kind="summary",
                kwargs={"method": "wald"},
                query=((_rmst_tau(case), np.asarray(case.Z, float)[:1]),),
                label="rmst_cb[wald]",
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
        boot = dict(
            kwargs={"method": "bootstrap", **_BOOT},
            wald=False,
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
            # The parametric bootstrap (#617): the calls with the same
            # n_boot and seed share one set of refits.
            Bound(
                "cb",
                on=_ON_ALL,
                query=_REGRESSION_LR_X[case.name],
                label="cb[bootstrap]",
                **boot,
            ),
            Bound(
                "param_cb", kind="param", label="param_cb[bootstrap]", **boot
            ),
            Bound(
                "quantile_cb",
                point="qf",
                label="quantile_cb[bootstrap]",
                **boot,
            ),
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
    if cls == "JointModel":
        return (Bound("cb", on=("sf", "ff"), kwargs=dict(JOINT_ITEM)),)
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
            # Wald bounds from the fit's covariance (#666)
            Bound("cb", on=_ON_SURVIVAL),
            _PARAM_CB,
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
    if name == "MultiStateProportionalHazards":
        return {"tie_method": ("efron", "breslow")}, {}, None
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
_FAST_BOUNDS += ("Weibull[bayes]", "Exponential[bayes]")


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
    start += [model.lfp_p] if model.lfp else []
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
    if cls in (
        "ParametricRegressionModel",
        "DistributionalRegressionModel",
        "FrailtyModel",
    ):
        return lambda d: fit(_no_event_level(d))
    if cls in (
        "SemiParametricRegressionModel",
        "FineGrayModel",
        "ProportionalOddsModel",
        "IntervalCoxPHModel",
        "CoxFrailtyModel",
        "RecurrentCoxModel",
        "MultiStateProportionalHazards",
        # (each transition's regression: a group with no events)
        "ParametricMultiStateRegression",
    ):
        return lambda d: fit(_no_event_level(d))
    if cls in (
        "CompetingRisksProportionalHazards",
        "ParametricCompetingRisksRegression",
    ):
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
    if cls == "CauseSpecificProportionalIntensity":
        # (the one init is every cause's: cause a's fit, moved)
        return _far_start(
            case,
            lambda m: _scaled_start(
                np.r_[m.models["a"].params, m.models["a"].coeffs]
            ),
        )
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
    if cls == "JointModel":
        return lambda d: fit(_no_event_level(d))
    if cls == "EquivalenceModel":
        # k a million times its estimate
        return _far_start(
            case, lambda m: _scaled_start(m.params, len(m.params) - 1)
        )
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
    "AalenAdditive": _CLOSED_FORM + " (least squares at each event time)",
    "KaplanMeier": _EXACT,
    "NelsonAalen": _EXACT,
    "FlemingHarrington": _EXACT,
    "CompetingRisks[Nelson-Aalen]": _EXACT,
    "CompetingRisks[Kaplan-Meier]": _EXACT,
    "CompetingRisks[tl]": _EXACT,
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
