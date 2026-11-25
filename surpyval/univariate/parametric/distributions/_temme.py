r"""
The tails of a gamma distribution with a large shape, by Temme's uniform
asymptotic expansion (DLMF 8.12), for the generalised gamma near its
lognormal limit.

The generalised gamma in Prentice's form puts a gamma variable of shape
:math:`a = 1 / Q^2` at :math:`x = a e^{Q w}`. As :math:`Q \to 0` the
shape grows without bound and :math:`x / a \to 1`, where the regularised
incomplete gamma functions of scipy lose their accuracy (a log cdf of -15.49
where it is -15.06 at :math:`Q = 10^{-4}`, :math:`w = -5`) and their
shape derivatives, taken by differences in :math:`a` and scaled by
:math:`da/dQ = -2/Q^3`, lose more. Temme's expansion is written in
:math:`(w, Q)` itself and is smooth through :math:`Q = 0`, where it is
exactly the standard normal:

.. math::
    S = \tfrac12 \operatorname{erfc}(\zeta) + \frac{Q\, e^{-\zeta^2}}
    {\sqrt{2\pi}} \sum_{k \ge 0} c_k(\eta)\, Q^{2k},
    \qquad \zeta = w \sqrt{h(Q w)}, \quad \eta = Q w \sqrt{2 h(Q w)},

with :math:`h(y) = (e^y - 1 - y) / y^2`, for the survival function of the
standardised log time :math:`w`, whichever the sign of :math:`Q`. The
coefficients :math:`c_k(\eta)` are Taylor series in :math:`\eta`, from
the recursion :math:`c_0 = 1/(\lambda - 1) - 1/\eta`, :math:`c_k =
\eta^{-1} c_{k-1}'(\eta) + (-1)^k \gamma_k / (\lambda - 1)` (DLMF 8.12.11;
:math:`\gamma_k` the coefficients of Stirling's series of
:math:`\Gamma^*`), computed in 80-digit arithmetic by
``scripts/temme_coefficients.py``. They are the table scipy's own
incomplete gamma uses for large shapes (``igam.c``).

Eight terms of the sum and thirty of each series keep the expansion to
about 1e-16 of the tail for :math:`|Q| < 0.1` and :math:`|\eta| \le 1`
(the series converge for :math:`|\eta| < 2 \sqrt{\pi}`); everything is
autograd numpy, so its derivatives in :math:`w` and :math:`Q` are
exact derivatives of the expansion.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy as onp

from surpyval.univariate.parametric.parametric_fitter import Boxable

from ._stable import _erfcx

#: The Taylor coefficients of :math:`c_k(\eta)`, row k, column n the
#: coefficient of :math:`\eta^n` (DLMF 8.12.11; scipy's ``igam.c`` table).
TEMME = onp.array(
    [
        (
            -0.3333333333333333,
            0.08333333333333333,
            -0.014814814814814815,
            0.0011574074074074073,
            0.0003527336860670194,
            -0.0001787551440329218,
            3.919263178522438e-05,
            -2.185448510679992e-06,
            -1.85406221071516e-06,
            8.296711340953087e-07,
            -1.7665952736826078e-07,
            6.707853543401498e-09,
            1.0261809784240309e-08,
            -4.382036018453353e-09,
            9.14769958223679e-10,
            -2.5514193994946248e-11,
            -5.830772132550426e-11,
            2.4361948020667415e-11,
            -5.0276692801141755e-12,
            1.1004392031956135e-13,
            3.371763262400985e-13,
            -1.392388722418162e-13,
            2.8534893807047445e-14,
            -5.139111834242572e-16,
            -1.9752288294349442e-15,
            8.099521156704561e-16,
            -1.6522531216398162e-16,
            2.5305430097478883e-18,
            1.1686939738559576e-17,
            -4.770037049820485e-18,
        ),
        (
            -0.001851851851851852,
            -0.003472222222222222,
            0.0026455026455026454,
            -0.0009902263374485596,
            0.00020576131687242798,
            -4.018775720164609e-07,
            -1.8098550334489977e-05,
            7.64916091608111e-06,
            -1.6120900894563446e-06,
            4.647127802807434e-09,
            1.378633446915721e-07,
            -5.752545603517705e-08,
            1.1951628599778148e-08,
            -1.7543241719747647e-11,
            -1.0091543710600413e-09,
            4.162792991842583e-10,
            -8.56390702649298e-11,
            6.067215101604758e-14,
            7.1624989648114856e-12,
            -2.933186643771437e-12,
            5.996696365683689e-13,
            -2.1671786527323313e-16,
            -4.978339972369262e-14,
            2.0291628823713425e-14,
            -4.13125571381061e-15,
            8.286516239883097e-19,
            3.4100308869333327e-16,
            -1.3854195302893971e-16,
            2.812346653228875e-17,
            -3.406444194143029e-21,
        ),
        (
            0.004133597883597883,
            -0.0026813271604938273,
            0.0007716049382716049,
            2.0093878600823047e-06,
            -0.0001073665322636516,
            5.2923448829120125e-05,
            -1.2760635188618728e-05,
            3.423578734096138e-08,
            1.3721957309062934e-06,
            -6.298992138380055e-07,
            1.4280614206064242e-07,
            -2.0477098421990866e-10,
            -1.409252991086752e-08,
            6.228974084922022e-09,
            -1.3670488396617114e-09,
            9.428356159014678e-13,
            1.2872252400089318e-10,
            -5.5645956134363323e-11,
            1.197593554636698e-11,
            -4.1689782251838634e-15,
            -1.0940640427884595e-12,
            4.662239946390136e-13,
            -9.905105763906907e-14,
            1.8931876768373515e-17,
            8.859221872591127e-15,
            -3.737820398046405e-15,
            7.868833639035156e-16,
            -9.000027395741211e-20,
            -6.928881229347671e-17,
            2.9020384270164786e-17,
        ),
        (
            0.0006494341563786008,
            0.00022947209362139917,
            -0.0004691894943952557,
            0.00026772063206283885,
            -7.561801671883977e-05,
            -2.396505113867297e-07,
            1.1082654115347302e-05,
            -5.6749528269915965e-06,
            1.4230900732435883e-06,
            -2.7861080291528143e-11,
            -1.6958404091930278e-07,
            8.099464905388083e-08,
            -1.9111168485973655e-08,
            2.3928620439808118e-12,
            2.0620131815488797e-09,
            -9.460496661855133e-10,
            2.1541049775774907e-10,
            -1.388823336813903e-14,
            -2.1894761681963938e-11,
            9.790998951171684e-12,
            -2.178219188018096e-12,
            6.208819573407901e-17,
            2.126978363279737e-13,
            -9.344688791517433e-14,
            2.045367122678285e-14,
            -2.58260790403495e-19,
            -1.9405297673344544e-15,
            8.415979290484816e-16,
            -1.8200430439538226e-16,
            1.0735443641247309e-21,
        ),
        (
            -0.0008618882909167117,
            0.0007840392217200666,
            -0.0002990724803031902,
            -1.4638452578843418e-06,
            6.641498215465122e-05,
            -3.968365047179435e-05,
            1.1375726970678419e-05,
            2.507497226237533e-10,
            -1.6954149536558305e-06,
            8.907507532205309e-07,
            -2.292934834000805e-07,
            2.956794137544049e-11,
            2.8865829742708783e-08,
            -1.4189739437803219e-08,
            3.4463580499464896e-09,
            -2.3024517174528067e-13,
            -3.9409233028046403e-10,
            1.86023389685045e-10,
            -4.356323005056618e-11,
            1.278600101629623e-15,
            4.67927502665792e-12,
            -2.149246470613483e-12,
            4.908815614809652e-13,
            -6.33859148489156e-18,
            -5.045332069080094e-14,
            2.2722958222901286e-14,
            -5.096082608472402e-15,
            3.0552097557171355e-20,
            5.069021676310552e-16,
            -2.249383695648181e-16,
        ),
        (
            -0.00033679855336635813,
            -6.972813758365857e-05,
            0.0002772753244959392,
            -0.00019932570516188847,
            6.797780477937208e-05,
            1.419062920643967e-07,
            -1.3594048189768693e-05,
            8.018470256334202e-06,
            -2.291481176508095e-06,
            -3.252473551298454e-10,
            3.4652846491085265e-07,
            -1.8447187191171344e-07,
            4.8240967037894184e-08,
            -1.7989466721743514e-14,
            -6.306194500013523e-09,
            3.162417628774568e-09,
            -7.840924253697429e-10,
            5.192679165254041e-15,
            9.358944242306784e-11,
            -4.513426216163278e-11,
            1.0799129993116828e-11,
            -3.661886712685252e-17,
            -1.210902069055155e-12,
            5.680743584990564e-13,
            -1.3249659916340829e-13,
            1.8987240764284076e-19,
            1.4193390236794701e-14,
            -6.523214701424697e-15,
            1.4925242636202885e-15,
            -8.800389458732369e-22,
        ),
        (
            0.0005313079364639922,
            -0.0005921664373536939,
            0.0002708782096718045,
            7.902353232660328e-07,
            -8.153969367561969e-05,
            5.61168275310625e-05,
            -1.8329116582843375e-05,
            -3.0796134506033047e-09,
            3.465155368803609e-06,
            -2.0291327396058603e-06,
            5.788792863149004e-07,
            2.338630673826657e-13,
            -8.828600746330484e-08,
            4.7435958880408125e-08,
            -1.2545415020710383e-08,
            8.649648858010293e-14,
            1.6846058979264062e-09,
            -8.575492823577594e-10,
            2.1598224929232125e-10,
            -7.613230520476153e-16,
            -2.6639822008536144e-11,
            1.3065700536611057e-11,
            -3.1799163902367977e-12,
            4.710976121367431e-18,
            3.6902800842763465e-13,
            -1.7612674046201426e-13,
            4.179066786051478e-14,
            -2.5344679379178804e-20,
            -4.632065942001605e-15,
            2.165145485964643e-15,
        ),
        (
            0.00034436760689237765,
            5.171790908260592e-05,
            -0.00033493161081142234,
            0.0002812695154763237,
            -0.00010976582244684731,
            -1.2741009095484485e-07,
            2.7744451511563645e-05,
            -1.8263488805711332e-05,
            5.7876949497350525e-06,
            4.93875893393627e-10,
            -1.0595367014026043e-06,
            6.166714376110408e-07,
            -1.7562973359060463e-07,
            -1.297447328701544e-12,
            2.695423606288966e-08,
            -1.4578352908731272e-08,
            3.887645959386175e-09,
            -3.881002251019412e-17,
            -5.327994173877286e-10,
            2.7437977643314844e-10,
            -6.995796092070568e-11,
            2.589986387486848e-17,
            8.856689099669639e-12,
            -4.403168815871311e-12,
            1.0865561947091654e-12,
            -2.0467988447416678e-19,
            -1.2969794421692939e-13,
            6.278922059147728e-14,
            -1.5112948371679396e-14,
            1.1708345473797399e-21,
        ),
    ]
)

#: Where the expansion is used: ``|Q|`` below this, and ``|eta|`` below
#: ``ETA_MAX``.
Q_MAX = 0.1
ETA_MAX = 1.0

_SQRT_2PI = float(onp.sqrt(2.0 * onp.pi))

#: :math:`1 / (n + 2)!`, the Taylor coefficients of :math:`h(y)`.
_H_SERIES = tuple(
    1.0 / float(onp.prod(onp.arange(1, n + 3))) for n in range(14)
)
#: Below this :math:`|y|` the series of :math:`h`; above it, from expm1
#: (which loses at most :math:`2 \epsilon / |y|` to cancellation there).
_H_SMALL = 0.1


def h(y: Boxable) -> Boxable:
    r""":math:`h(y) = (e^y - 1 - y) / y^2`, with :math:`h(0) = 1/2`:
    positive everywhere, and ``q (e^{Qw} - 1 - Qw) = w^2 h(Qw)`` for
    ``q = 1/Q^2``."""
    small = np.abs(y) < _H_SMALL
    if np.all(small):
        out: Any = _H_SERIES[-1]
        for c in reversed(_H_SERIES[:-1]):
            out = out * y + c
        return out
    ys = np.where(small, y, 0.0)
    yl = np.where(small, 1.0, y)
    series: Any = _H_SERIES[-1]
    for c in reversed(_H_SERIES[:-1]):
        series = series * ys + c
    with onp.errstate(over="ignore"):
        direct = (np.expm1(yl) - yl) / yl**2
    return np.where(small, series, direct)


def _sum(eta: Boxable, Q: Boxable) -> Boxable:
    r""":math:`\sum_k c_k(\eta) Q^{2k}`."""
    q2 = Q * Q
    total = 0.0
    for row in reversed(TEMME):
        ck = row[-1]
        for c in reversed(row[:-1]):
            ck = ck * eta + c
        total = total * q2 + ck
    return total


def small_tail(w: Boxable, Q: Boxable) -> tuple[Boxable, Boxable]:
    r"""
    The log of the smaller of the survival and distribution functions of
    the standardised log time :math:`w` of a generalised gamma with shape
    :math:`Q`, and whether it is the survival function (``zeta >= 0``).

    For :math:`\zeta \geq 0`, :math:`\ln S = -\zeta^2 + \ln(\tfrac12
    \operatorname{erfcx}(\zeta) + Q \Sigma / \sqrt{2\pi})`, and for
    :math:`\zeta < 0` the same with :math:`|\zeta|` and :math:`-Q` is
    :math:`\ln F`: no term underflows or cancels in either tail.
    """
    log_bracket, upper, hy = _log_bracket(w, Q)
    return -(w * w) * hy + log_bracket, upper


def _log_bracket(w: Boxable, Q: Boxable) -> tuple[Boxable, Boxable, Boxable]:
    r"""``small_tail`` without its :math:`-\zeta^2`: the log of the
    bracket, whether it is the survival function, and :math:`h(Q w)`."""
    y = Q * w
    hy = h(y)
    root = np.sqrt(hy)
    zeta = w * root
    eta = y * root * onp.sqrt(2.0)
    upper = zeta >= 0
    sign = np.where(upper, 1.0, -1.0)
    bracket = 0.5 * _erfcx(np.abs(zeta)) + sign * Q * _sum(eta, Q) / _SQRT_2PI
    return np.log(bracket), upper, hy


def small_tail_gamma(
    w: Boxable, Q: Boxable, zeta2: Boxable
) -> tuple[Boxable, Boxable]:
    r"""``small_tail`` with :math:`\zeta^2` given: for a gamma of shape
    :math:`a = 1 / Q^2` at :math:`x`, :math:`\zeta^2 = a (x / a - 1 -
    \ln(x / a))`, the deviance that the saddle-point forms compute to a
    few ulps (``surpyval.utils.saddle_point.bd0(a, x)``), where ``w^2
    h(Q w)`` carries the rounding of four products."""
    log_bracket, upper, _ = _log_bracket(w, Q)
    return -zeta2 + log_bracket, upper


def eta_of(w: Boxable, Q: Boxable) -> Boxable:
    r""":math:`\eta` at ``w`` and ``Q`` (plain values: for choosing
    where the expansion holds)."""
    y = onp.asarray(Q * w, dtype=float)
    return y * onp.sqrt(2.0 * h(y))
