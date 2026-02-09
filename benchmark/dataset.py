"""
Benchmark dataset: 100 quantitative finance calculation examples.

10 categories x 10 examples each. Every expected_answer is computed
using Python's math module -- nothing is hard-coded.
"""

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FinancialExample:
    question: str
    expected_answer: float
    tolerance: float
    category: str


# ---------------------------------------------------------------------------
# Helper functions (Black-Scholes & related)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Cumulative standard-normal distribution (Abramowitz & Stegun 26.2.17)."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs / 2.0)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _bs_d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _bs_d2(S, K, T, r, sigma):
    return _bs_d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def _bs_call(S, K, T, r, sigma):
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _bs_put(S, K, T, r, sigma):
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ---------------------------------------------------------------------------
# 1. option_pricing  (10 examples)
# ---------------------------------------------------------------------------

def _option_pricing() -> list[FinancialExample]:
    TOL = 0.05
    CAT = "option_pricing"
    examples = []

    # 1) ATM call
    S, K, T, r, sig = 100, 100, 1.0, 0.05, 0.20
    ans = _bs_call(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) OTM put
    S, K, T, r, sig = 100, 90, 0.5, 0.03, 0.25
    ans = _bs_put(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European put option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Deep ITM call
    S, K, T, r, sig = 150, 100, 1.0, 0.05, 0.30
    ans = _bs_call(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Deep OTM call
    S, K, T, r, sig = 80, 120, 0.25, 0.08, 0.40
    ans = _bs_call(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Long-dated put
    S, K, T, r, sig = 100, 110, 3.0, 0.04, 0.35
    ans = _bs_put(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European put option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) High-vol call
    S, K, T, r, sig = 50, 55, 0.5, 0.06, 0.50
    ans = _bs_call(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Low-vol put
    S, K, T, r, sig = 200, 190, 1.0, 0.02, 0.10
    ans = _bs_put(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European put option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Short-dated call
    S, K, T, r, sig = 105, 100, 0.25, 0.10, 0.15
    ans = _bs_call(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) ITM put
    S, K, T, r, sig = 90, 100, 0.75, 0.05, 0.20
    ans = _bs_put(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European put option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) High-rate call
    S, K, T, r, sig = 120, 130, 2.0, 0.10, 0.25
    ans = _bs_call(S, K, T, r, sig)
    examples.append(FinancialExample(
        question=f"Calculate the Black-Scholes price of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 2. greeks  (10 examples)
# ---------------------------------------------------------------------------

def _greeks() -> list[FinancialExample]:
    TOL = 0.005
    CAT = "greeks"
    examples = []

    # 1) Delta of ATM call
    S, K, T, r, sig = 100, 100, 1.0, 0.05, 0.20
    d1 = _bs_d1(S, K, T, r, sig)
    ans = _norm_cdf(d1)
    examples.append(FinancialExample(
        question=f"Calculate the delta of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Delta of OTM put
    S, K, T, r, sig = 100, 110, 0.5, 0.03, 0.25
    d1 = _bs_d1(S, K, T, r, sig)
    ans = _norm_cdf(d1) - 1.0
    examples.append(FinancialExample(
        question=f"Calculate the delta of a European put option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Gamma
    S, K, T, r, sig = 100, 100, 1.0, 0.05, 0.20
    d1 = _bs_d1(S, K, T, r, sig)
    ans = _norm_pdf(d1) / (S * sig * math.sqrt(T))
    examples.append(FinancialExample(
        question=f"Calculate the gamma of a European option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Vega (per 1% move)
    S, K, T, r, sig = 100, 100, 1.0, 0.05, 0.20
    d1 = _bs_d1(S, K, T, r, sig)
    ans = S * _norm_pdf(d1) * math.sqrt(T) / 100.0
    examples.append(FinancialExample(
        question=f"Calculate the vega (per 1% change in volatility) of a European option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Theta of call
    S, K, T, r, sig = 100, 100, 1.0, 0.05, 0.20
    d1 = _bs_d1(S, K, T, r, sig)
    d2 = _bs_d2(S, K, T, r, sig)
    ans = -(S * _norm_pdf(d1) * sig) / (2.0 * math.sqrt(T)) - r * K * math.exp(-r * T) * _norm_cdf(d2)
    examples.append(FinancialExample(
        question=f"Calculate the theta (per year) of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) Rho of call
    S, K, T, r, sig = 100, 100, 1.0, 0.05, 0.20
    d2 = _bs_d2(S, K, T, r, sig)
    ans = K * T * math.exp(-r * T) * _norm_cdf(d2)
    examples.append(FinancialExample(
        question=f"Calculate the rho of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Gamma of OTM option
    S, K, T, r, sig = 50, 60, 0.5, 0.04, 0.30
    d1 = _bs_d1(S, K, T, r, sig)
    ans = _norm_pdf(d1) / (S * sig * math.sqrt(T))
    examples.append(FinancialExample(
        question=f"Calculate the gamma of a European option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Delta of deep ITM call
    S, K, T, r, sig = 150, 100, 1.0, 0.05, 0.20
    d1 = _bs_d1(S, K, T, r, sig)
    ans = _norm_cdf(d1)
    examples.append(FinancialExample(
        question=f"Calculate the delta of a European call option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Vega of long-dated option (per 1%)
    S, K, T, r, sig = 100, 105, 2.0, 0.03, 0.25
    d1 = _bs_d1(S, K, T, r, sig)
    ans = S * _norm_pdf(d1) * math.sqrt(T) / 100.0
    examples.append(FinancialExample(
        question=f"Calculate the vega (per 1% change in volatility) of a European option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) Theta of put
    S, K, T, r, sig = 95, 100, 0.5, 0.05, 0.30
    d1 = _bs_d1(S, K, T, r, sig)
    d2 = _bs_d2(S, K, T, r, sig)
    # Theta_put = -(S*N'(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-rT)*N(-d2)
    ans = -(S * _norm_pdf(d1) * sig) / (2.0 * math.sqrt(T)) + r * K * math.exp(-r * T) * _norm_cdf(-d2)
    examples.append(FinancialExample(
        question=f"Calculate the theta (per year) of a European put option with S={S}, K={K}, T={T}, r={r}, sigma={sig}.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 3. var_portfolio  (10 examples)
# ---------------------------------------------------------------------------

def _var_portfolio() -> list[FinancialExample]:
    TOL = 100.0
    CAT = "var_portfolio"
    examples = []

    z95 = 1.6448536269514729
    z99 = 2.3263478740408408

    # 1) Single asset, 95% VaR, 1-day
    V, sigma, hp = 1_000_000, 0.02, 1
    ans = V * z95 * sigma * math.sqrt(hp / 252)
    examples.append(FinancialExample(
        question="Calculate the 1-day 95% parametric VaR for a portfolio worth $1,000,000 with daily volatility 2.0%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Single asset, 99% VaR, 10-day
    V, sigma, hp = 5_000_000, 0.015, 10
    ans = V * z99 * sigma * math.sqrt(hp / 252)
    examples.append(FinancialExample(
        question="Calculate the 10-day 99% parametric VaR for a portfolio worth $5,000,000 with daily volatility 1.5%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Two-asset 95% VaR, 1-day
    V = 2_000_000
    w1, w2 = 0.6, 0.4
    s1, s2, rho = 0.02, 0.03, 0.5
    sp = math.sqrt(w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * s1 * s2 * rho)
    ans = V * z95 * sp * math.sqrt(1 / 252)
    examples.append(FinancialExample(
        question=("Calculate the 1-day 95% parametric VaR for a $2,000,000 portfolio with two assets. "
                  "Weights: 0.6 and 0.4. Daily volatilities: 2.0% and 3.0%. Correlation: 0.5."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Two-asset 99% VaR, 5-day
    V = 10_000_000
    w1, w2 = 0.7, 0.3
    s1, s2, rho = 0.01, 0.025, 0.3
    sp = math.sqrt(w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * s1 * s2 * rho)
    hp = 5
    ans = V * z99 * sp * math.sqrt(hp / 252)
    examples.append(FinancialExample(
        question=("Calculate the 5-day 99% parametric VaR for a $10,000,000 portfolio with two assets. "
                  "Weights: 0.7 and 0.3. Daily volatilities: 1.0% and 2.5%. Correlation: 0.3."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Single asset, annual vol, 95%, 1-day
    V, annual_vol, hp = 3_000_000, 0.25, 1
    daily_vol = annual_vol / math.sqrt(252)
    ans = V * z95 * daily_vol * math.sqrt(hp / 252)
    examples.append(FinancialExample(
        question=("Calculate the 1-day 95% parametric VaR for a $3,000,000 portfolio. "
                  "The annual volatility is 25.0%. Assume 252 trading days."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) Two-asset uncorrelated, 95%, 1-day
    V = 4_000_000
    w1, w2 = 0.5, 0.5
    s1, s2, rho = 0.018, 0.022, 0.0
    sp = math.sqrt(w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * s1 * s2 * rho)
    ans = V * z95 * sp * math.sqrt(1 / 252)
    examples.append(FinancialExample(
        question=("Calculate the 1-day 95% parametric VaR for a $4,000,000 portfolio with two uncorrelated assets. "
                  "Weights: 0.5 and 0.5. Daily volatilities: 1.8% and 2.2%. Correlation: 0.0."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Single asset, 99%, 20-day
    V, sigma, hp = 500_000, 0.03, 20
    ans = V * z99 * sigma * math.sqrt(hp / 252)
    examples.append(FinancialExample(
        question="Calculate the 20-day 99% parametric VaR for a portfolio worth $500,000 with daily volatility 3.0%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Two-asset perfectly correlated, 95%, 1-day
    V = 1_500_000
    w1, w2 = 0.4, 0.6
    s1, s2, rho = 0.02, 0.025, 1.0
    sp = math.sqrt(w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * s1 * s2 * rho)
    ans = V * z95 * sp * math.sqrt(1 / 252)
    examples.append(FinancialExample(
        question=("Calculate the 1-day 95% parametric VaR for a $1,500,000 portfolio with two perfectly correlated assets. "
                  "Weights: 0.4 and 0.6. Daily volatilities: 2.0% and 2.5%. Correlation: 1.0."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Two-asset negatively correlated, 99%, 1-day
    V = 8_000_000
    w1, w2 = 0.55, 0.45
    s1, s2, rho = 0.02, 0.03, -0.4
    sp = math.sqrt(w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * s1 * s2 * rho)
    ans = V * z99 * sp * math.sqrt(1 / 252)
    examples.append(FinancialExample(
        question=("Calculate the 1-day 99% parametric VaR for an $8,000,000 portfolio with two assets. "
                  "Weights: 0.55 and 0.45. Daily volatilities: 2.0% and 3.0%. Correlation: -0.4."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) Single asset, 95%, 1-day, large portfolio
    V, sigma, hp = 50_000_000, 0.012, 1
    ans = V * z95 * sigma * math.sqrt(hp / 252)
    examples.append(FinancialExample(
        question="Calculate the 1-day 95% parametric VaR for a portfolio worth $50,000,000 with daily volatility 1.2%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 4. bond_pricing  (10 examples)
# ---------------------------------------------------------------------------

def _bond_price(face, coupon_rate, ytm, n):
    """Annual coupon bond price."""
    C = face * coupon_rate
    price = sum(C / (1 + ytm) ** t for t in range(1, n + 1)) + face / (1 + ytm) ** n
    return price


def _bond_duration(face, coupon_rate, ytm, n):
    """Macaulay duration for annual coupon bond."""
    C = face * coupon_rate
    P = _bond_price(face, coupon_rate, ytm, n)
    dur = sum(t * C / (1 + ytm) ** t for t in range(1, n + 1)) + n * face / (1 + ytm) ** n
    return dur / P


def _bond_convexity(face, coupon_rate, ytm, n):
    """Convexity for annual coupon bond."""
    C = face * coupon_rate
    P = _bond_price(face, coupon_rate, ytm, n)
    conv = sum(t * (t + 1) * C / (1 + ytm) ** (t + 2) for t in range(1, n + 1))
    conv += n * (n + 1) * face / (1 + ytm) ** (n + 2)
    return conv / P


def _bond_pricing() -> list[FinancialExample]:
    TOL = 0.10
    CAT = "bond_pricing"
    examples = []

    # 1) Par bond
    F, c, y, n = 1000, 0.05, 0.05, 10
    ans = _bond_price(F, c, y, n)
    examples.append(FinancialExample(
        question=f"Calculate the price of a {n}-year bond with face value ${F}, annual coupon rate {c*100}%, and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Premium bond
    F, c, y, n = 1000, 0.06, 0.04, 10
    ans = _bond_price(F, c, y, n)
    examples.append(FinancialExample(
        question=f"Calculate the price of a {n}-year bond with face value ${F}, annual coupon rate {c*100}%, and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Discount bond
    F, c, y, n = 1000, 0.03, 0.06, 5
    ans = _bond_price(F, c, y, n)
    examples.append(FinancialExample(
        question=f"Calculate the price of a {n}-year bond with face value ${F}, annual coupon rate {c*100}%, and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Zero-coupon bond
    F, y, n = 1000, 0.05, 10
    ans = F / (1 + y) ** n
    examples.append(FinancialExample(
        question=f"Calculate the price of a {n}-year zero-coupon bond with face value ${F} and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Duration (Macaulay)
    F, c, y, n = 1000, 0.05, 0.05, 10
    ans = _bond_duration(F, c, y, n)
    examples.append(FinancialExample(
        question=f"Calculate the Macaulay duration of a {n}-year bond with face value ${F}, annual coupon rate {c*100}%, and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) Duration of premium bond
    F, c, y, n = 1000, 0.08, 0.05, 20
    ans = _bond_duration(F, c, y, n)
    examples.append(FinancialExample(
        question=f"Calculate the Macaulay duration of a {n}-year bond with face value ${F}, annual coupon rate {c*100}%, and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Convexity
    F, c, y, n = 1000, 0.05, 0.05, 10
    ans = _bond_convexity(F, c, y, n)
    examples.append(FinancialExample(
        question=f"Calculate the convexity of a {n}-year bond with face value ${F}, annual coupon rate {c*100}%, and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Short-term bond price
    F, c, y, n = 1000, 0.04, 0.03, 3
    ans = _bond_price(F, c, y, n)
    examples.append(FinancialExample(
        question=f"Calculate the price of a {n}-year bond with face value ${F}, annual coupon rate {c*100}%, and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Long-term bond price
    F, c, y, n = 1000, 0.07, 0.06, 30
    ans = _bond_price(F, c, y, n)
    examples.append(FinancialExample(
        question=f"Calculate the price of a {n}-year bond with face value ${F}, annual coupon rate {c*100}%, and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) Duration of zero-coupon
    F, y, n = 1000, 0.05, 10
    ans = float(n)  # Macaulay duration of a zero-coupon bond equals its maturity
    examples.append(FinancialExample(
        question=f"Calculate the Macaulay duration of a {n}-year zero-coupon bond with face value ${F} and yield to maturity {y*100}%.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 5. compound_interest  (10 examples)
# ---------------------------------------------------------------------------

def _compound_interest() -> list[FinancialExample]:
    TOL = 0.01
    CAT = "compound_interest"
    examples = []

    # 1) Annual compounding
    PV, r, n, t = 10000, 0.05, 1, 10
    ans = PV * (1 + r / n) ** (n * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $10,000 invested at 5.0% annual interest, compounded annually, for 10 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Monthly compounding
    PV, r, n, t = 5000, 0.06, 12, 5
    ans = PV * (1 + r / n) ** (n * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $5,000 invested at 6.0% annual interest, compounded monthly, for 5 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Quarterly compounding
    PV, r, n, t = 25000, 0.04, 4, 15
    ans = PV * (1 + r / n) ** (n * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $25,000 invested at 4.0% annual interest, compounded quarterly, for 15 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Continuous compounding
    PV, r, t = 10000, 0.08, 10
    ans = PV * math.exp(r * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $10,000 invested at 8.0% annual interest with continuous compounding for 10 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Daily compounding
    PV, r, n, t = 1000, 0.03, 365, 20
    ans = PV * (1 + r / n) ** (n * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $1,000 invested at 3.0% annual interest, compounded daily (365 days/year), for 20 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) Semi-annual compounding
    PV, r, n, t = 50000, 0.07, 2, 8
    ans = PV * (1 + r / n) ** (n * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $50,000 invested at 7.0% annual interest, compounded semi-annually, for 8 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Continuous compounding, large PV
    PV, r, t = 100000, 0.05, 30
    ans = PV * math.exp(r * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $100,000 invested at 5.0% annual interest with continuous compounding for 30 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Monthly compounding, short term
    PV, r, n, t = 20000, 0.12, 12, 1
    ans = PV * (1 + r / n) ** (n * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $20,000 invested at 12.0% annual interest, compounded monthly, for 1 year.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Quarterly compounding, long term
    PV, r, n, t = 15000, 0.06, 4, 25
    ans = PV * (1 + r / n) ** (n * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $15,000 invested at 6.0% annual interest, compounded quarterly, for 25 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) Continuous compounding, low rate
    PV, r, t = 75000, 0.01, 5
    ans = PV * math.exp(r * t)
    examples.append(FinancialExample(
        question="Calculate the future value of $75,000 invested at 1.0% annual interest with continuous compounding for 5 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 6. loan_amortization  (10 examples)
# ---------------------------------------------------------------------------

def _loan_pmt(P, annual_rate, years):
    """Monthly payment for a fully amortizing loan."""
    r = annual_rate / 12.0
    n = years * 12
    return P * r * (1 + r) ** n / ((1 + r) ** n - 1)


def _loan_balance(P, annual_rate, years, k):
    """Remaining balance after k monthly payments."""
    r = annual_rate / 12.0
    pmt = _loan_pmt(P, annual_rate, years)
    return P * (1 + r) ** k - pmt * ((1 + r) ** k - 1) / r


def _loan_total_interest(P, annual_rate, years):
    """Total interest paid over the life of the loan."""
    pmt = _loan_pmt(P, annual_rate, years)
    n = years * 12
    return pmt * n - P


def _loan_amortization() -> list[FinancialExample]:
    TOL = 0.10
    CAT = "loan_amortization"
    examples = []

    # 1) Standard mortgage payment
    P, rate, yrs = 300000, 0.06, 30
    ans = _loan_pmt(P, rate, yrs)
    examples.append(FinancialExample(
        question="Calculate the monthly payment for a $300,000 mortgage at 6.0% annual interest over 30 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Car loan payment
    P, rate, yrs = 25000, 0.05, 5
    ans = _loan_pmt(P, rate, yrs)
    examples.append(FinancialExample(
        question="Calculate the monthly payment for a $25,000 car loan at 5.0% annual interest over 5 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Total interest on mortgage
    P, rate, yrs = 250000, 0.045, 30
    ans = _loan_total_interest(P, rate, yrs)
    examples.append(FinancialExample(
        question="Calculate the total interest paid on a $250,000 mortgage at 4.5% annual interest over 30 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Remaining balance after 60 payments
    P, rate, yrs = 200000, 0.05, 30
    k = 60
    ans = _loan_balance(P, rate, yrs, k)
    examples.append(FinancialExample(
        question="Calculate the remaining balance on a $200,000 mortgage at 5.0% annual interest (30-year term) after 60 monthly payments.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Short-term personal loan payment
    P, rate, yrs = 10000, 0.08, 3
    ans = _loan_pmt(P, rate, yrs)
    examples.append(FinancialExample(
        question="Calculate the monthly payment for a $10,000 personal loan at 8.0% annual interest over 3 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) Remaining balance after 120 payments
    P, rate, yrs = 400000, 0.04, 30
    k = 120
    ans = _loan_balance(P, rate, yrs, k)
    examples.append(FinancialExample(
        question="Calculate the remaining balance on a $400,000 mortgage at 4.0% annual interest (30-year term) after 120 monthly payments.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Total interest on car loan
    P, rate, yrs = 35000, 0.07, 6
    ans = _loan_total_interest(P, rate, yrs)
    examples.append(FinancialExample(
        question="Calculate the total interest paid on a $35,000 car loan at 7.0% annual interest over 6 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) 15-year mortgage payment
    P, rate, yrs = 350000, 0.035, 15
    ans = _loan_pmt(P, rate, yrs)
    examples.append(FinancialExample(
        question="Calculate the monthly payment for a $350,000 mortgage at 3.5% annual interest over 15 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Remaining balance after 36 payments on car loan
    P, rate, yrs = 20000, 0.06, 5
    k = 36
    ans = _loan_balance(P, rate, yrs, k)
    examples.append(FinancialExample(
        question="Calculate the remaining balance on a $20,000 car loan at 6.0% annual interest (5-year term) after 36 monthly payments.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) Total interest on 15-year mortgage
    P, rate, yrs = 500000, 0.055, 15
    ans = _loan_total_interest(P, rate, yrs)
    examples.append(FinancialExample(
        question="Calculate the total interest paid on a $500,000 mortgage at 5.5% annual interest over 15 years.",
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 7. npv_irr  (10 examples)
# ---------------------------------------------------------------------------

def _npv(rate, cashflows):
    """NPV of cashflows where cashflows[0] is at t=0."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def _npv_irr() -> list[FinancialExample]:
    TOL = 1.0
    CAT = "npv_irr"
    examples = []

    # 1) Simple project NPV
    r = 0.10
    cfs = [-100000, 30000, 35000, 40000, 45000]
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 10.0% discount rate with cash flows: "
                  "Year 0: -$100,000; Year 1: $30,000; Year 2: $35,000; Year 3: $40,000; Year 4: $45,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Even cash flows
    r = 0.08
    cfs = [-200000] + [60000] * 5
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 8.0% discount rate with an initial investment of "
                  "$200,000 followed by $60,000 per year for 5 years."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Negative NPV
    r = 0.15
    cfs = [-500000, 80000, 90000, 100000, 110000, 120000]
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 15.0% discount rate with cash flows: "
                  "Year 0: -$500,000; Year 1: $80,000; Year 2: $90,000; Year 3: $100,000; "
                  "Year 4: $110,000; Year 5: $120,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Growing cash flows
    r = 0.12
    cfs = [-150000, 20000, 40000, 60000, 80000, 100000]
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 12.0% discount rate with cash flows: "
                  "Year 0: -$150,000; Year 1: $20,000; Year 2: $40,000; Year 3: $60,000; "
                  "Year 4: $80,000; Year 5: $100,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Short project
    r = 0.05
    cfs = [-50000, 25000, 30000]
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 5.0% discount rate with cash flows: "
                  "Year 0: -$50,000; Year 1: $25,000; Year 2: $30,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) Large project, low rate
    r = 0.03
    cfs = [-1000000] + [150000] * 10
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 3.0% discount rate with an initial investment of "
                  "$1,000,000 followed by $150,000 per year for 10 years."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Irregular cash flows
    r = 0.10
    cfs = [-75000, 10000, 15000, 20000, 25000, 30000, 35000]
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 10.0% discount rate with cash flows: "
                  "Year 0: -$75,000; Year 1: $10,000; Year 2: $15,000; Year 3: $20,000; "
                  "Year 4: $25,000; Year 5: $30,000; Year 6: $35,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Two-year payback
    r = 0.07
    cfs = [-40000, 22000, 22000]
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 7.0% discount rate with cash flows: "
                  "Year 0: -$40,000; Year 1: $22,000; Year 2: $22,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Negative cash flow in middle
    r = 0.09
    cfs = [-120000, 50000, 60000, -10000, 70000, 80000]
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 9.0% discount rate with cash flows: "
                  "Year 0: -$120,000; Year 1: $50,000; Year 2: $60,000; Year 3: -$10,000; "
                  "Year 4: $70,000; Year 5: $80,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) High discount rate
    r = 0.20
    cfs = [-80000, 30000, 30000, 30000, 30000]
    ans = _npv(r, cfs)
    examples.append(FinancialExample(
        question=("Calculate the NPV of a project at a 20.0% discount rate with an initial investment of "
                  "$80,000 followed by $30,000 per year for 4 years."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 8. portfolio_metrics  (10 examples)
# ---------------------------------------------------------------------------

def _portfolio_metrics() -> list[FinancialExample]:
    TOL = 0.005
    CAT = "portfolio_metrics"
    examples = []

    # 1) Weighted return - 2 assets
    w = [0.6, 0.4]
    r = [0.12, 0.08]
    ans = sum(wi * ri for wi, ri in zip(w, r))
    examples.append(FinancialExample(
        question=("Calculate the weighted portfolio return for two assets with weights 0.6 and 0.4 "
                  "and returns 12% and 8%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Weighted return - 3 assets
    w = [0.5, 0.3, 0.2]
    r = [0.10, 0.07, 0.15]
    ans = sum(wi * ri for wi, ri in zip(w, r))
    examples.append(FinancialExample(
        question=("Calculate the weighted portfolio return for three assets with weights 0.5, 0.3, 0.2 "
                  "and returns 10%, 7%, 15%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Portfolio variance - 2 assets
    w1, w2 = 0.6, 0.4
    s1, s2, rho = 0.20, 0.30, 0.25
    ans = w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * s1 * s2 * rho
    examples.append(FinancialExample(
        question=("Calculate the portfolio variance for two assets with weights 0.6 and 0.4, "
                  "volatilities 20% and 30%, and correlation 0.25."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Portfolio standard deviation - 2 assets
    w1, w2 = 0.5, 0.5
    s1, s2, rho = 0.15, 0.25, 0.40
    var_p = w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * s1 * s2 * rho
    ans = math.sqrt(var_p)
    examples.append(FinancialExample(
        question=("Calculate the portfolio standard deviation for two equally weighted assets "
                  "with volatilities 15% and 25% and correlation 0.40."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Sharpe ratio
    Rp, Rf, sigma_p = 0.14, 0.03, 0.18
    ans = (Rp - Rf) / sigma_p
    examples.append(FinancialExample(
        question=("Calculate the Sharpe ratio for a portfolio with return 14.0%, "
                  "risk-free rate 3.0%, and volatility 18.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) Sharpe ratio - another
    Rp, Rf, sigma_p = 0.09, 0.02, 0.12
    ans = (Rp - Rf) / sigma_p
    examples.append(FinancialExample(
        question=("Calculate the Sharpe ratio for a portfolio with return 9.0%, "
                  "risk-free rate 2.0%, and volatility 12.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Portfolio variance - 3 assets
    w = [0.4, 0.35, 0.25]
    s = [0.18, 0.22, 0.30]
    rho12, rho13, rho23 = 0.30, 0.10, 0.50
    var_p = (w[0]**2 * s[0]**2 + w[1]**2 * s[1]**2 + w[2]**2 * s[2]**2
             + 2 * w[0] * w[1] * s[0] * s[1] * rho12
             + 2 * w[0] * w[2] * s[0] * s[2] * rho13
             + 2 * w[1] * w[2] * s[1] * s[2] * rho23)
    ans = var_p
    examples.append(FinancialExample(
        question=("Calculate the portfolio variance for three assets with weights 0.4, 0.35, 0.25; "
                  "volatilities 18%, 22%, 30%; and correlations rho12=0.30, rho13=0.10, rho23=0.50."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Information ratio
    Rp, Rb, te = 0.11, 0.09, 0.05
    ans = (Rp - Rb) / te
    examples.append(FinancialExample(
        question=("Calculate the information ratio for a portfolio with return 11.0%, "
                  "benchmark return 9.0%, and tracking error 5.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Sortino ratio (simplified: downside deviation given)
    Rp, Rf, dd = 0.15, 0.03, 0.10
    ans = (Rp - Rf) / dd
    examples.append(FinancialExample(
        question=("Calculate the Sortino ratio for a portfolio with return 15.0%, "
                  "risk-free rate 3.0%, and downside deviation 10.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) Weighted return - 4 assets
    w = [0.25, 0.25, 0.25, 0.25]
    r = [0.08, 0.12, 0.06, 0.10]
    ans = sum(wi * ri for wi, ri in zip(w, r))
    examples.append(FinancialExample(
        question=("Calculate the weighted portfolio return for four equally weighted assets "
                  "with returns 8%, 12%, 6%, 10%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 9. derivatives_misc  (10 examples)
# ---------------------------------------------------------------------------

def _derivatives_misc() -> list[FinancialExample]:
    TOL = 0.10
    CAT = "derivatives_misc"
    examples = []

    # 1) Put-call parity - find call from put
    S, K, T, r = 100, 100, 1.0, 0.05
    put_price = 5.57
    # C = P + S - K*exp(-rT)
    ans = put_price + S - K * math.exp(-r * T)
    examples.append(FinancialExample(
        question=("Using put-call parity, find the call price given: S=100, K=100, T=1.0, r=0.05, "
                  "and the put price is $5.57."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Put-call parity - find put from call
    S, K, T, r = 110, 105, 0.5, 0.04
    call_price = 12.30
    # P = C - S + K*exp(-rT)
    ans = call_price - S + K * math.exp(-r * T)
    examples.append(FinancialExample(
        question=("Using put-call parity, find the put price given: S=110, K=105, T=0.5, r=0.04, "
                  "and the call price is $12.30."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Forward price (no dividends)
    S, r, T = 50, 0.06, 1.0
    ans = S * math.exp(r * T)
    examples.append(FinancialExample(
        question="Calculate the forward price of an asset with spot price $50, risk-free rate 6.0%, and maturity 1.0 year (no dividends).",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Forward price with dividend yield
    S, r, q, T = 100, 0.05, 0.02, 0.5
    ans = S * math.exp((r - q) * T)
    examples.append(FinancialExample(
        question=("Calculate the forward price of an asset with spot price $100, risk-free rate 5.0%, "
                  "continuous dividend yield 2.0%, and maturity 0.5 years."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) Forward price, longer maturity
    S, r, T = 200, 0.03, 2.0
    ans = S * math.exp(r * T)
    examples.append(FinancialExample(
        question="Calculate the forward price of an asset with spot price $200, risk-free rate 3.0%, and maturity 2.0 years (no dividends).",
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) Covered call payoff
    S_T, K, premium = 55, 50, 3.0
    S_0 = 48
    ans = (S_T - S_0) + premium - max(S_T - K, 0)
    examples.append(FinancialExample(
        question=("Calculate the profit of a covered call strategy where the stock was bought at $48, "
                  "a call was sold with strike $50 for a premium of $3.0, and the stock price at expiry is $55."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Covered call payoff (OTM expiry)
    S_T, K, premium, S_0 = 45, 50, 3.0, 48
    ans = (S_T - S_0) + premium - max(S_T - K, 0)
    examples.append(FinancialExample(
        question=("Calculate the profit of a covered call strategy where the stock was bought at $48, "
                  "a call was sold with strike $50 for a premium of $3.0, and the stock price at expiry is $45."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Protective put payoff
    S_0, S_T, K, premium = 100, 85, 95, 4.0
    ans = (S_T - S_0) + max(K - S_T, 0) - premium
    examples.append(FinancialExample(
        question=("Calculate the profit of a protective put strategy where the stock was bought at $100, "
                  "a put was bought with strike $95 for a premium of $4.0, and the stock price at expiry is $85."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Protective put payoff (stock goes up)
    S_0, S_T, K, premium = 100, 120, 95, 4.0
    ans = (S_T - S_0) + max(K - S_T, 0) - premium
    examples.append(FinancialExample(
        question=("Calculate the profit of a protective put strategy where the stock was bought at $100, "
                  "a put was bought with strike $95 for a premium of $4.0, and the stock price at expiry is $120."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) Forward with dividend yield, high yield
    S, r, q, T = 150, 0.07, 0.04, 1.5
    ans = S * math.exp((r - q) * T)
    examples.append(FinancialExample(
        question=("Calculate the forward price of an asset with spot price $150, risk-free rate 7.0%, "
                  "continuous dividend yield 4.0%, and maturity 1.5 years."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# 10. depreciation_tax  (10 examples)
# ---------------------------------------------------------------------------

def _depreciation_tax() -> list[FinancialExample]:
    TOL = 0.10
    CAT = "depreciation_tax"
    examples = []

    # 1) Straight-line depreciation
    cost, salvage, life = 100000, 10000, 10
    ans = (cost - salvage) / life
    examples.append(FinancialExample(
        question=("Calculate the annual straight-line depreciation for an asset costing $100,000 "
                  "with a salvage value of $10,000 and a useful life of 10 years."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 2) Straight-line depreciation - another
    cost, salvage, life = 50000, 5000, 5
    ans = (cost - salvage) / life
    examples.append(FinancialExample(
        question=("Calculate the annual straight-line depreciation for an asset costing $50,000 "
                  "with a salvage value of $5,000 and a useful life of 5 years."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 3) Declining balance depreciation - year 1
    cost, rate, year = 80000, 0.20, 1
    ans = cost * (1 - rate) ** (year - 1) * rate
    examples.append(FinancialExample(
        question=("Calculate the declining balance depreciation for year 1 of an asset costing $80,000 "
                  "with a depreciation rate of 20.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 4) Declining balance depreciation - year 3
    cost, rate, year = 80000, 0.20, 3
    ans = cost * (1 - rate) ** (year - 1) * rate
    examples.append(FinancialExample(
        question=("Calculate the declining balance depreciation for year 3 of an asset costing $80,000 "
                  "with a depreciation rate of 20.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 5) After-tax cash flow
    revenue, tax_rate, depreciation = 200000, 0.30, 15000
    ans = revenue * (1 - tax_rate) + depreciation * tax_rate
    examples.append(FinancialExample(
        question=("Calculate the after-tax cash flow given revenue of $200,000, "
                  "tax rate of 30.0%, and depreciation of $15,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 6) After-tax cash flow - higher tax
    revenue, tax_rate, depreciation = 500000, 0.35, 40000
    ans = revenue * (1 - tax_rate) + depreciation * tax_rate
    examples.append(FinancialExample(
        question=("Calculate the after-tax cash flow given revenue of $500,000, "
                  "tax rate of 35.0%, and depreciation of $40,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 7) Tax shield from interest
    interest, tax_rate = 50000, 0.30
    ans = interest * tax_rate
    examples.append(FinancialExample(
        question=("Calculate the tax shield from an interest expense of $50,000 "
                  "with a tax rate of 30.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 8) Tax shield - larger
    interest, tax_rate = 120000, 0.25
    ans = interest * tax_rate
    examples.append(FinancialExample(
        question=("Calculate the tax shield from an interest expense of $120,000 "
                  "with a tax rate of 25.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 9) Effective tax rate
    tax_paid, pre_tax_income = 45000, 180000
    ans = tax_paid / pre_tax_income
    examples.append(FinancialExample(
        question=("Calculate the effective tax rate if a company paid $45,000 in taxes "
                  "on a pre-tax income of $180,000."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    # 10) Declining balance - book value after 5 years
    cost, rate, n = 120000, 0.25, 5
    ans = cost * (1 - rate) ** n
    examples.append(FinancialExample(
        question=("Calculate the book value after 5 years for an asset costing $120,000 "
                  "with a declining balance depreciation rate of 25.0%."),
        expected_answer=ans, tolerance=TOL, category=CAT))

    return examples


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_dataset() -> list[FinancialExample]:
    examples = []
    examples.extend(_option_pricing())
    examples.extend(_greeks())
    examples.extend(_var_portfolio())
    examples.extend(_bond_pricing())
    examples.extend(_compound_interest())
    examples.extend(_loan_amortization())
    examples.extend(_npv_irr())
    examples.extend(_portfolio_metrics())
    examples.extend(_derivatives_misc())
    examples.extend(_depreciation_tax())
    return examples


# ---------------------------------------------------------------------------
# DSPy conversion helper
# ---------------------------------------------------------------------------

def as_dspy_examples(examples: list[FinancialExample]) -> list:
    import dspy
    return [
        dspy.Example(question=ex.question, answer=str(round(ex.expected_answer, 4))).with_inputs("question")
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset():
    ds = load_dataset()
    assert len(ds) == 100, f"Expected 100, got {len(ds)}"
    for ex in ds:
        assert math.isfinite(ex.expected_answer), f"Bad: {ex.question}"
        assert ex.tolerance > 0
    cats = set(ex.category for ex in ds)
    assert len(cats) == 10, f"Expected 10 categories, got {len(cats)}"
    for cat in cats:
        count = sum(1 for ex in ds if ex.category == cat)
        assert count == 10, f"Category {cat} has {count} examples, expected 10"
    print(f"Dataset OK: {len(ds)} examples, {len(cats)} categories")
    for cat in sorted(cats):
        print(f"  {cat}: 10 examples")


if __name__ == "__main__":
    verify_dataset()
