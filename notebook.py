

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Term Structure Models""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Mertons' Model

        $$
        dr_t = \\mu dt + σ dW_t
        $$ 


        Here $\\mu$ and σ are constants and $W_t$ is a standard Brownian Motion. 
        This model was actually proposed by Merton in his classic 1973 paper *Rational Option Pricing*, in footnote 43. Here is his comment on the model

        > Although this process is not realistic becuase it implies a positive probability of negative interest rates, ...


        This will be a good model to cut our teeth on before we expand to more realistic models. One reason is this model admits a closed form solution to we can check our tree and simulation results against the closed form result.

        We will price zero coupon bonds using this model three different ways:

        1. Closed form solution to the SDE
        2. BinomialTree Tree approximation
        3. Monte-Carlo simulation

        \\begin{aligned}
        r_t &= \\mu t + \\sigma W_t \\\\
        P(t, T) &= E_t^Q \\left[e^{-\\int_t^T r_s ds}\\right] \\\\
        P(t, T) &= e^{-r_t (T-t) -\\frac{1}{2} \\mu (T-t)^2 + \\frac{1}{6} \\sigma^2 (T-t)^3}
        \\end{aligned}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Vasicek
        The Vasicek model is the continuous time analogue of an AR(1) process. It is also an *Affine Term Structure Model*. The short rate is mean reverting to $\frac{b}{a}$ and evolves according to 

        $$
        dr_t = k(\\theta - r_t) dt + σ dW_t, \\quad (k>0)
        $$ 

        The zero coupon bond prices can be calculated analytically and the formula is

        \\begin{aligned}
        P(t,T) &= e^{A(t,T)-B(t,T)r_t} \\\\
        B(t,T) &= \\frac{1}{k} \\big( 1 - e^{-k(T-t)} \\big)\\\\
        A(t,T) &=  \\left( B(t,T) - T + t \\right) \\left(\\theta - \\frac{\\sigma^2}{2k^2} \\right) - \\frac{\\sigma^2 B^2 (t,T)}{4k} 
        \\end{aligned}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Cox-Ingersol-Ross (CIR)
        The CIR model is another affine term structure model with a closed form solution for zero coupon bond prices.

        $$
        dr_t = k (\theta -r_t)dt + \sigma \sqrt{r_t} dW_t
        $$

        The difference from Vasicek being the $\sqrt{r_t}$ scaling factor on the diffusion term.

        $$
        \begin{aligned}
         P(t,T) &= A(t,T)e^{-B(t,T)r_t}\\
        A(t,T) &= \left[ \frac{2h \exp((k+h)(T-t)/2)}{2h + (k+h)(\exp((T-t)h)-1}  \right ]^{2k\theta/ \sigma^2}\\
        B(t, T) &= \frac{2(\exp((T-t)h)-1}{2h+(k+h)(exp((T-t)h)-1} \\
        h &= \sqrt{k^2 + 2\sigma^2}   m
        \end{aligned}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## G2 Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## CIR2 Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## GV2+ Model
        A  Guassain variant of Salomon Brothers 2+ model

        \begin{aligned}
        dx_t &= \lambda dt + \sigma_x dW_x \\
        dy_t &= -\gamma y_t dt + \sigma_y dW_y \\
        dz_t &= -k(z_t - x_t - y_t) \\
        r_t &= z_t + \phi\\
        \end{aligned}

        We calculate the price of a zero coupon bond, the volatility terms in $A(t,T)$ are approximations that for all practical purposes should be fine.

        \begin{aligned}
        P(t, T) &= \exp\big(A(t,T) - B(t,T)z_t -C(t,T)x_t -D(t,T)y_t\big)\\
        A(t,T)&≈−ϕ(T−t)−λ\left[\frac{(T−t)^2}{2}−\frac{(T−t)}{k}+\frac{B(t,T)}{k}\right]−\frac{\sigma_x^2}{6}(T−t)^3−\frac{\sigma_y^2}{2 \gamma^2}\left[T−t−\frac{2}{\gamma}\left(1−e^{γ(T−t)}\right)\right]\\
        B(t,T)&= \frac{1−e^{−k(T−t)}}{k} \\
        C(t,T) &= (T-t) - B(t, T) \\
        D(t,T) &= \frac{1-e^{-\gamma(T-t)}}{\gamma}-\frac{(1-e^{-\gamma(T-t)})}{\gamma(k-\gamma)}-\frac{B(t,T)}{k-\gamma} \\
        \end{aligned}
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    return mo, np, plt


@app.cell
def _():
    from src.core.calibration import Calibrator
    from src.engines.closed_form import ClosedFormCIR, ClosedFormG2, ClosedFormVasicek, ClosedFormCIR2, ClosedFormMerton, ClosedFormGV2P
    from src.engines.binomial_tree import BinomialVasicek, BinomialMerton
    from src.engines.monte_carlo import MonteCarloMerton, MonteCarloVasicek
    from src.models.cir import CIR, CIR
    from src.models.g2 import G2, G2
    from src.models.vasicek import Vasicek
    from src.models.cir2 import CIR2
    from src.models.merton import Merton
    from src.models.gv2p import GV2P
    from src.optim.least_squares import SciPyLeastSquares

    return (
        CIR,
        CIR2,
        Calibrator,
        ClosedFormCIR,
        ClosedFormCIR2,
        ClosedFormG2,
        ClosedFormGV2P,
        ClosedFormMerton,
        ClosedFormVasicek,
        G2,
        GV2P,
        Merton,
        SciPyLeastSquares,
        Vasicek,
    )


@app.cell
def _(plt):
    def plot_yield_curve(maturities, merton_yields, vas_yields, cir_yields, g2_yields, cir2_yields, gv2p_yields, market_data, factors: int = 1):
        fig, ax = plt.subplots(figsize=(11, 6))
        if factors == 1:
            ax.plot(maturities, 100 * merton_yields)
            ax.plot(maturities, 100 * vas_yields)
            ax.plot(maturities, 100 * cir_yields)
        elif factors == 2:
            ax.plot(maturities, 100 * g2_yields)
            ax.plot(maturities, 100 * cir2_yields)
            ax.plot(maturities, 100 * gv2p_yields)
        t_mkt, y_mkt = zip(*market_data)
        y_mkt = [y * 100.0 for y in y_mkt]
        ax.scatter(t_mkt, y_mkt,
               s = 20.0,
               marker="o",
               zorder=3,
               label="Market points", 
               color="black")
        ax.set_xlabel("Maturity (years)", labelpad=10)
        ax.set_ylabel("Continuously Compounded Spot Rate", labelpad=10)
        ax.xaxis.grid(alpha=0.7, linewidth=0.5)
        ax.yaxis.grid(alpha=0.7, linewidth=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        if factors == 1:
            ax.legend(["Merton", "Vasicek", "CIR"], edgecolor='none', borderpad=2)
            ax.set_title( "One Factor Short Rate Models", fontsize=20, pad=15)
        elif factors == 2:
            ax.legend(["G2", "CIR2", "GV2P"], edgecolor='none', borderpad=2)
            ax.set_title( "Two Factor Short Rate Models", fontsize=20, pad=15)
        fig.tight_layout()
        plt.show()

    return (plot_yield_curve,)


@app.cell
def _(
    CIR,
    CIR2,
    Calibrator,
    ClosedFormCIR,
    ClosedFormCIR2,
    ClosedFormG2,
    ClosedFormGV2P,
    ClosedFormMerton,
    ClosedFormVasicek,
    G2,
    GV2P,
    Merton,
    SciPyLeastSquares,
    Vasicek,
):
    # market_data = [(0.25, 0.04), (2.0, 0.041), (10.0, 0.045), (30.0, 0.05)]
    # market_data = [(1.0, 0.0175), (5.0, 0.0155), (10.0, 0.0168), (30.0, 0.0212)]
    market_data = [(0.25, 0.04), (1.0, 0.0398), (2.0, 0.0382), (5.0, 0.0391), (10.0, 0.0411), (15.0, 0.0425), (20.0, 0.0430), (29.0, 0.0435), (30.0, 0.0436)]

    optimizer = SciPyLeastSquares()

    merton_model = Merton(r0=0.04, mu=0.0, sigma=0.01)
    merton_engine = ClosedFormMerton()
    merton_calib = Calibrator(merton_model, merton_engine, optimizer)
    merton_calib.calibrate(market_data)

    g2_model = G2(x0=0.0, y0=0.0, a=0.02, b=0.01, rho=-0.5, phi=0.03, sigma_x=0.05, sigma_y=0.01)
    g2_engine = ClosedFormG2()
    g2_calib = Calibrator(g2_model, g2_engine, optimizer)
    g2_calib.calibrate(market_data)

    cir_model = CIR(r0=0.0, kappa=0.0, theta=0.00, sigma=0.01)
    cir_engine = ClosedFormCIR()
    cir_calib = Calibrator(cir_model, cir_engine, optimizer)
    cir_calib.calibrate(market_data)

    vas_model = Vasicek(r0=0.04, kappa=0.01, theta=0.1, sigma=0.01)
    vas_engine = ClosedFormVasicek()
    vas_calib = Calibrator(vas_model, vas_engine, optimizer)
    vas_calib.calibrate(market_data)

    cir2_model = CIR2(r0_1=0.01, r0_2=0.01, kappa1=0.02, kappa2=0.01, theta1=0.1, 
                              theta2=0.2, sigma_x=0.05, sigma_y=0.05)
    cir2_engine = ClosedFormCIR2()
    cir2_calib = Calibrator(cir2_model, cir2_engine, optimizer)
    cir2_calib.calibrate(market_data)

    gv2p_model = GV2P(x0=0.0, y0=0.0, z0=0.0, gamma=0.5, k=0.2,lambda_=0.01,phi=0.005, sigma_x=0.005, sigma_y=0.03)
    gv2p_engine = ClosedFormGV2P()
    gv2p_calib = Calibrator(gv2p_model, gv2p_engine, optimizer)
    gv2p_calib.calibrate(market_data)
    return (
        cir2_engine,
        cir2_model,
        cir_engine,
        cir_model,
        g2_engine,
        g2_model,
        gv2p_engine,
        gv2p_model,
        market_data,
        merton_engine,
        merton_model,
        vas_engine,
        vas_model,
    )


@app.cell
def _(vas_model):
    print(vas_model)
    return


@app.cell
def _(
    cir2_engine,
    cir2_model,
    cir_engine,
    cir_model,
    g2_engine,
    g2_model,
    gv2p_engine,
    gv2p_model,
    merton_engine,
    merton_model,
    np,
    vas_engine,
    vas_model,
):
    maturities = np.linspace(0.25, 30, 120)

    merton_prices = [merton_engine.P(merton_model, t) for t in maturities]
    merton_yields = merton_engine.spot_rate(merton_prices, maturities)

    g2_prices = [g2_engine.P(g2_model, t) for t in maturities]
    g2_yields = g2_engine.spot_rate(g2_prices, maturities)

    cir_prices = [cir_engine.P(cir_model, t) for t in maturities]
    cir_yields = cir_engine.spot_rate(cir_prices, maturities)

    vas_prices = [vas_engine.P(vas_model, t) for t in maturities]
    vas_yields = vas_engine.spot_rate(vas_prices, maturities)

    cir2_prices = [cir2_engine.P(cir2_model, t) for t in maturities]
    cir2_yields = cir2_engine.spot_rate(cir2_prices, maturities)

    gv2p_prices = [gv2p_engine.P(gv2p_model, t) for t in maturities]
    gv2p_yields = gv2p_engine.spot_rate(gv2p_prices, maturities)
    return (
        cir2_yields,
        cir_yields,
        g2_yields,
        gv2p_yields,
        maturities,
        merton_yields,
        vas_yields,
    )


@app.cell
def _(merton_model):
    print(merton_model)
    return


@app.cell
def _(cir_model):
    print(cir_model)
    return


@app.cell
def _(vas_model):
    print(vas_model)
    return


@app.cell
def _(
    cir2_yields,
    cir_yields,
    g2_yields,
    gv2p_yields,
    market_data,
    maturities,
    merton_yields,
    plot_yield_curve,
    vas_yields,
):
    plot_yield_curve(maturities, merton_yields, vas_yields, cir_yields, g2_yields, cir2_yields, gv2p_yields, market_data)
    return


@app.cell
def _(
    cir2_yields,
    cir_yields,
    g2_yields,
    gv2p_yields,
    market_data,
    maturities,
    merton_yields,
    plot_yield_curve,
    vas_yields,
):
    plot_yield_curve(maturities, merton_yields, vas_yields, cir_yields, g2_yields, cir2_yields, gv2p_yields, market_data, factors=2)
    return


@app.cell
def _(g2_model):
    print(g2_model)
    return


@app.cell
def _(cir2_model):
    print(cir2_model)
    return


@app.cell
def _(gv2p_model):
    print(gv2p_model)
    return


if __name__ == "__main__":
    app.run()
