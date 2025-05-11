

import marimo

__generated_with = "0.13.1"
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
    mo.md(
        r"""
        ## G2 Model
        ###Two Additive Factor Gaussian Model

        \begin{aligned}
        dx_t &= -ax_t dt + \sigma_x dW_x \\
        dy_t &= -by_t dt + \sigma_y dW_y \\
        r_t &= x_t + y_t + \phi\\
        \end{aligned}

        $$
        dW_x(t) dW_y(t) = \rho
        $$

        \begin{aligned}
        P(t, T) &= ...\\
        \end{aligned}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## CIR2 Model
        ### Two Facotor Cox Ingersol Ross Model
        $$
        \\begin{aligned}
        dx_t &= -\\kappa_x(\\theta_x- x_t) dt + \\sigma_x \\sqrt{x_t} dW_x \\\\
        dy_t &= -\\kappa_y(\\theta_y- y_t) dt + \\sigma_y \\sqrt{y_t} dW_y \\\\
        r_t &= x_t + y_t \\\\
        \\end{aligned}
        $$
        """
    )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## G2+ Model

        $$
        \\begin{aligned}
        dx_t &= -ax_t dt + \sigma_x dW_x \\\\
        dy_t &= -by_t dt + \sigma_y dW_y \\\\
        dz_t &= -k(z_t - x_t - y_t) \\\\
        r_t &= z_t + \phi\\
        \\end{aligned}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(context='notebook', style='whitegrid', palette='colorblind')
    return mo, np, plt


@app.cell(hide_code=True)
def _():
    from src.core.calibration import Calibrator
    from src.core.util import spot_rate

    from src.engines.closed_form import price_closed_form_cir, price_closed_form_g2, price_closed_form_vasicek, price_closed_form_cir2, price_closed_form_merton, price_closed_form_gv2p, price_closed_form_v2, price_closed_form_g2plus

    from src.engines.binomial_tree import price_binomial_vasicek, price_binomial_merton
    from src.engines.monte_carlo import price_monte_carlo_merton, price_monte_carlo_vasicek, plot_paths_vasicek
    from src.models.cir import CIR
    from src.models.g2 import G2
    from src.models.vasicek import Vasicek
    from src.models.cir2 import CIR2
    from src.models.merton import Merton
    from src.models.gv2p import GV2P
    from src.models.v2 import V2
    from src.models.g2_plus import G2plus
    from src.optim.least_squares import SciPyLeastSquares

    return (
        CIR,
        CIR2,
        Calibrator,
        G2,
        Merton,
        SciPyLeastSquares,
        Vasicek,
        plot_paths_vasicek,
        price_closed_form_cir,
        price_closed_form_cir2,
        price_closed_form_g2,
        price_closed_form_merton,
        price_closed_form_vasicek,
        spot_rate,
    )


@app.cell(hide_code=True)
def _(plt):
    def plot_yield_curve(maturities, merton_yields, vas_yields, cir_yields, g2_yields, cir2_yields, market_data, factors: int = 1):
        fig, ax = plt.subplots(figsize=(11, 6))
        if factors == 1:
            ax.plot(maturities, 100 * merton_yields)
            ax.plot(maturities, 100 * vas_yields)
            ax.plot(maturities, 100 * cir_yields)
        elif factors == 2:
            ax.plot(maturities, 100 * g2_yields)
            ax.plot(maturities, 100 * cir2_yields)
            # ax.plot(maturities, 100 * g2p_yields)
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
            ax.legend(["G2", "CIR2"], edgecolor='none', borderpad=2)
            ax.set_title( "Two Factor Short Rate Models", fontsize=20, pad=15)
        fig.tight_layout()
        plt.show()

    return (plot_yield_curve,)


@app.cell
def _(
    CIR,
    CIR2,
    Calibrator,
    G2,
    Merton,
    SciPyLeastSquares,
    Vasicek,
    np,
    price_closed_form_cir,
    price_closed_form_cir2,
    price_closed_form_g2,
    price_closed_form_merton,
    price_closed_form_vasicek,
    spot_rate,
):
    market_data = [(0.25, 0.0433), (2.0, 0.0383), (5.0, 0.0392), (10.0, 0.0433), (30.0, 0.0479)]
    # market_data = [(1.0, 0.0175), (5.0, 0.0155), (10.0, 0.0168), (30.0, 0.0212)]
    # market_data = [(0.25, 0.04), (2.0, 0.0382), (10.0, 0.0411), (30.0, 0.0436), (100.0, 0.0430)]

    maturities = np.linspace(0.25, 50, 200)
    optimizer = SciPyLeastSquares()

    merton_model, merton_engine = Calibrator.calibrate_closed_form(model_cls=Merton,market_data=market_data, r0=0.04, mu=0.002, sigma=0.002)
    merton_prices = [price_closed_form_merton(merton_model, t) for t in maturities]
    merton_yields = spot_rate(merton_prices, maturities)

    vas_model, vas_engine = Calibrator.calibrate_closed_form(model_cls=Vasicek, market_data=market_data, r0=0.0, kappa=0.01, theta=0.01, sigma=0.001)
    vas_prices = [price_closed_form_vasicek(vas_model, t) for t in maturities]
    vas_yields = spot_rate(vas_prices, maturities)

    cir_model, cir_engine = Calibrator.calibrate_closed_form(model_cls=CIR, market_data=market_data, r0=0.0, kappa=0.01, theta=0.00, sigma=0.02)
    cir_prices = [price_closed_form_cir(cir_model, t) for t in maturities]
    cir_yields = spot_rate(cir_prices, maturities)

    g2_model, g2_engine = Calibrator.calibrate_closed_form(model_cls=G2, market_data=market_data, x0=0.0, y0=0.00, a=0.02, b=0.01, rho=-0.5, phi=0.03, sigma_x=0.05, sigma_y=0.01)
    g2_prices = [price_closed_form_g2(g2_model, t) for t in maturities]
    g2_yields = spot_rate(g2_prices, maturities)

    cir2_model, cir2_engine = Calibrator.calibrate_closed_form(model_cls=CIR2, market_data=market_data, r0_1=0.01, r0_2=0.01,  kappa1=0.02, kappa2=0.01, theta1=0.01, theta2=0.02, sigma_x=0.05, sigma_y=0.05)
    cir2_prices = [price_closed_form_cir2(cir2_model, t) for t in maturities]
    cir2_yields = spot_rate(cir2_prices, maturities)
    return (
        cir2_model,
        cir2_yields,
        cir_model,
        cir_yields,
        g2_model,
        g2_yields,
        market_data,
        maturities,
        merton_model,
        merton_yields,
        vas_model,
        vas_yields,
    )


@app.cell
def _(cir_model, merton_model, vas_model):
    print(merton_model)
    print(vas_model)
    print(cir_model)
    return


@app.cell
def _(
    cir2_yields,
    cir_yields,
    g2_yields,
    market_data,
    maturities,
    merton_yields,
    plot_yield_curve,
    vas_yields,
):
    plot_yield_curve(maturities, merton_yields, vas_yields, cir_yields, g2_yields, cir2_yields, market_data)
    return


@app.cell
def _(
    cir2_yields,
    cir_yields,
    g2_yields,
    market_data,
    maturities,
    merton_yields,
    plot_yield_curve,
    vas_yields,
):
    plot_yield_curve(maturities, merton_yields, vas_yields, cir_yields, g2_yields, cir2_yields, market_data, factors=2)
    return


@app.cell
def _(cir2_model, g2_model):
    print(g2_model)
    print(cir2_model)
    return


@app.cell
def _(Vasicek, plot_paths_vasicek):
    mc0_model = Vasicek(r0=0.0389, kappa=0.134, theta=0.0439, sigma=0.001)
    print(mc0_model)
    plot_paths_vasicek(mc0_model, 50, 30, 0.25, n=50)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
