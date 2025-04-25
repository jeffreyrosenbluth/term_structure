

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    return np, plt


@app.cell
def _():
    from src.core.calibration import Calibrator
    from src.engines.closed_form import ClosedFormCIR, ClosedFormG2, ClosedFormVasicek, ClosedFormCIR2
    from src.models.cir import CIR, CIRParams
    from src.models.g2 import G2, G2Params
    from src.models.vasicek import Vasicek, VasicekParams
    from src.models.cir2 import CIR2, CIR2Params
    from src.optim.least_squares import SciPyLeastSquares

    return (
        CIR,
        CIR2,
        CIR2Params,
        CIRParams,
        Calibrator,
        ClosedFormCIR,
        ClosedFormCIR2,
        ClosedFormG2,
        ClosedFormVasicek,
        G2,
        G2Params,
        SciPyLeastSquares,
        Vasicek,
        VasicekParams,
    )


@app.cell
def _(plt):
    def plot_yield_curve(maturities, vas_yields, cir_yields, g2_yields, cir2_yields, market_data):
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(maturities, 100 * vas_yields)
        ax.plot(maturities, 100 * cir_yields)
        ax.plot(maturities, 100 * g2_yields)
        ax.plot(maturities, 100 * cir2_yields)
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
        ax.legend(["Vasicek", "CIR", "G2", "CIR2"], edgecolor='none', borderpad=2)
        ax.set_title( "Short Rate Models", fontsize=20, pad=15)
        fig.tight_layout()
        plt.show()

    return (plot_yield_curve,)


@app.cell
def _(
    CIR,
    CIR2,
    CIR2Params,
    CIRParams,
    Calibrator,
    ClosedFormCIR,
    ClosedFormCIR2,
    ClosedFormG2,
    ClosedFormVasicek,
    G2,
    G2Params,
    SciPyLeastSquares,
    Vasicek,
    VasicekParams,
):
    market_data = [(1.0, 0.0175), (5.0, 0.0155), (10.0, 0.0168), (30.0, 0.0212)]

    optimizer = SciPyLeastSquares()

    g2_params0 = G2Params(a=0.001, b=0.001, rho=0.0, phi=0, sigma_x=0.01, sigma_y=0.01)
    g2_model = G2(g2_params0)
    g2_engine = ClosedFormG2()
    g2_calib = Calibrator(g2_model, g2_engine, optimizer)
    g2_calib.calibrate(market_data)

    cir_params0 = CIRParams(r0=0.0, kappa=0.0, theta=0.00, sigma=0.01)
    cir_model = CIR(cir_params0)
    cir_engine = ClosedFormCIR()
    cir_calib = Calibrator(cir_model, cir_engine, optimizer)
    cir_calib.calibrate(market_data)

    vas_params0 = VasicekParams(r0=0.01, kappa=0.01, theta=0.0, sigma=0.01)
    vas_model = Vasicek(vas_params0)
    vas_engine = ClosedFormVasicek()
    vas_calib = Calibrator(vas_model, vas_engine, optimizer)
    vas_calib.calibrate(market_data)

    cir2_params0 = CIR2Params(r0_1=0.01, r0_2=0.01, kappa1=0.02, kappa2=0.01, theta1=0.1, 
                              theta2=0.2, sigma1=0.01, sigma2=0.01)
    cir2_model = CIR2(cir2_params0)
    cir2_engine = ClosedFormCIR2()
    cir2_calib = Calibrator(cir2_model, cir2_engine, optimizer)
    cir2_calib.calibrate(market_data)
    return (
        cir2_engine,
        cir2_model,
        cir_engine,
        cir_model,
        g2_engine,
        g2_model,
        market_data,
        vas_engine,
        vas_model,
    )


@app.cell
def _(
    cir2_engine,
    cir2_model,
    cir_engine,
    cir_model,
    g2_engine,
    g2_model,
    np,
    vas_engine,
    vas_model,
):
    maturities = np.linspace(0.25, 30, 360)

    g2_prices = [g2_engine.P(g2_model, t) for t in maturities]
    g2_yields = g2_engine.spot_rate(g2_prices, maturities)

    cir_prices = [cir_engine.P(cir_model, t) for t in maturities]
    cir_yields = cir_engine.spot_rate(cir_prices, maturities)

    vas_prices = [vas_engine.P(vas_model, t) for t in maturities]
    vas_yields = vas_engine.spot_rate(vas_prices, maturities)

    cir2_prices = [cir2_engine.P(cir2_model, t) for t in maturities]
    cir2_yields = cir2_engine.spot_rate(cir2_prices, maturities)
    return cir2_yields, cir_yields, g2_yields, maturities, vas_yields


@app.cell
def _(
    cir2_yields,
    cir_yields,
    g2_yields,
    market_data,
    maturities,
    plot_yield_curve,
    vas_yields,
):
    plot_yield_curve(maturities, vas_yields, cir_yields, g2_yields, cir2_yields, market_data)
    return


@app.cell
def _(g2_model):
    print(g2_model)
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
def _(cir2_model):
    print(cir2_model)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
