

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
    from src.engines.closed_form import ClosedFormCIR, ClosedFormG2, ClosedFormVasicek
    from src.models.cir import CIR, CIRParams
    from src.models.g2 import G2, G2Params
    from src.models.vasicek import Vasicek, VasicekParams
    from src.optim.least_squares import SciPyLeastSquares

    return Calibrator, ClosedFormG2, G2, G2Params, SciPyLeastSquares


@app.cell
def _(plt):
    def plot_yield_curve(maturities, yields, market_data):
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(maturities, 100 * yields)
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
        # ax.legend(["Merton", "Vasicek", "CIR"], edgecolor='none', borderpad=2)
        ax.set_title( "Short Rate Models", fontsize=20, pad=15)
        fig.tight_layout()
        plt.show()
    
    return (plot_yield_curve,)


@app.cell
def _(Calibrator, ClosedFormG2, G2, G2Params, SciPyLeastSquares, np):
    market_data = [(1.0, 0.0175), (5.0, 0.0155), (10.0, 0.0168), (30.0, 0.0212)]

    params0 = G2Params(r0=0.04, a=0.01, b=0.01, rho=0.0, phi=0, sigma_x=0.005, sigma_y=0.005)
    model = G2(params0)
    engine = ClosedFormG2()
    optimizer = SciPyLeastSquares()

    bounds = [(-np.inf, np.inf),  # r0: no bounds
              (-np.inf, np.inf),  # a
              (-np.inf, np.inf),  # b
              (-1, 1),            # rho: correlation between -1 and 1
              (-np.inf, np.inf)]  # phi

    optimizer = SciPyLeastSquares(bounds=bounds)

    calib = Calibrator(model, engine, optimizer)
    calib.calibrate(market_data)

    print("Fitted:", calib.model.params())
    return engine, market_data, model


@app.cell
def _(model):
    print(model.params().r0)
    print(model.params().a)
    print(model.params().b)
    print(model.params().rho)
    print(model.params().phi)
    return


@app.cell
def _(engine, model, np):
    maturities = np.linspace(0.25, 30, 120)
    prices = [engine.P(model, t) for t in maturities]
    yields = engine.spot_rate(prices, maturities)
    return maturities, yields


@app.cell
def _(market_data, maturities, plot_yield_curve, yields):
    plot_yield_curve(maturities, yields, market_data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
