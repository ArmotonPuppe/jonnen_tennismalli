"""
Malli vs Markkina — tarkkuusanalyysi
Kuinka lähelle mallin todennäköisyydet osuvat markkinaan?
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtest_pipeline import (
    download_atp_data, laske_spw_per_ottelu,
    laske_rolling_pre_match, rakenna_piirteet, aja_regressio,
    laske_elo_historia
)
from clv_backtest import (
    download_odds_data, yhdista_ottelut, laske_clv, MATCHUP_K
)

def main():
    print("Ladataan data...")
    raw     = download_atp_data()
    match   = laske_spw_per_ottelu(raw)
    rolling = laske_rolling_pre_match(match)
    feat    = rakenna_piirteet(match, rolling)
    reg     = aja_regressio(feat)
    coefs   = reg["coefs"]
    print("Lasketaan Elo-historia...")
    elo_df  = laske_elo_historia(raw)
    odds    = download_odds_data()
    yht     = yhdista_ottelut(feat, odds)
    df      = laske_clv(yht, coefs=coefs, elo_df=elo_df)

    mp  = df["model_prob"].values
    mkt = df["fair_w"].values
    clv = df["clv"].values

    mae  = np.mean(np.abs(mp - mkt))
    bias = np.mean(mp - mkt)
    corr = np.corrcoef(mp, mkt)[0, 1]

    print(f"\n{'─'*45}")
    print(f"  Otteluita analysoitu     : {len(df):,}")
    print(f"  Korrelaatio (malli↔mkt) : {corr:.3f}")
    print(f"  Keskimääräinen abs. ero  : {mae*100:.2f} pp")
    print(f"  Systemaattinen harha     : {bias*100:+.2f} pp")
    print(f"    (+ = malli yliarvioi voittajia, − = aliarvioi)")
    print(f"{'─'*45}")

    # Jakauma CLV-eroista
    percentiles = [10, 25, 50, 75, 90]
    print("\n  CLV-jakauma (malli − markkina):")
    for p in percentiles:
        print(f"    P{p:2d}: {np.percentile(clv*100, p):+.1f}%")

    # ── GRAAFIT ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Malli vs Markkina — tarkkuusanalyysi", fontsize=13)

    # 1. Scatter: malli vs markkina
    ax = axes[0]
    ax.scatter(mkt * 100, mp * 100, alpha=0.15, s=8, color="steelblue")
    ax.plot([10, 90], [10, 90], "k--", lw=1.5, label="Täydellinen")
    lf = np.polyfit(mkt, mp, 1)
    xs = np.linspace(mkt.min(), mkt.max(), 100)
    ax.plot(xs * 100, np.polyval(lf, xs) * 100, "r-", lw=1.5,
            label=f"Sovite (slope={lf[0]:.2f})")
    ax.set_xlabel("Markkinatodennäköisyys (%)")
    ax.set_ylabel("Mallin todennäköisyys (%)")
    ax.set_title(f"Scatter (r={corr:.3f})")
    ax.legend(fontsize=8)

    # 2. Harhajakauma (model_prob - market_prob)
    ax = axes[1]
    ax.hist(clv * 100, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0,    color="red",   lw=2, label="Nollaraja")
    ax.axvline(bias * 100, color="orange", lw=2, ls="--",
               label=f"Keskiarvo ({bias*100:+.2f}%)")
    ax.set_xlabel("Malli − Markkina (pp)")
    ax.set_ylabel("Otteluita")
    ax.set_title(f"Harhajakauma (MAE={mae*100:.2f} pp)")
    ax.legend(fontsize=8)

    # 3. Mallin tarkkuus todennäköisyysvyöhykkeittäin
    ax = axes[2]
    bins = np.linspace(0.2, 0.85, 8)
    centers, maes, counts = [], [], []
    for i in range(len(bins) - 1):
        mask = (mkt >= bins[i]) & (mkt < bins[i+1])
        if mask.sum() > 20:
            centers.append((bins[i] + bins[i+1]) / 2 * 100)
            maes.append(np.mean(np.abs(clv[mask])) * 100)
            counts.append(mask.sum())

    bars = ax.bar(centers, maes, width=8, color="steelblue", edgecolor="white")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Markkinatodennäköisyys (%)")
    ax.set_ylabel("Keskimääräinen abs. ero (pp)")
    ax.set_title("Tarkkuus vyöhykkeittäin")

    plt.tight_layout()
    plt.savefig("malli_vs_markkina.png", dpi=150)
    plt.show()
    print("\nGraafi tallennettu: malli_vs_markkina.png")

if __name__ == "__main__":
    main()
