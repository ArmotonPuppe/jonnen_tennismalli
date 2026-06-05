"""
Elo-painon optimointi slope-ongelman korjaamiseksi.

Laskee alustakohtaiset Elo-luvut historiasta ja testaa
eri Elo-painoja — löytää painon joka korjaa slope 0.49 → 1.0.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid

from backtest_pipeline import (
    download_atp_data, laske_spw_per_ottelu,
    laske_rolling_pre_match, rakenna_piirteet
)
from clv_backtest import (
    download_odds_data, yhdista_ottelut, laske_clv, MATCHUP_K
)

# ─────────────────────────────────────────────
# ELO-LASKENTA
# ─────────────────────────────────────────────

ELO_K       = 32
ELO_START   = 1500
SURFACES    = ["Hard", "Clay", "Grass"]

def laske_elo_historia(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Laskee alustakohtaiset Elo-luvut jokaiselle pelaajalle
    jokaisen ottelun JÄLKEEN (pre-match = arvo ennen ottelua).
    """
    df = raw_df.copy()
    df = df[df["surface"].isin(SURFACES)].sort_values("date")
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)

    elo = {}  # {pelaaja: {surface: elo}}

    def get_elo(player, surface):
        return elo.get(player, {}).get(surface, ELO_START)

    def set_elo(player, surface, value):
        if player not in elo:
            elo[player] = {}
        elo[player][surface] = value

    records = []
    for _, row in df.iterrows():
        w = row["winner_name"]
        l = row["loser_name"]
        s = row["surface"]
        d = row["date"]

        elo_w = get_elo(w, s)
        elo_l = get_elo(l, s)

        records.append({
            "date":    d,
            "surface": s,
            "winner":  w,
            "loser":   l,
            "pre_elo_w": elo_w,
            "pre_elo_l": elo_l,
        })

        # Päivitä Elo ottelun jälkeen
        exp_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        new_elo_w = elo_w + ELO_K * (1 - exp_w)
        new_elo_l = elo_l + ELO_K * (0 - (1 - exp_w))
        set_elo(w, s, new_elo_w)
        set_elo(l, s, new_elo_l)

    return pd.DataFrame(records)


def elo_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


# ─────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────

def testaa_elo_paino(df_clv: pd.DataFrame, elo_df: pd.DataFrame,
                     painot: list) -> pd.DataFrame:
    """
    Testaa eri Elo-painoja ja mittaa slope ja MAE markkinaan nähden.
    df_clv: yhdistetty CLV-data (sisältää model_prob, fair_w, winner, loser, date, surface)
    elo_df: pre-match Elo-historia
    """
    # Yhdistä Elo CLV-dataan
    elo_idx = elo_df.set_index(["winner", "loser", "surface", "date"])

    results = []
    for paino in painot:
        probs = []
        markets = []

        for _, row in df_clv.iterrows():
            try:
                e = elo_idx.loc[(row["winner"], row["loser"],
                                 row["surface"], row["date"])]
                if isinstance(e, pd.DataFrame):
                    e = e.iloc[0]
                elo_w = e["pre_elo_w"]
                elo_l = e["pre_elo_l"]
            except KeyError:
                continue

            elo_p  = elo_prob(elo_w, elo_l)
            spw_p  = row["model_prob"]   # pelkkä SPW/RPW

            # Sekoita logit-avaruudessa
            def logit(p):
                p = np.clip(p, 1e-6, 1 - 1e-6)
                return np.log(p / (1 - p))

            combined_logit = (1 - paino) * logit(spw_p) + paino * logit(elo_p)
            combined_prob  = expit(combined_logit)

            probs.append(combined_prob)
            markets.append(row["fair_w"])

        probs   = np.array(probs)
        markets = np.array(markets)

        slope, _ = np.polyfit(markets, probs, 1)
        mae      = np.mean(np.abs(probs - markets))
        clv_avg  = np.mean(probs - markets)
        corr     = np.corrcoef(probs, markets)[0, 1]

        results.append({
            "elo_paino": paino,
            "slope":     round(slope, 3),
            "mae_pp":    round(mae * 100, 2),
            "clv_avg":   round(clv_avg * 100, 2),
            "korrelaatio": round(corr, 3),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# PÄÄOHJELMA
# ─────────────────────────────────────────────

def main():
    print("Ladataan data...")
    raw     = download_atp_data()
    match   = laske_spw_per_ottelu(raw)
    rolling = laske_rolling_pre_match(match)
    feat    = rakenna_piirteet(match, rolling)
    odds    = download_odds_data()
    yht     = yhdista_ottelut(feat, odds)
    df_clv  = laske_clv(yht)

    print("Lasketaan Elo-historia...")
    elo_df  = laske_elo_historia(raw)

    painot = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(f"Testataan {len(painot)} Elo-painoa...\n")
    tulokset = testaa_elo_paino(df_clv, elo_df, painot)

    print(tulokset.to_string(index=False))

    # Paras paino (slope lähimpänä 1.0)
    paras = tulokset.iloc[(tulokset["slope"] - 1.0).abs().argsort().iloc[0]]
    print(f"\n✅ Paras Elo-paino (slope → 1.0): {paras['elo_paino']}")
    print(f"   Slope: {paras['slope']} | MAE: {paras['mae_pp']} pp | "
          f"CLV: {paras['clv_avg']:+.2f}%")

    # Graafi
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Elo-painon vaikutus malliin", fontsize=13)

    axes[0].plot(tulokset["elo_paino"], tulokset["slope"], "bo-")
    axes[0].axhline(1.0, color="red", ls="--", label="Tavoite (slope=1)")
    axes[0].set_xlabel("Elo-paino"); axes[0].set_ylabel("Slope")
    axes[0].set_title("Slope vs Elo-paino"); axes[0].legend()

    axes[1].plot(tulokset["elo_paino"], tulokset["mae_pp"], "ro-")
    axes[1].set_xlabel("Elo-paino"); axes[1].set_ylabel("MAE (pp)")
    axes[1].set_title("Tarkkuus vs Elo-paino")

    axes[2].plot(tulokset["elo_paino"], tulokset["clv_avg"], "go-")
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_xlabel("Elo-paino"); axes[2].set_ylabel("CLV keskiarvo (%)")
    axes[2].set_title("CLV vs Elo-paino")

    plt.tight_layout()
    plt.savefig("elo_optimointi.png", dpi=150)
    plt.show()

    print(f"\nPäivitä tennis_model.py:")
    print(f"  pohja_logit = {1-paras['elo_paino']:.2f} * matchup_logit + "
          f"{paras['elo_paino']:.2f} * elo_logit")


if __name__ == "__main__":
    main()
