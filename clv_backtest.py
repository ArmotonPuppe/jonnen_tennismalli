"""
CLV Backtest — Closing Line Value
Yhdistää backtest_pipeline.py:n malliennusteet
tennis-data.co.uk:n Pinnacle-sulkeviin kertoimiin.

Mittaa: onko mallilla edgeä markkinaan nähden?
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Tuo funktiot olemassa olevasta pipelinesta
from backtest_pipeline import (
    BAYES_K,
    download_atp_data,
    laske_spw_per_ottelu,
    laske_rolling_pre_match,
    rakenna_piirteet,
    bayes_adj,
    aja_regressio,
    laske_elo_historia,
    elo_prob,
)

warnings.filterwarnings("ignore")

ODDS_DIR  = Path("odds_data")
YEARS     = range(2015, 2025)
MATCHUP_K = 26.0    # Regressiosta saatu kerroin
MIN_CLV   = 0.03    # Minimiedge vedon tekemiseen (3%)
ELO_PAINO = 0.40    # vastaa tennis_model.py:n 0.6/0.4 sekoitusta


# ─────────────────────────────────────────────
# 1. ODDS-DATA (tennis-data.co.uk)
# ─────────────────────────────────────────────

def download_odds_data() -> pd.DataFrame:
    """
    Lukee ATP-kertoimet odds_data-kansiosta.

    Lataa tiedostot manuaalisesti osoitteesta:
      http://www.tennis-data.co.uk/alldata.php
    Nimeä ne muodossa: atp_2015.xlsx, atp_2016.xlsx, ...
    ja siirrä odds_data-kansioon.
    """
    ODDS_DIR.mkdir(exist_ok=True)
    dfs = []

    # Hyväksytään sekä .xlsx että .csv
    for year in YEARS:
        for suffix in [".xlsx", ".csv"]:
            path = ODDS_DIR / f"atp_{year}{suffix}"
            if path.exists():
                try:
                    df = (pd.read_excel(path) if suffix == ".xlsx"
                          else pd.read_csv(path, encoding="latin-1"))
                    dfs.append(df)
                    print(f"  Luettu: {path.name} ({len(df)} ottelua)")
                except Exception as e:
                    print(f"  Lukuvirhe {path.name}: {e}")
                break  # löytyi tämä vuosi, ei tarkisteta toista formaattia

    if not dfs:
        raise RuntimeError(
            "\n\nOdds-tiedostoja ei löydy!\n"
            "Lataa ATP-vuodet osoitteesta:\n"
            "  http://www.tennis-data.co.uk/alldata.php\n"
            f"ja tallenna ne kansioon: {ODDS_DIR.resolve()}\n"
            "Nimeä tiedostot: atp_2015.xlsx, atp_2016.xlsx, ..."
        )

    odds = pd.concat(dfs, ignore_index=True)

    # Normalisoi sarakkeet
    odds.columns = [c.strip() for c in odds.columns]

    # Päivämäärä
    if "Date" in odds.columns:
        odds["date"] = pd.to_datetime(odds["Date"], dayfirst=True, errors="coerce")
    odds = odds.dropna(subset=["date"])

    # Pinnacle-kertoimet (PSW/PSL) tai vaihtoehto (B365W/B365L)
    if "B365W" in odds.columns and "B365L" in odds.columns:
        odds = odds.rename(columns={"B365W": "odds_w", "B365L": "odds_l"})
        print("  Käytetään Bet365-kertoimia (B365W/B365L)")
    elif "PSW" in odds.columns and "PSL" in odds.columns:
        odds = odds.rename(columns={"PSW": "odds_w", "PSL": "odds_l"})
        print("  Käytetään Pinnacle-kertoimia (PSW/PSL)")
    else:
        raise RuntimeError(f"Kertoimia ei löydy. Sarakkeet: {list(odds.columns)}")

    odds = odds.dropna(subset=["odds_w", "odds_l"])
    odds = odds[(odds["odds_w"] > 1.0) & (odds["odds_l"] > 1.0)]

    return odds[["date", "Winner", "Loser", "odds_w", "odds_l"]]


# ─────────────────────────────────────────────
# 2. NIMIVERTAILU (Tennis Abstract ↔ tennis-data)
# ─────────────────────────────────────────────

def normalisoi_nimi(nimi: str) -> str:
    """
    Normalisoi nimen vertailua varten.
    Tennis Abstract:    'Taylor Fritz'   → 'fritz'   (viimeinen sana)
    tennis-data.co.uk:  'Fritz T.'       → 'fritz'   (kaikki paitsi viimeinen sana)
    Tunnistaa formaatin: jos viimeinen sana on yksikirjaiminen initial → td-formaatti.
    """
    if not isinstance(nimi, str):
        return ""
    osat = nimi.strip().split()
    if not osat:
        return ""
    # tennis-data formaatti: "Sukunimi E." tai "Sukunimi van den E."
    if len(osat) >= 2 and len(osat[-1].rstrip(".")) <= 2:
        return " ".join(osat[:-1]).lower().replace("'", "").replace("-", "")
    # Tennis Abstract formaatti: "Etunimi Sukunimi"
    return osat[-1].lower().replace("'", "").replace("-", "")


def yhdista_ottelut(feat: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()
    odds = odds.copy()

    feat["date_d"] = feat["date"].dt.date
    odds["date_d"] = odds["date"].dt.date

    feat["w_suku"] = feat["winner"].apply(normalisoi_nimi)
    feat["l_suku"] = feat["loser"].apply(normalisoi_nimi)
    odds["w_suku"] = odds["Winner"].apply(normalisoi_nimi)
    odds["l_suku"] = odds["Loser"].apply(normalisoi_nimi)

    yhdistetty = feat.merge(
        odds[["date_d", "w_suku", "l_suku", "odds_w", "odds_l"]],
        on=["date_d", "w_suku", "l_suku"],
        how="inner"
    )

    print(f"  Yhdistetty: {len(yhdistetty):,} ottelua "
          f"({len(yhdistetty)/len(feat)*100:.0f}% malliotteluista)")
    return yhdistetty


# ─────────────────────────────────────────────
# 3. MALLIN TODENNÄKÖISYYS
# ─────────────────────────────────────────────

SLOPE_KORJAUS = 1 / 0.54  # mitattu slope malli_vs_markkina.py:stä (log5-mallilla)


def logit_prob(row: pd.Series, coefs: dict, elo_idx=None, slope_korjaus: bool = False) -> float:
    """Laskee mallin todennäköisyyden: 60% SPW/RPW-regressio + 40% Elo."""
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    matchup_logit = 0.0
    for feat, k in coefs.items():
        if feat in row.index and not np.isnan(row[feat]):
            matchup_logit += row[feat] * k

    if elo_idx is not None:
        try:
            e = elo_idx.loc[(row["winner"], row["loser"], row["surface"], row["date"])]
            if isinstance(e, pd.DataFrame):
                e = e.iloc[0]
            elo_p     = elo_prob(e["pre_elo_w"], e["pre_elo_l"])
            elo_logit = _logit(elo_p)
            combined  = (1 - ELO_PAINO) * matchup_logit + ELO_PAINO * elo_logit
        except (KeyError, TypeError):
            combined = matchup_logit
    else:
        combined = matchup_logit

    if slope_korjaus:
        combined = max(-1.5, min(1.5, combined)) * SLOPE_KORJAUS

    return 1 / (1 + np.exp(-combined))


# ─────────────────────────────────────────────
# 4. CLV-LASKENTA
# ─────────────────────────────────────────────

def laske_clv(df: pd.DataFrame, coefs: dict = None, elo_df: pd.DataFrame = None,
              slope_korjaus: bool = False) -> pd.DataFrame:
    """
    Laskee CLV jokaiselle ottelulle.

    fair_prob = markkinatodennäköisyys ilman marginaalia (vig poistettu)
    clv       = mallin ennuste - markkinan fair prob
    slope_korjaus: venyttää logitit ×(1/slope) ennen todennäköisyyttä
    """
    if coefs is None:
        coefs = {"dom_diff": MATCHUP_K}
    df = df.copy()

    # Rakenna Elo-hakemisto jos data annettu
    elo_idx = None
    if elo_df is not None:
        elo_idx = elo_df.set_index(["winner", "loser", "surface", "date"])

    # Mallin ennuste voittajalle: SPW/RPW + Elo 40% (+ optionaalinen slope-korjaus)
    df["model_prob"] = df.apply(
        lambda r: logit_prob(r, coefs, elo_idx, slope_korjaus), axis=1
    )

    # Markkinan fair prob (vig poistettu)
    df["raw_w"]    = 1 / df["odds_w"]
    df["raw_l"]    = 1 / df["odds_l"]
    df["vig"]      = df["raw_w"] + df["raw_l"]
    df["fair_w"]   = df["raw_w"] / df["vig"]

    # CLV: positiivinen = mallilla edge
    df["clv"] = df["model_prob"] - df["fair_w"]

    return df


# ─────────────────────────────────────────────
# 5. ROI-SIMULAATIO
# ─────────────────────────────────────────────

def simuloi_roi(df: pd.DataFrame, min_clv: float = MIN_CLV) -> dict:
    """
    Simuloi realistinen vedonlyönti:
    - Lasketaan CLV molemmille pelaajille
    - Vedetään sille jolla on positiivinen CLV yli kynnyksen
    - dom_diff > 0: malli suosii voittajaa → veto voittajalle → WIN
    - dom_diff < 0: malli suosii häviäjää → veto häviäjälle → LOSE
    """
    df = df.copy()

    # CLV häviäjälle (mallin näkökulma: 1 - model_prob vs 1 - fair_w)
    df["model_prob_l"] = 1 - df["model_prob"]
    df["fair_l"]       = 1 - df["fair_w"]
    df["clv_l"]        = df["model_prob_l"] - df["fair_l"]

    # Veto voittajalle kun malli suosii voittajaa
    bet_winner = df[df["clv"] >= min_clv].copy()
    bet_winner["tulos"]  = 1                        # voittaja voittaa aina
    bet_winner["tuotto"] = bet_winner["odds_w"] - 1

    # Veto häviäjälle kun malli suosii häviäjää
    bet_loser = df[df["clv_l"] >= min_clv].copy()
    bet_loser["tulos"]  = 0                         # häviäjä häviää aina
    bet_loser["tuotto"] = -1.0

    vedot = (pd.concat([bet_winner, bet_loser], ignore_index=True)
               .sort_values("date")
               .reset_index(drop=True))

    if len(vedot) == 0:
        return {"vetoja": 0, "roi_pct": 0.0, "clv_keskiarvo": 0.0,
                "clv_positiiviset_pct": 0.0}, pd.DataFrame()

    roi = vedot["tuotto"].sum() / len(vedot)

    stats = {
        "vetoja":               len(vedot),
        "roi_pct":              round(roi * 100, 2),
        "clv_keskiarvo":        round(df["clv"].mean() * 100, 2),
        "clv_positiiviset_pct": round((df["clv"] > 0).mean() * 100, 1),
        "oikein_pct":           round(vedot["tulos"].mean() * 100, 1),
    }
    return stats, vedot


# ─────────────────────────────────────────────
# 6. VISUALISOINTI
# ─────────────────────────────────────────────

def piirrä_tulokset(df: pd.DataFrame, vedot: pd.DataFrame,
                    df_k: pd.DataFrame = None, vedot_k: pd.DataFrame = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("CLV Backtest — Mallin edge markkinaan nähden", fontsize=13)

    # 1. CLV-jakauma (raaka vs slope-korjattu)
    ax = axes[0]
    ax.hist(df["clv"] * 100, bins=40, color="steelblue", edgecolor="white",
            alpha=0.6, label="Raaka")
    if df_k is not None:
        ax.hist(df_k["clv"] * 100, bins=40, color="orange", edgecolor="white",
                alpha=0.5, label="Slope-korjattu")
    ax.axvline(0, color="red", lw=2, label="Nollaraja")
    ax.axvline(MIN_CLV * 100, color="green", lw=2, ls="--",
               label=f"Kynnys ({MIN_CLV*100:.0f}%)")
    ax.set_xlabel("CLV (%)")
    ax.set_ylabel("Otteluita")
    ax.set_title("CLV-jakauma")
    ax.legend(fontsize=8)

    # 2. Kumulatiivinen ROI — raaka vs slope-korjattu
    ax = axes[1]
    for v, label, color in [(vedot, "Raaka", "red"), (vedot_k, "Slope-korj.", "orange")]:
        if v is not None and len(v) > 0:
            v = v.reset_index(drop=True)
            v["kum_tuotto"] = v["tuotto"].cumsum()
            final = v["kum_tuotto"].iloc[-1]
            c = "green" if final > 0 else color
            ax.plot(v.index + 1, v["kum_tuotto"], color=c, lw=1.5,
                    label=f"{label} (n={len(v)}, {final:+.0f})")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Vetoa")
    ax.set_ylabel("Kumulatiivinen tuotto (yksikköä)")
    ax.set_title("Kumulatiivinen ROI")
    ax.legend(fontsize=8)

    # 3. CLV alustattain
    ax = axes[2]
    surfaces = df.groupby("surface")["clv"].mean() * 100
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in surfaces.values]
    ax.bar(surfaces.index, surfaces.values, color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Keskimääräinen CLV (%)")
    ax.set_title("CLV alustattain (raaka)")

    plt.tight_layout()
    plt.savefig("clv_tulokset.png", dpi=150)
    plt.show()
    print("  Graafi tallennettu: clv_tulokset.png")


# ─────────────────────────────────────────────
# PÄÄOHJELMA
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  CLV Backtest")
    print("=" * 50)

    # --- Mallidata (backtest pipeline) ---
    print("\n[1/5] Ladataan ATP-data...")
    raw = download_atp_data()
    print(f"      {len(raw):,} ottelua")

    print("\n[2/5] Lasketaan per-ottelu SPW/RPW...")
    match_df = laske_spw_per_ottelu(raw)

    print("\n[3/5] Rolling pre-match tilastot...")
    rolling = laske_rolling_pre_match(match_df)
    feat = rakenna_piirteet(match_df, rolling)
    print(f"      {len(feat):,} ottelua mallissa")

    # Aja regressio saadaksesi kertoimet kaikille piirteille
    print("      Ajetaan regressio kertoimille...")
    reg = aja_regressio(feat)
    coefs = reg["coefs"]
    print(f"      Kertoimet: {coefs}")

    # Laske Elo-historia (40% paino kuten tennis_model.py:ssä)
    print("      Lasketaan Elo-historia...")
    elo_df = laske_elo_historia(raw)
    print(f"      {len(elo_df):,} Elo-tietuetta")

    # --- Odds-data ---
    print("\n[4/5] Ladataan Pinnacle-kertoimet...")
    odds = download_odds_data()
    print(f"      {len(odds):,} ottelua kertoimilla")

    # --- Yhdistä ja laske CLV ---
    print("\n[5/5] Yhdistetään ja lasketaan CLV...")
    yhdistetty = yhdista_ottelut(feat, odds)

    # Versio 1: raaka malli
    tulokset      = laske_clv(yhdistetty, coefs=coefs, elo_df=elo_df, slope_korjaus=False)
    roi, vedot_df = simuloi_roi(tulokset)

    # Versio 2: slope-korjattu
    tulokset_k      = laske_clv(yhdistetty, coefs=coefs, elo_df=elo_df, slope_korjaus=True)
    roi_k, vedot_k  = simuloi_roi(tulokset_k)

    # --- Tulokset ---
    if len(tulokset) == 0:
        print("\n  ⚠️  Ei yhdistettyjä otteluita — nimiformaatin ongelma")
        return

    print(f"\n{'─'*50}")
    print(f"  Otteluita analysoitu : {len(tulokset):,}")
    print(f"\n  {'Mittari':<25} {'Raaka':>10} {'Slope-korj':>12}")
    print(f"  {'─'*47}")
    print(f"  {'CLV keskiarvo':<25} {roi['clv_keskiarvo']:>+9.2f}% {roi_k['clv_keskiarvo']:>+11.2f}%")
    print(f"  {'CLV > 0 (%)':<25} {roi['clv_positiiviset_pct']:>10.1f}% {roi_k['clv_positiiviset_pct']:>11.1f}%")
    print(f"  {'Vetoja':<25} {roi['vetoja']:>10} {roi_k['vetoja']:>12}")
    print(f"  {'ROI':<25} {roi['roi_pct']:>+9.2f}% {roi_k['roi_pct']:>+11.2f}%")
    print(f"{'─'*50}")

    if roi["clv_keskiarvo"] > 0:
        print("\n  ✅ Malli löytää positiivista CLV:tä — edgeä voi olla")
        print("     Vahvista lisää vedoilla ennen oikeaa panostamista.")
    else:
        print("\n  ❌ Negatiivinen CLV — malli häviää markkinalle")
        print("     Parametreja tai dataa täytyy parantaa ennen panostamista.")

    json_out = {**roi, "otteluita_analysoitu": len(tulokset)}
    with open("clv_tulokset.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=4, ensure_ascii=False)
    print("\n  Tulokset tallennettu: clv_tulokset.json")

    piirrä_tulokset(tulokset, vedot_df, df_k=tulokset_k, vedot_k=vedot_k)


if __name__ == "__main__":
    main()
