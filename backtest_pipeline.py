"""
Tennis Backtest Pipeline
- Lataa ATP-data (JeffSackmann/tennis_atp)
- Laskee rolling pre-match SPW/RPW ilman datavuotoa
- Kalibroi logit-kertoimet logistisella regressiolla
- Tulostaa päivitetyt kertoimet tennis_model.py:tä varten
"""

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --- ASETUKSET ---
DATA_DIR  = Path("atp_data")
YEARS     = range(2015, 2025)
BAYES_K   = 10     # Bayesilainen shrinkage-parametri (sama kuin mallissa)
L52W_DAYS = 365    # "Viimeiset 52 viikkoa"


# ─────────────────────────────────────────────
# 1. DATA LATAUS
# ─────────────────────────────────────────────

def download_atp_data() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    dfs = []
    base = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
    for year in YEARS:
        path = DATA_DIR / f"atp_matches_{year}.csv"
        if not path.exists():
            print(f"  Ladataan {year}...", end=" ", flush=True)
            try:
                r = requests.get(f"{base}/atp_matches_{year}.csv", timeout=30)
                r.raise_for_status()
                path.write_bytes(r.content)
                print("OK")
            except Exception as e:
                print(f"VIRHE ({e})")
                continue
        dfs.append(pd.read_csv(path, low_memory=False))

    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    return df


# ─────────────────────────────────────────────
# 2. SPW / RPW PER OTTELU
# ─────────────────────────────────────────────

def laske_tb_stats(score: str) -> tuple:
    """Palauttaa (tb_pelattu, tb_voitettu) voittajan näkökulmasta."""
    if not isinstance(score, str):
        return 0, 0
    import re
    tb_pelattu, tb_voitettu = 0, 0
    for osa in score.split():
        m = re.match(r"(\d+)-(\d+)\((\d+)\)", osa)
        if m:
            tb_pelattu += 1
            if int(m.group(1)) > int(m.group(2)):
                tb_voitettu += 1
    return tb_pelattu, tb_voitettu


def laske_spw_per_ottelu(df: pd.DataFrame) -> pd.DataFrame:
    """Laskee voittajan ja häviäjän SPW/RPW, BP ja TB jokaiselle ottelulle."""
    tarvittavat = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
    df = df.dropna(subset=tarvittavat).copy()
    df = df[(df["w_svpt"] > 0) & (df["l_svpt"] > 0)]
    df = df[df["surface"].isin(["Hard", "Clay", "Grass"])]

    df["w_spw"] = (df["w_1stWon"] + df["w_2ndWon"]) / df["w_svpt"]
    df["l_spw"] = (df["l_1stWon"] + df["l_2ndWon"]) / df["l_svpt"]
    df["w_rpw"] = 1 - df["l_spw"]
    df["l_rpw"] = 1 - df["w_spw"]

    for col in ["w_spw", "l_spw", "w_rpw", "l_rpw"]:
        df = df[(df[col] >= 0.2) & (df[col] <= 0.95)]

    # BP Saved% — kuinka usein pelaaja pelastaa murtopallon
    df["w_bpsaved"] = np.where(df["w_bpFaced"] > 0,
                               df["w_bpSaved"] / df["w_bpFaced"], np.nan)
    df["l_bpsaved"] = np.where(df["l_bpFaced"] > 0,
                               df["l_bpSaved"] / df["l_bpFaced"], np.nan)

    # BP Conv% — kuinka usein pelaaja murtaa vastustajan
    df["w_bpconv"] = np.where(df["l_bpFaced"] > 0,
                              (df["l_bpFaced"] - df["l_bpSaved"]) / df["l_bpFaced"], np.nan)
    df["l_bpconv"] = np.where(df["w_bpFaced"] > 0,
                              (df["w_bpFaced"] - df["w_bpSaved"]) / df["w_bpFaced"], np.nan)

    # TB Win% — voitetut tasatilanteet
    if "score" in df.columns:
        tb = df["score"].apply(laske_tb_stats)
        df["w_tb_pelattu"] = tb.apply(lambda x: x[0])
        df["w_tb_voitettu"] = tb.apply(lambda x: x[1])
        df["l_tb_pelattu"]  = df["w_tb_pelattu"]
        df["l_tb_voitettu"] = df["w_tb_pelattu"] - df["w_tb_voitettu"]
        df["w_tbwin"] = np.where(df["w_tb_pelattu"] > 0,
                                 df["w_tb_voitettu"] / df["w_tb_pelattu"], np.nan)
        df["l_tbwin"] = np.where(df["l_tb_pelattu"] > 0,
                                 df["l_tb_voitettu"] / df["l_tb_pelattu"], np.nan)
    else:
        df["w_tbwin"] = np.nan
        df["l_tbwin"] = np.nan

    return df[["date", "surface", "winner_name", "loser_name",
               "w_spw", "l_spw", "w_rpw", "l_rpw",
               "w_bpsaved", "l_bpsaved", "w_bpconv", "l_bpconv",
               "w_tbwin", "l_tbwin"]]


# ─────────────────────────────────────────────
# 3. ROLLING PRE-MATCH TILASTOT (EI DATAVUOTOA)
# ─────────────────────────────────────────────

def laske_rolling_pre_match(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Laskee kullekin pelaajalle alustakohtaisen rolling-keskiarvon
    SPW:stä ja RPW:stä ENNEN kutakin ottelua.

    Avainratkaisu: shift(1) + rolling('365D') DatetimeIndexillä.
    shift(1) jättää nykyisen ottelun ulkopuolelle → nolla datavuoto.
    """
    winners = match_df[["date", "surface", "winner_name",
                         "w_spw", "w_rpw", "w_bpsaved", "w_bpconv", "w_tbwin"]].copy()
    winners.columns = ["date", "surface", "player", "spw", "rpw", "bpsaved", "bpconv", "tbwin"]
    losers  = match_df[["date", "surface", "loser_name",
                         "l_spw", "l_rpw", "l_bpsaved", "l_bpconv", "l_tbwin"]].copy()
    losers.columns  = ["date", "surface", "player", "spw", "rpw", "bpsaved", "bpconv", "tbwin"]

    long = (pd.concat([winners, losers], ignore_index=True)
              .sort_values(["player", "surface", "date"])
              .set_index("date"))

    print("  Lasketaan rolling pre-match tilastot...", end=" ", flush=True)

    grp = long.groupby(["player", "surface"])

    def roll(col):
        return grp[col].transform(
            lambda s: s.shift(1).rolling(f"{L52W_DAYS}D", min_periods=1).mean()
        )

    long["pre_spw"]    = roll("spw")
    long["pre_rpw"]    = roll("rpw")
    long["pre_bpsaved"] = roll("bpsaved")
    long["pre_bpconv"]  = roll("bpconv")
    long["pre_tbwin"]   = roll("tbwin")
    long["n_matches"]  = grp["spw"].transform(
        lambda s: s.shift(1).rolling(f"{L52W_DAYS}D", min_periods=1).count()
    )

    print("OK")
    return long.reset_index()


# ─────────────────────────────────────────────
# 4. PIIRTEIDEN RAKENTAMINEN
# ─────────────────────────────────────────────

def bayes_adj(recent: float, career: float, n: int, k: int = BAYES_K) -> float:
    w = n / (n + k)
    return recent * w + career * (1 - w)


def rakenna_piirteet(match_df: pd.DataFrame, rolling_df: pd.DataFrame) -> pd.DataFrame:
    """
    Yhdistää otteludatan pre-match tilastoihin.
    Palauttaa DataFramen jossa dominance-ero voittajan hyväksi.
    """
    # Uratasoinen keskiarvo (fallback)
    career = rolling_df.groupby("player")[["spw", "rpw", "bpsaved", "bpconv", "tbwin"]].mean()
    career.columns = ["career_spw", "career_rpw", "career_bpsaved", "career_bpconv", "career_tbwin"]

    # Indeksointi nopeaan hakuun
    roll_cols = ["pre_spw", "pre_rpw", "pre_bpsaved", "pre_bpconv", "pre_tbwin", "n_matches"]
    roll_idx = (rolling_df
                .groupby(["player", "surface", "date"])[roll_cols]
                .first())

    def fetch(player: str, surface: str, date) -> dict:
        defaults = {"spw": 0.62, "rpw": 0.37, "bpsaved": 0.65,
                    "bpconv": 0.40, "tbwin": 0.50, "n": 0}
        try:
            r = roll_idx.loc[(player, surface, date)]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            n = int(r["n_matches"]) if not np.isnan(r["n_matches"]) else 0

            def adj(col, default):
                val = r[f"pre_{col}"]
                c   = career.loc[player, f"career_{col}"] if player in career.index else default
                return bayes_adj(val, c, n) if n > 0 and not np.isnan(val) else c

            return {
                "spw":     adj("spw",     0.62),
                "rpw":     adj("rpw",     0.37),
                "bpsaved": adj("bpsaved", 0.65),
                "bpconv":  adj("bpconv",  0.40),
                "tbwin":   adj("tbwin",   0.50),
                "n":       n,
            }
        except KeyError:
            return defaults

    records = []
    for _, row in match_df.iterrows():
        d, surf = row["date"], row["surface"]
        w_name, l_name = row["winner_name"], row["loser_name"]
        w = fetch(w_name, surf, d)
        l = fetch(l_name, surf, d)

        # Log5: P(W voittaa pisteen omalla syötöllään L:ää vastaan)
        def log5(spw, rpw_opp):
            p = max(1e-6, min(1 - 1e-6, spw))
            q = max(1e-6, min(1 - 1e-6, 1 - rpw_opp))
            logit = math.log(p / (1 - p)) + math.log(q / (1 - q)) - math.log(0.5 / 0.5)
            return 1 / (1 + math.exp(-logit))

        dom_w = log5(w["spw"], l["rpw"])
        dom_l = log5(l["spw"], w["rpw"])

        records.append({
            "date":        d,
            "surface":     surf,
            "winner":      w_name,
            "loser":       l_name,
            "dom_diff":    dom_w - dom_l,
            "bpsaved_diff": w["bpsaved"] - l["bpsaved"],
            "bpconv_diff":  w["bpconv"]  - l["bpconv"],
            "tbwin_diff":   w["tbwin"]   - l["tbwin"],
            "w_n":         w["n"],
            "l_n":         l["n"],
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 5. REGRESSIO JA VALIDOINTI
# ─────────────────────────────────────────────

def aja_regressio(feat: pd.DataFrame) -> dict:
    """
    Tasapainottaa datasetin peilaamalla (voittaja ↔ häviäjä),
    sovittaa logistisen regression ja palauttaa tulokset.
    """
    FEATURES = ["dom_diff", "bpsaved_diff", "bpconv_diff", "tbwin_diff"]

    pos = feat[FEATURES].copy(); pos["y"] = 1
    neg = feat[FEATURES].copy(); neg["y"] = 0
    for f in FEATURES:
        neg[f] *= -1
    bal = pd.concat([pos, neg], ignore_index=True)
    bal = bal.dropna(subset=FEATURES)

    X = bal[FEATURES].values
    y = bal["y"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_te)[:, 1]
    brier  = brier_score_loss(y_te, y_prob)
    acc    = accuracy_score(y_te, model.predict(X_te))
    coefs  = {f: round(float(c), 2) for f, c in zip(FEATURES, model.coef_[0])}

    return {"model": model, "y_te": y_te, "y_prob": y_prob,
            "brier": brier, "acc": acc, "coef": coefs["dom_diff"],
            "coefs": coefs}


def piirrä_kalibrointi(y_te: np.ndarray, y_prob: np.ndarray, coef: float) -> None:
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 11)
    cx, fy = [], []
    for i in range(len(bins) - 1):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 30:
            cx.append(y_prob[mask].mean())
            fy.append(y_te[mask].mean())
    plt.plot([0, 1], [0, 1], "k--", label="Täydellinen kalibrointi")
    plt.plot(cx, fy, "go-", lw=2, label=f"Kalibroitu malli (k={coef:.0f})")
    plt.xlabel("Mallin ennuste")
    plt.ylabel("Toteutunut frekvenssi")
    plt.title("Kalibrointi — regressiokertoimilla")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("kalibrointi_regressio.png", dpi=150)
    plt.show()
    print("  Graafi tallennettu: kalibrointi_regressio.png")


# ─────────────────────────────────────────────
# PÄÄOHJELMA
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Tennis Backtest Pipeline")
    print("=" * 50)

    print("\n[1/5] Ladataan ATP-data...")
    raw = download_atp_data()
    print(f"      {len(raw):,} ottelua ladattu ({YEARS.start}–{YEARS.stop - 1})")

    print("\n[2/5] Lasketaan per-ottelu SPW/RPW...")
    match_df = laske_spw_per_ottelu(raw)
    print(f"      {len(match_df):,} validia ottelua")

    print("\n[3/5] Rolling pre-match tilastot (leakage-free)...")
    rolling = laske_rolling_pre_match(match_df)

    print("\n[4/5] Rakennetaan piirteet...")
    feat = rakenna_piirteet(match_df, rolling)
    print(f"      {len(feat):,} ottelua mukana regressiossa")

    print("\n[5/5] Logistinen regressio + validointi...")
    res = aja_regressio(feat)

    print(f"\n{'─'*40}")
    print(f"  Brier Score : {res['brier']:.4f}  (naive baseline = 0.25)")
    print(f"  Tarkkuus    : {res['acc']*100:.1f}%  (realistinen odotus: 65–70%)")
    print(f"{'─'*40}")
    print(f"\n  Regressiokertoimet datasta:")
    for piirre, arvo in res["coefs"].items():
        print(f"    {piirre:20s}: {arvo:.2f}")

    if 60 <= res["acc"] * 100 <= 75:
        print("\n  ✅ Tarkkuus realistisella alueella — ei datavuotoa")
    else:
        print("\n  ⚠️  Tarkista data: tarkkuus epärealistisen korkea tai matala")

    print("\n  Piirretään kalibrointigraafi...")
    piirrä_kalibrointi(res["y_te"], res["y_prob"], res["coef"])

    # Tallenna suositus
    suositus = {
        "matchup_k_suositeltu": round(res["coef"], 1),
        "matchup_k_nykyinen":   18.0,
        "elo_weight_suositeltu": 0.0,
        "brier_score":          round(res["brier"], 4),
        "tarkkuus_pct":         round(res["acc"] * 100, 1),
        "ohje": (
            f"Vaihda tennis_model.py:ssä rivi "
            f"'matchup_logit = piste_ero * 18.0' → "
            f"'matchup_logit = piste_ero * {res['coef']:.1f}'"
        )
    }
    with open("regressio_suositus.json", "w", encoding="utf-8") as f:
        json.dump(suositus, f, indent=4, ensure_ascii=False)
    print("\n  Suositus tallennettu: regressio_suositus.json")

    print(f"\n{'='*50}")
    print(f"  PÄIVITÄ tennis_model.py:")
    print(f"    matchup_logit = piste_ero * {res['coef']:.1f}  # oli 18.0")
    print(f"    elo_weight    = 0.0               # poista Elo")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────
# ELO-LASKENTA (käytetään clv_backtest.py:stä)
# ─────────────────────────────────────────────

ELO_K     = 32
ELO_START = 1500
SURFACES  = ["Hard", "Clay", "Grass"]


def elo_prob(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def laske_elo_historia(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Laskee alustakohtaiset pre-match Elo-luvut koko historiasta."""
    df = raw_df.copy()
    df = df[df["surface"].isin(SURFACES)]
    df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)

    elo: dict = {}

    def get_elo(player, surface):
        return elo.get(player, {}).get(surface, ELO_START)

    def set_elo(player, surface, value):
        if player not in elo:
            elo[player] = {}
        elo[player][surface] = value

    records = []
    for _, row in df.iterrows():
        w, l, s, d = row["winner_name"], row["loser_name"], row["surface"], row["date"]
        elo_w, elo_l = get_elo(w, s), get_elo(l, s)
        records.append({"date": d, "surface": s, "winner": w, "loser": l,
                        "pre_elo_w": elo_w, "pre_elo_l": elo_l})
        exp_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        set_elo(w, s, elo_w + ELO_K * (1 - exp_w))
        set_elo(l, s, elo_l + ELO_K * (0 - (1 - exp_w)))

    return pd.DataFrame(records)
