"""
Tennisvedonlyöntimallin Backtesting-skripti v2
===============================================
Testaa Elo- ja SPW/RPW-signaalien optimaalisen painotuksen
sekä muut mallin parametrit historiallisella datalla.

Asenna riippuvuudet:
    pip install requests pandas numpy matplotlib scipy

Ajo:
    python backtest.py
"""

import math
import json
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from io import StringIO

# ============================================================
# 1. MALLIN LASKENTALOGIIKKA
# ============================================================

def calculate_elo_prob(elo_a, elo_b):
    return 1 / (1 + math.pow(10, (elo_b - elo_a) / 400))

def to_logit(p):
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))

def to_prob(logit_val):
    return 1 / (1 + math.exp(-logit_val))

def get_bayesian_weight(matches, k_factor=10):
    return matches / (matches + k_factor)

def calculate_adjusted_stat(stat_l52w, stat_career, matches_l52w):
    weight = get_bayesian_weight(matches_l52w)
    return (stat_l52w * weight) + (stat_career * (1 - weight))

def run_model(row, params=None):
    """
    params = [matchup_kerroin, elo_weight]
    elo_weight: 0.0 = pelkkä SPW/RPW, 1.0 = pelkkä Elo, 0.5 = tasan
    """
    if params is None:
        matchup_k = 18.0
        elo_weight = 1.0  # Nykyinen malli: Elo täysillä + SPW/RPW päälle
    else:
        matchup_k = params[0]
        elo_weight = params[1]

    elo_a = row['elo_a']
    elo_b = row['elo_b']
    spw_a = row['spw_a']
    rpw_a = row['rpw_a']
    spw_b = row['spw_b']
    rpw_b = row['rpw_b']
    matches_a = row.get('matches_a', 20)
    matches_b = row.get('matches_b', 20)
    court_speed = row.get('court_speed', 0.85)

    # Elo-signaali logit-avaruudessa
    base_prob_a = calculate_elo_prob(elo_a, elo_b)
    elo_logit = to_logit(base_prob_a)

    # SPW/RPW-signaali logit-avaruudessa
    adj_spw_a = calculate_adjusted_stat(spw_a, spw_a, matches_a)
    adj_rpw_a = calculate_adjusted_stat(rpw_a, rpw_a, matches_a)
    adj_spw_b = calculate_adjusted_stat(spw_b, spw_b, matches_b)
    adj_rpw_b = calculate_adjusted_stat(rpw_b, rpw_b, matches_b)

    piste_voitto_a = (adj_spw_a + (1 - adj_rpw_b)) / 2
    piste_voitto_b = (adj_spw_b + (1 - adj_rpw_a)) / 2
    piste_ero = piste_voitto_a - piste_voitto_b
    matchup_logit = piste_ero * matchup_k * (court_speed ** 1.5)

    # YHDISTÄMINEN PAINOTETUSTI
    # elo_weight=1.0: pelkkä Elo (ei SPW/RPW)
    # elo_weight=0.5: puoliksi molempia
    # elo_weight=0.0: pelkkä SPW/RPW (ei Eloa)
    final_logit = (elo_weight * elo_logit) + ((1 - elo_weight) * matchup_logit)

    return to_prob(final_logit)


# ============================================================
# 2. DATAN HAKU
# ============================================================

SURFACE_SPEEDS = {
    'Clay': 0.80,
    'Hard': 1.00,
    'Grass': 1.25,
}

def hae_ottelu_csv(vuosi: int):
    url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{vuosi}.csv"
    print(f"  Haetaan {vuosi}...")
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text), low_memory=False)
    except Exception as e:
        print(f"  Virhe {vuosi}: {e}")
        return None

def prepare_backtest_data(years):
    all_matches = []
    for year in years:
        df = hae_ottelu_csv(year)
        if df is None:
            continue
        required = ['winner_rank', 'loser_rank', 'w_svpt', 'w_1stWon',
                    'w_2ndWon', 'l_svpt', 'l_1stWon', 'l_2ndWon', 'surface']
        df = df.dropna(subset=required)
        df = df[df['winner_rank'] > 0]
        df = df[df['loser_rank'] > 0]

        for _, row in df.iterrows():
            try:
                elo_w = max(1200, 2100 - math.log(float(row['winner_rank']) + 1) * 200)
                elo_l = max(1200, 2100 - math.log(float(row['loser_rank']) + 1) * 200)
                w_svpt = float(row['w_svpt'])
                l_svpt = float(row['l_svpt'])
                if w_svpt < 10 or l_svpt < 10:
                    continue
                spw_w = (float(row['w_1stWon']) + float(row['w_2ndWon'])) / w_svpt
                spw_l = (float(row['l_1stWon']) + float(row['l_2ndWon'])) / l_svpt
                rpw_w = 1 - spw_l
                rpw_l = 1 - spw_w
                surface = row.get('surface', 'Hard')
                court_speed = SURFACE_SPEEDS.get(surface, 1.0)

                if hash(str(row.get('match_num', 0)) + str(year)) % 2 == 0:
                    all_matches.append({
                        'elo_a': elo_w, 'elo_b': elo_l,
                        'spw_a': spw_w, 'rpw_a': rpw_w,
                        'spw_b': spw_l, 'rpw_b': rpw_l,
                        'matches_a': 20, 'matches_b': 20,
                        'court_speed': court_speed,
                        'surface': surface, 'result': 1, 'year': year,
                    })
                else:
                    all_matches.append({
                        'elo_a': elo_l, 'elo_b': elo_w,
                        'spw_a': spw_l, 'rpw_a': rpw_l,
                        'spw_b': spw_w, 'rpw_b': rpw_w,
                        'matches_a': 20, 'matches_b': 20,
                        'court_speed': court_speed,
                        'surface': surface, 'result': 0, 'year': year,
                    })
            except (ValueError, ZeroDivisionError):
                continue

    df_out = pd.DataFrame(all_matches)
    print(f"\nYhteensä {len(df_out)} ottelua.")
    return df_out


# ============================================================
# 3. METRIIKAT
# ============================================================

def brier_score(probs, results):
    return float(np.mean((np.array(probs) - np.array(results)) ** 2))

def accuracy(probs, results):
    return float(np.mean((np.array(probs) >= 0.5).astype(int) == np.array(results)))

def calibration_data(probs, results, n_bins=10):
    probs = np.array(probs)
    results = np.array(results)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, freqs, counts = [], [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            centers.append(probs[mask].mean())
            freqs.append(results[mask].mean())
            counts.append(mask.sum())
    return np.array(centers), np.array(freqs), np.array(counts)


# ============================================================
# 4. ELO-PAINO ANALYYSI
# ============================================================

def analysoi_elo_paino(df):
    print("\nAnalysoidaan Elo-painon vaikutusta (0.0 → 1.0)...")
    results = df['result'].values
    painot = np.arange(0.0, 1.05, 0.05)
    brier_scores = []

    for w in painot:
        probs = np.array([run_model(row, [18.0, w]) for _, row in df.iterrows()])
        bs = brier_score(probs, results)
        brier_scores.append(bs)
        print(f"  elo_weight={w:.2f} → Brier={bs:.4f}")

    paras_idx = int(np.argmin(brier_scores))
    paras_paino = float(painot[paras_idx])
    paras_brier = float(brier_scores[paras_idx])

    print(f"\n→ Paras elo_weight: {paras_paino:.2f} (Brier={paras_brier:.4f})")
    print(f"→ Nykyinen (elo_weight=1.0): Brier={brier_scores[-1]:.4f}")
    parannus = brier_scores[-1] - paras_brier
    print(f"→ Parannus: {parannus:.4f} ({parannus/brier_scores[-1]*100:.1f}%)")

    return {
        'painot': painot.tolist(),
        'brier_scores': [float(b) for b in brier_scores],
        'paras_paino': paras_paino,
        'paras_brier': paras_brier,
        'nykyinen_brier': float(brier_scores[-1]),
    }


def optimoi_parametrit(df, paras_elo_paino):
    print("\nOptimoidaan matchup_k ja elo_weight yhdessä...")
    results = df['result'].values

    def objective(params):
        probs = np.array([run_model(row, params) for _, row in df.iterrows()])
        return brier_score(probs, results)

    x0 = [18.0, paras_elo_paino]
    bounds = [(5.0, 40.0), (0.0, 1.0)]
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 100})

    return {
        'matchup_k': float(result.x[0]),
        'elo_weight': float(result.x[1]),
        'brier': float(result.fun),
        'success': bool(result.success),
    }


# ============================================================
# 5. VISUALISOINTI
# ============================================================

def plot_kaikki(df, elo_analyysi, opt_params):
    results = df['result'].values

    probs_nykyinen = np.array([run_model(row, [18.0, 1.0]) for _, row in df.iterrows()])
    probs_paras = np.array([
        run_model(row, [18.0, elo_analyysi['paras_paino']]) for _, row in df.iterrows()
    ])
    probs_opt = np.array([
        run_model(row, [opt_params['matchup_k'], opt_params['elo_weight']])
        for _, row in df.iterrows()
    ])

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Tennisvedonlyöntimalli — Elo-paino Analyysi', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Kaavio 1: Brier vs elo_weight
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(elo_analyysi['painot'], elo_analyysi['brier_scores'],
             'o-', color='steelblue', linewidth=2, markersize=5)
    ax1.axvline(elo_analyysi['paras_paino'], color='darkorange', linestyle='--',
                label=f"Paras: {elo_analyysi['paras_paino']:.2f}")
    ax1.axvline(1.0, color='red', linestyle=':', alpha=0.7, label='Nykyinen (1.0)')
    ax1.set_xlabel('Elo-paino  (0 = pelkkä SPW/RPW  ·  1 = pelkkä Elo)')
    ax1.set_ylabel('Brier Score (alempi = parempi)')
    ax1.set_title('Mikä Elo-paino toimii parhaiten?')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Kaavio 2: Kalibrointi
    ax2 = fig.add_subplot(gs[0, 2])
    for probs, label, color in [
        (probs_nykyinen, 'Nykyinen (1.0)', 'steelblue'),
        (probs_paras, f"Paras ({elo_analyysi['paras_paino']:.2f})", 'darkorange'),
        (probs_opt, 'Optimoitu', 'green'),
    ]:
        bc, af, _ = calibration_data(probs, results)
        ax2.plot(bc, af, 'o-', color=color, alpha=0.8, markersize=4, label=label)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax2.set_xlabel('Mallin ennuste'); ax2.set_ylabel('Toteutunut frekvenssi')
    ax2.set_title('Kalibrointi'); ax2.legend(fontsize=7)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3)

    # Kaavio 3: Alustittain
    ax3 = fig.add_subplot(gs[1, 0:2])
    surf_data = []
    for surf in df['surface'].unique():
        mask = df['surface'] == surf
        if mask.sum() < 20:
            continue
        r = results[mask]
        surf_data.append({
            'Alusta': f"{surf} (n={mask.sum()})",
            'Nykyinen': brier_score(probs_nykyinen[mask], r),
            'Paras paino': brier_score(probs_paras[mask], r),
            'Optimoitu': brier_score(probs_opt[mask], r),
        })
    if surf_data:
        df_s = pd.DataFrame(surf_data)
        x = np.arange(len(df_s)); w = 0.25
        ax3.bar(x - w, df_s['Nykyinen'], w, label='Nykyinen', color='steelblue', alpha=0.8)
        ax3.bar(x, df_s['Paras paino'], w, label='Paras paino', color='darkorange', alpha=0.8)
        ax3.bar(x + w, df_s['Optimoitu'], w, label='Optimoitu', color='green', alpha=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(df_s['Alusta'])
        ax3.set_ylabel('Brier Score'); ax3.set_title('Brier Score alustittain')
        ax3.legend(); ax3.grid(True, alpha=0.3, axis='y')

    # Kaavio 4: Taulukko
    ax4 = fig.add_subplot(gs[1, 2]); ax4.axis('off')
    td = [
        ['', 'Nykyinen', 'Paras paino', 'Optimoitu'],
        ['Brier ↓',
         f"{brier_score(probs_nykyinen, results):.4f}",
         f"{brier_score(probs_paras, results):.4f}",
         f"{brier_score(probs_opt, results):.4f}"],
        ['Tarkkuus ↑',
         f"{accuracy(probs_nykyinen, results):.1%}",
         f"{accuracy(probs_paras, results):.1%}",
         f"{accuracy(probs_opt, results):.1%}"],
        ['elo_weight', '1.00', f"{elo_analyysi['paras_paino']:.2f}", f"{opt_params['elo_weight']:.2f}"],
        ['matchup_k', '18.00', '18.00', f"{opt_params['matchup_k']:.2f}"],
    ]
    tbl = ax4.table(cellText=td[1:], colLabels=td[0], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.9)
    for j in range(4):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    ax4.set_title('Yhteenveto', fontweight='bold', pad=20)

    plt.savefig('backtest_raportti.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Kaavio tallennettu: backtest_raportti.png")
    plt.show()


# ============================================================
# 6. PÄÄOHJELMA
# ============================================================

def main():
    print("=" * 60)
    print("TENNISVEDONLYÖNTIMALLI — ELO-PAINO ANALYYSI")
    print("=" * 60)

    VUODET = [2022, 2023]
    print(f"\n1. Haetaan data vuosilta {VUODET}...")
    df = prepare_backtest_data(VUODET)
    if len(df) < 100:
        print("VIRHE: Liian vähän dataa.")
        return

    print("\n2. Analysoidaan Elo- ja SPW/RPW-signaalien optimaalinen paino...")
    elo_analyysi = analysoi_elo_paino(df)

    print("\n3. Optimoidaan matchup_k ja elo_weight yhdessä...")
    opt_params = optimoi_parametrit(df, elo_analyysi['paras_paino'])
    print(f"   matchup_k: {opt_params['matchup_k']:.2f}")
    print(f"   elo_weight: {opt_params['elo_weight']:.2f}")

    # Tallenna
    paino = elo_analyysi['paras_paino']
    if paino >= 0.95:
        selitys = "Pelkkä Elo riittää — SPW/RPW ei tuo lisäarvoa"
    elif paino <= 0.05:
        selitys = "Pelkkä SPW/RPW riittää — Elo ei tuo lisäarvoa"
    else:
        selitys = f"Elo {paino*100:.0f}% + SPW/RPW {(1-paino)*100:.0f}% on optimaalisin yhdistelmä"

    output = {
        'suositus': {
            'elo_weight': round(paino, 2),
            'matchup_k': round(opt_params['matchup_k'], 2),
            'selitys': selitys,
        },
        'parannus_nykyiseen': f"{(elo_analyysi['nykyinen_brier'] - elo_analyysi['paras_brier']) / elo_analyysi['nykyinen_brier'] * 100:.1f}%",
    }
    with open('backtest_tulokset.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUOSITUS:")
    print("=" * 60)
    print(f"  {selitys}")
    print(f"  Parannus nykyiseen malliin: {output['parannus_nykyiseen']}")
    print("\nTulokset: backtest_tulokset.json")

    print("\n4. Piirretään kaaviot...")
    plot_kaikki(df, elo_analyysi, opt_params)
    print("\nVALMIS.")


if __name__ == '__main__':
    main()
