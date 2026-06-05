"""
Tennisvedonlyöntimallin Datan Hakuputki
========================================
Hakee automaattisesti pelaajan SPW, RPW ja cElo-arvot
Jeff Sackin ATP-tietokannasta (GitHub) ennen jokaista ottelua.

Asenna riippuvuudet:
    pip install requests pandas numpy

Käyttö:
    python data_pipeline.py --pelaaja_a "Carlos Alcaraz" --pelaaja_b "Jannik Sinner" --alusta Clay

    tai interaktiivisesti:
    python data_pipeline.py
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests

# ============================================================
# ASETUKSET
# ============================================================

CACHE_HAKEMISTO = ".tennis_cache"
CACHE_VANHENEMINEN_H = 24  # Välimuisti vanhenee 24 tunnissa

# Jeff Sackmanin GitHub-repo — avointa dataa, ei API-avainta tarvita
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

ALUSTA_MAP = {
    "Massa": "Clay",
    "Kova": "Hard",
    "Ruoho": "Grass",
    "Clay": "Clay",
    "Hard": "Hard",
    "Grass": "Grass",
}

SURFACE_SPEEDS = {
    "Clay": 0.80,
    "Hard": 1.00,
    "Grass": 1.25,
}

# Kuinka monta viimeistä vuotta L52w-laskentaan
L52W_VUODET = 1  # Viimeiset ~52 viikkoa = 1 vuosi


# ============================================================
# 1. VÄLIMUISTI (Cache) — vältetään turhat verkkohaut
# ============================================================

def _cache_polku(avain: str) -> str:
    os.makedirs(CACHE_HAKEMISTO, exist_ok=True)
    # Sanitoidaan avain tiedostonimeksi
    turvallinen = avain.replace("/", "_").replace(" ", "_").replace(":", "_")
    return os.path.join(CACHE_HAKEMISTO, f"{turvallinen}.json")

def cache_hae(avain: str):
    polku = _cache_polku(avain)
    if not os.path.exists(polku):
        return None
    try:
        with open(polku, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Tarkista vanheneminen
        tallennettu = datetime.fromisoformat(data["_aikaleima"])
        if datetime.now() - tallennettu > timedelta(hours=CACHE_VANHENEMINEN_H):
            return None
        return data["arvo"]
    except Exception:
        return None

def cache_tallenna(avain: str, arvo):
    polku = _cache_polku(avain)
    try:
        with open(polku, "w", encoding="utf-8") as f:
            json.dump({"_aikaleima": datetime.now().isoformat(), "arvo": arvo}, f)
    except Exception:
        pass


# ============================================================
# 2. DATAN HAKU GITHUBISTA
# ============================================================

def hae_ottelu_csv(vuosi: int) -> pd.DataFrame | None:
    """Hakee ATP-otteluiden CSV-tiedoston annetulle vuodelle."""
    cache_avain = f"atp_matches_{vuosi}"
    cached = cache_hae(cache_avain)
    if cached is not None:
        return pd.DataFrame(cached)

    url = f"{BASE_URL}/atp_matches_{vuosi}.csv"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), low_memory=False)
        cache_tallenna(cache_avain, df.to_dict(orient="records"))
        return df
    except requests.HTTPError as e:
        if r.status_code == 404:
            return None  # Vuotta ei vielä olemassa
        print(f"  HTTP-virhe {vuosi}: {e}")
        return None
    except Exception as e:
        print(f"  Virhe haettaessa {vuosi}: {e}")
        return None

def hae_pelaaja_csv() -> pd.DataFrame | None:
    """Hakee ATP-pelaajien master-tiedoston (nimet, ID:t)."""
    cache_avain = "atp_players"
    cached = cache_hae(cache_avain)
    if cached is not None:
        return pd.DataFrame(cached)

    url = f"{BASE_URL}/atp_players.csv"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        cache_tallenna(cache_avain, df.to_dict(orient="records"))
        return df
    except Exception as e:
        print(f"  Virhe haettaessa pelaajatietoja: {e}")
        return None


# ============================================================
# 3. PELAAJAN NIMEN ETSINTÄ
# ============================================================

def etsi_pelaaja_id(nimi: str, pelaajat_df: pd.DataFrame) -> tuple[int | None, str | None]:
    """
    Etsii pelaajan ID:n nimellä (osittainen haku, kirjainkoosta riippumaton).
    Palauttaa (player_id, virallinen_nimi) tai (None, None).
    """
    nimi_lower = nimi.lower().strip()

    # Muodostetaan koko nimi first_name + last_name
    pelaajat_df = pelaajat_df.copy()
    pelaajat_df["full_name"] = (
        pelaajat_df["name_first"].fillna("").str.lower()
        + " "
        + pelaajat_df["name_last"].fillna("").str.lower()
    ).str.strip()

    # Tarkka haku ensin
    tarkka = pelaajat_df[pelaajat_df["full_name"] == nimi_lower]
    if len(tarkka) == 1:
        r = tarkka.iloc[0]
        return int(r["player_id"]), f"{r['name_first']} {r['name_last']}"

    # Osahaku sukunimellä
    osahaku = pelaajat_df[
        pelaajat_df["name_last"].str.lower().str.contains(nimi_lower.split()[-1], na=False)
    ]
    if len(osahaku) == 1:
        r = osahaku.iloc[0]
        return int(r["player_id"]), f"{r['name_first']} {r['name_last']}"
    elif len(osahaku) > 1:
        print(f"  Useita osumia nimelle '{nimi}':")
        for _, r in osahaku.head(5).iterrows():
            print(f"    {r['name_first']} {r['name_last']} (ID: {r['player_id']})")
        return None, None

    return None, None


# ============================================================
# 4. TILASTOJEN LASKENTA OTTELUTASOLTA
# ============================================================

def laske_tilastot(
    player_id: int,
    alusta: str,
    kaikki_ottelut: pd.DataFrame,
    l52w_paivat: int = 365,
) -> dict | None:
    """
    Laskee pelaajalle SPW%, RPW%, ottelumäärän ja ranking-Elo-approksimaation
    sekä koko uralta että viimeiseltä 52 viikolta.

    Palauttaa dict tai None jos dataa ei löydy.
    """
    if kaikki_ottelut is None or len(kaikki_ottelut) == 0:
        return None

    # Suodatetaan pelaajan ottelut (voitot + häviöt)
    voitot = kaikki_ottelut[kaikki_ottelut["winner_id"] == player_id].copy()
    hapiot = kaikki_ottelut[kaikki_ottelut["loser_id"] == player_id].copy()

    # Merkitään rooli
    voitot["rooli"] = "winner"
    hapiot["rooli"] = "loser"

    # Yhdistetään
    kaikki = pd.concat([voitot, hapiot], ignore_index=True)

    # Suodatetaan alusta
    if alusta != "All":
        kaikki = kaikki[kaikki["surface"] == alusta]

    if len(kaikki) == 0:
        return None

    # Päivämääräsuodatus L52w-laskentaa varten
    kaikki["tourney_date"] = pd.to_datetime(kaikki["tourney_date"], format="%Y%m%d", errors="coerce")
    raja_pvm = datetime.now() - timedelta(days=l52w_paivat)
    viimeaikainen = kaikki[kaikki["tourney_date"] >= raja_pvm]

    def laske_spw_rpw(df_osa):
        """Laskee SPW ja RPW annetulle ottelujoukolle."""
        syotto_pisteet = 0
        syotto_yhteensa = 0
        palautus_pisteet = 0
        palautus_yhteensa = 0
        ottelu_maara = 0

        for _, rivi in df_osa.iterrows():
            rooli = rivi["rooli"]

            if rooli == "winner":
                w_svpt = rivi.get("w_svpt", np.nan)
                w_won = (rivi.get("w_1stWon", 0) or 0) + (rivi.get("w_2ndWon", 0) or 0)
                l_svpt = rivi.get("l_svpt", np.nan)
                l_won = (rivi.get("l_1stWon", 0) or 0) + (rivi.get("l_2ndWon", 0) or 0)
            else:
                w_svpt = rivi.get("l_svpt", np.nan)
                w_won = (rivi.get("l_1stWon", 0) or 0) + (rivi.get("l_2ndWon", 0) or 0)
                l_svpt = rivi.get("w_svpt", np.nan)
                l_won = (rivi.get("w_1stWon", 0) or 0) + (rivi.get("w_2ndWon", 0) or 0)

            # Syöttö
            if pd.notna(w_svpt) and float(w_svpt) >= 5:
                syotto_pisteet += float(w_won)
                syotto_yhteensa += float(w_svpt)

            # Palautus = vastustajan syöttöpisteet käännettyinä
            if pd.notna(l_svpt) and float(l_svpt) >= 5:
                palautus_yhteensa += float(l_svpt)
                palautus_pisteet += float(l_svpt) - float(l_won)  # Voitetut palautuspisteet

            ottelu_maara += 1

        spw = (syotto_pisteet / syotto_yhteensa * 100) if syotto_yhteensa > 0 else None
        rpw = (palautus_pisteet / palautus_yhteensa * 100) if palautus_yhteensa > 0 else None
        return spw, rpw, ottelu_maara

    spw_ura, rpw_ura, n_ura = laske_spw_rpw(kaikki)
    spw_l52, rpw_l52, n_l52 = laske_spw_rpw(viimeaikainen)

    # Ranking → Elo-approksimaatio viimeisimmästä ottelusta
    viimeisin = kaikki.sort_values("tourney_date", ascending=False).iloc[0]
    if viimeisin["rooli"] == "winner":
        ranking = viimeisin.get("winner_rank", None)
    else:
        ranking = viimeisin.get("loser_rank", None)

    try:
        elo_approx = max(1200, 2100 - math.log(float(ranking) + 1) * 200)
    except (ValueError, TypeError):
        elo_approx = 1700  # Oletusarvo jos ranking puuttuu

    return {
        "spw_ura": round(spw_ura, 1) if spw_ura else None,
        "rpw_ura": round(rpw_ura, 1) if rpw_ura else None,
        "spw_l52": round(spw_l52, 1) if spw_l52 else None,
        "rpw_l52": round(rpw_l52, 1) if rpw_l52 else None,
        "ottelu_maara_ura": n_ura,
        "ottelu_maara_l52": n_l52,
        "elo_approx": round(elo_approx),
        "ranking": int(ranking) if pd.notna(ranking) else None,
        "alusta": alusta,
    }


# ============================================================
# 5. PÄÄFUNKTIO — haetaan pelaajan kaikki tilastot
# ============================================================

def hae_pelaajan_tilastot(
    nimi: str,
    alusta_fi: str,
    vuodet: list[int] | None = None,
) -> dict | None:
    """
    Hakee pelaajan tilastot automaattisesti.
    Tämä on putkilinjan pääfunktio — kutsu tätä app.py:stä.

    Args:
        nimi: Pelaajan nimi (esim. "Carlos Alcaraz" tai "Alcaraz")
        alusta_fi: Alusta suomeksi tai englanniksi ("Massa", "Clay" jne.)
        vuodet: Lista vuosista dataa varten. Oletus: [tänä_vuonna - 1, tänä_vuonna]

    Returns:
        dict tilastoilla tai None jos pelaajaa ei löydy
    """
    if vuodet is None:
        nyky = datetime.now().year
        vuodet = [nyky - 2, nyky - 1, nyky]

    alusta_en = ALUSTA_MAP.get(alusta_fi, "Hard")

    print(f"\n{'='*50}")
    print(f"Haetaan tilastot: {nimi} | {alusta_fi} ({alusta_en})")
    print(f"{'='*50}")

    # Haetaan pelaajatietokanta
    print("Ladataan pelaajatietokanta...")
    pelaajat_df = hae_pelaaja_csv()
    if pelaajat_df is None:
        print("VIRHE: Pelaajatietokantaa ei saatu ladattua.")
        return None

    # Etsitään pelaajan ID
    player_id, virallinen_nimi = etsi_pelaaja_id(nimi, pelaajat_df)
    if player_id is None:
        print(f"VIRHE: Pelaajaa '{nimi}' ei löydy tietokannasta.")
        print("Vinkki: Kokeile sukunimellä tai tarkista kirjoitusasu.")
        return None

    print(f"Löydettiin: {virallinen_nimi} (ID: {player_id})")

    # Haetaan otteludata usealta vuodelta
    print(f"Ladataan otteludata vuosilta {vuodet}...")
    kehykset = []
    for vuosi in vuodet:
        df = hae_ottelu_csv(vuosi)
        if df is not None:
            kehykset.append(df)
            print(f"  {vuosi}: {len(df)} ottelua")
        time.sleep(0.3)  # Kohteliaisuusviive GitHubille

    if not kehykset:
        print("VIRHE: Ei otteludataa saatavilla.")
        return None

    kaikki_ottelut = pd.concat(kehykset, ignore_index=True)

    # Lasketaan tilastot
    print(f"\nLasketaan tilastot alustalla '{alusta_en}'...")
    tilastot = laske_tilastot(player_id, alusta_en, kaikki_ottelut)

    if tilastot is None:
        print(f"VAROITUS: Ei tarpeeksi dataa alustalla '{alusta_en}'.")
        print("Kokeillaan kaikilla alustoilla...")
        tilastot = laske_tilastot(player_id, "All", kaikki_ottelut)
        if tilastot:
            tilastot["huomio"] = "Tilastot laskettu kaikilla alustoilla (ei alustaspesifistä dataa)"

    if tilastot is None:
        print("VIRHE: Ei löydy dataa pelaajalle.")
        return None

    tilastot["nimi"] = virallinen_nimi
    tilastot["player_id"] = player_id
    tilastot["haettu"] = datetime.now().isoformat()

    return tilastot


def tulosta_tilastot(tilastot: dict, nimi: str):
    """Tulostaa tilastot siististi."""
    print(f"\n📊 TILASTOT: {tilastot.get('nimi', nimi)}")
    print(f"   Alusta:      {tilastot['alusta']}")
    print(f"   Ranking:     #{tilastot['ranking']}")
    print(f"   cElo (approx): {tilastot['elo_approx']}")
    print()
    print(f"   {'Tilasto':<15} {'Koko ura':>10} {'L52 viikkoa':>12} {'Ottelut (L52)':>14}")
    print(f"   {'-'*55}")
    print(f"   {'SPW %':<15} {tilastot['spw_ura'] or 'N/A':>10} {tilastot['spw_l52'] or 'N/A':>12} {tilastot['ottelu_maara_l52']:>14}")
    print(f"   {'RPW %':<15} {tilastot['rpw_ura'] or 'N/A':>10} {tilastot['rpw_l52'] or 'N/A':>12}")

    if "huomio" in tilastot:
        print(f"\n   ⚠️  {tilastot['huomio']}")


# ============================================================
# 6. APP.PY -INTEGRAATIO (Streamlit-yhteensopiva)
# ============================================================

def hae_pelaajapari(
    nimi_a: str,
    nimi_b: str,
    alusta_fi: str,
) -> tuple[dict | None, dict | None]:
    """
    Hakee molemmat pelaajat yhdellä kutsulla.
    Suunniteltu kutsuttavaksi Streamlit-sovelluksesta.

    Käyttö app.py:ssä:
        from data_pipeline import hae_pelaajapari
        tilastot_a, tilastot_b = hae_pelaajapari("Alcaraz", "Sinner", "Massa")
    """
    tilastot_a = hae_pelaajan_tilastot(nimi_a, alusta_fi)
    tilastot_b = hae_pelaajan_tilastot(nimi_b, alusta_fi)
    return tilastot_a, tilastot_b


def tilastot_lomakkeelle(tilastot: dict) -> dict:
    """
    Muuntaa pipeline-tuloksen suoraan app.py-lomakkeen kenttiin sopivaksi.
    Palauttaa dict jonka avaimet vastaavat Streamlit-widgettien oletusarvoja.
    """
    if tilastot is None:
        return {}

    return {
        "elo": tilastot.get("elo_approx", 1800),
        "spw_l52": tilastot.get("spw_l52") or tilastot.get("spw_ura") or 63.0,
        "rpw_l52": tilastot.get("rpw_l52") or tilastot.get("rpw_ura") or 37.0,
        "spw_ura": tilastot.get("spw_ura") or 63.0,
        "rpw_ura": tilastot.get("rpw_ura") or 37.0,
        "matches_l52": tilastot.get("ottelu_maara_l52", 10),
    }


# ============================================================
# 7. KOMENTORIVI-KÄYTTÖLIITTYMÄ
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hae tennispelaajan tilastot automaattisesti"
    )
    parser.add_argument("--pelaaja_a", type=str, default=None, help="Pelaaja A:n nimi")
    parser.add_argument("--pelaaja_b", type=str, default=None, help="Pelaaja B:n nimi")
    parser.add_argument(
        "--alusta",
        type=str,
        default="Hard",
        choices=["Clay", "Hard", "Grass", "Massa", "Kova", "Ruoho"],
        help="Ottelun alusta"
    )
    parser.add_argument(
        "--tallenna",
        action="store_true",
        help="Tallenna tulokset pelaajatietokanta.json-tiedostoon (app.py-yhteensopiva)"
    )
    args = parser.parse_args()

    # Interaktiivinen tila jos nimiä ei annettu
    if args.pelaaja_a is None:
        print("\n🎾 Tennistilastojen Hakuputki")
        print("─" * 40)
        args.pelaaja_a = input("Pelaaja A:n nimi: ").strip()
        args.pelaaja_b = input("Pelaaja B:n nimi: ").strip()
        alusta_valinta = input("Alusta (Clay/Hard/Grass): ").strip() or "Hard"
        args.alusta = alusta_valinta
        args.tallenna = input("Tallenna tietokantaan? (k/e): ").strip().lower() == "k"

    # Hae tilastot
    tilastot_a = hae_pelaajan_tilastot(args.pelaaja_a, args.alusta)
    tilastot_b = hae_pelaajan_tilastot(args.pelaaja_b, args.alusta) if args.pelaaja_b else None

    # Tulosta
    if tilastot_a:
        tulosta_tilastot(tilastot_a, args.pelaaja_a)
    if tilastot_b:
        tulosta_tilastot(tilastot_b, args.pelaaja_b)

    # Tallenna app.py:n tietokantaformaattiin
    if args.tallenna and (tilastot_a or tilastot_b):
        alusta_fi = {v: k for k, v in ALUSTA_MAP.items() if k in ["Massa", "Kova", "Ruoho"]}.get(
            ALUSTA_MAP.get(args.alusta, "Hard"), "Kova"
        )
        tietokanta_polku = "pelaajatietokanta.json"
        if os.path.exists(tietokanta_polku):
            with open(tietokanta_polku, "r", encoding="utf-8") as f:
                tietokanta = json.load(f)
        else:
            tietokanta = {}

        for tilastot, nimi in [(tilastot_a, args.pelaaja_a), (tilastot_b, args.pelaaja_b)]:
            if tilastot is None:
                continue
            virallinen_nimi = tilastot["nimi"]
            kentat = tilastot_lomakkeelle(tilastot)
            if virallinen_nimi not in tietokanta:
                tietokanta[virallinen_nimi] = {}
            tietokanta[virallinen_nimi][alusta_fi] = {
                "ika": 25,  # Täytä manuaalisesti tai lisää DOB-laskenta
                "elo": kentat["elo"],
                "spw_l52": kentat["spw_l52"],
                "rpw_l52": kentat["rpw_l52"],
                "spw_ura": kentat["spw_ura"],
                "rpw_ura": kentat["rpw_ura"],
                "matches_l52": kentat["matches_l52"],
            }

        with open(tietokanta_polku, "w", encoding="utf-8") as f:
            json.dump(tietokanta, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Tallennettu tiedostoon: {tietokanta_polku}")

    # Näytä app.py-yhteensopivat arvot
    if tilastot_a:
        print(f"\n📋 KOPIOI SUORAAN APP.PY:HÖN ({args.pelaaja_a}):")
        kentat = tilastot_lomakkeelle(tilastot_a)
        for k, v in kentat.items():
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
