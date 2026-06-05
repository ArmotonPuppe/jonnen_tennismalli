import streamlit as st
import math
import os
import json

st.set_page_config(page_title="Tennisvedonlyönnin Lokaali Malli", layout="wide")

# --- LASKENTALOGIIKKA ---

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

def calculate_handicaps_from_match_prob(match_prob, format_type):
    """Laskee erätasoitukset analyyttisesti momentum-korjauksella."""
    print(format_type)
    low, high = 0.0, 1.0
    for _ in range(20):
        p = (low + high) / 2
        if format_type == "ATP (Paras 3:sta)":
            calc_m = (p**2) * (3 - 2*p)
        else:
            calc_m = (p**3) * (10 - 15*p + 6*(p**2))
        if calc_m > match_prob:
            high = p
        else:
            low = p
    p_a = p
    p_b = 1 - p_a
    MOMENTUM = 0.07
    tasaisuus = 4 * p_a * p_b
    if format_type == "ATP (Paras 3:sta)":
        hc_minus_a = max(0.01, p_a**2 - MOMENTUM * tasaisuus)
        hc_plus_a  = max(0.01, 1 - p_b**2 - MOMENTUM * tasaisuus)
        hc_minus_b = max(0.01, p_b**2 - MOMENTUM * tasaisuus)
        hc_plus_b  = max(0.01, 1 - p_a**2 - MOMENTUM * tasaisuus)
    else:
        hc_minus_a = max(0.01, (p_a**3 + 3*p_a**3*p_b) - MOMENTUM * tasaisuus * 1.4)
        hc_plus_a  = max(0.01, 1 - (p_b**3 + 3*p_b**3*p_a) - MOMENTUM * tasaisuus * 1.4)
        hc_minus_b = max(0.01, (p_b**3 + 3*p_b**3*p_a) - MOMENTUM * tasaisuus * 1.4)
        hc_plus_b  = max(0.01, 1 - (p_a**3 + 3*p_a**3*p_b) - MOMENTUM * tasaisuus * 1.4)

    print(hc_minus_a, hc_plus_a, hc_minus_b,hc_plus_b)
    return hc_minus_a, hc_plus_a, hc_minus_b, hc_plus_b


TIETOKANTA_TIEDOSTO = "pelaajatietokanta.json"

def lataa_pelaajat():
    if os.path.exists(TIETOKANTA_TIEDOSTO):
        with open(TIETOKANTA_TIEDOSTO, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def tallenna_pelaaja_kantaan(nimi, alusta, ika, elo, spw_l52, rpw_l52, spw_ura, rpw_ura, matches_l52):
    if nimi.strip() == "" or nimi == "Pelaaja A" or nimi == "Pelaaja B":
        return False
    pelaajat = lataa_pelaajat()
    if nimi not in pelaajat:
        pelaajat[nimi] = {}
    pelaajat[nimi][alusta] = {
        "ika": ika,
        "elo": elo,
        "spw_l52": spw_l52,
        "rpw_l52": rpw_l52,
        "spw_ura": spw_ura,
        "rpw_ura": rpw_ura,
        "matches_l52": matches_l52
    }
    with open(TIETOKANTA_TIEDOSTO, 'w', encoding='utf-8') as f:
        json.dump(pelaajat, f, indent=4)
    return True


# --- KÄYTTÖLIITTYMÄ ---
st.title("🎾 Pro-tason Tennisvedonlyöntimalli")
st.markdown("Syötä raakadata. Malli laskee bayesilaiset todennäköisyydet ja +EV-rajat.")

tallennetut_pelaajat = lataa_pelaajat()
pelaajien_nimet = ["-- Syötä uusi manuaalisesti --"] + sorted(list(tallennetut_pelaajat.keys()))

st.markdown("---")
st.subheader("Ottelun perusasetukset")
kolumni_alusta, kolumni_formaatti = st.columns(2)

with kolumni_alusta:
    valittu_alusta = st.selectbox("Valitse ottelun alusta", ["Massa", "Kova", "Ruoho"])

with kolumni_formaatti:
    otteluformaatti = st.radio("Turnausformaatti", ["ATP (Paras 3:sta)", "Grand Slam (Paras 5:stä)"])
st.markdown("---")

st.sidebar.header("Turnauksen Asetukset")
court_speed = st.sidebar.slider(
    "TA Surface Speed Rating (Kenttänopeus)", 0.5, 1.5, 0.85, 0.01,
    help="Rooma ~0.57, Madrid ~0.77, Wimbledon ~1.25")
ev_margin = st.sidebar.number_input(
    "Vaadittu arvoetu / +EV marginaali (%)", min_value=0.0, value=3.0, step=0.5)

col1, col2 = st.columns(2)

# --- PELAAJA A ---
with col1:
    st.header("Pelaaja A")
    valittu_a = st.selectbox("Valitse tietokannasta (A)", pelaajien_nimet, key="sel_a")

    if valittu_a != "-- Syötä uusi manuaalisesti --" and valittu_alusta in tallennetut_pelaajat.get(valittu_a, {}):
        data_a = tallennetut_pelaajat[valittu_a][valittu_alusta]
        nimi_a         = st.text_input("Pelaajan nimi (A)", value=valittu_a, key="n_a_db")
        elo_a          = st.number_input(f"cElo ({valittu_alusta}) (A)", value=data_a["elo"], step=10, key="e_a_db")
        ika_a          = st.number_input("Pelaajan ikä (A)", value=data_a.get("ika", 25), step=1, key="age_a_db")
        st.subheader("Syöttö (SPW %)")
        spw_l52w_a_raw = st.number_input("SPW % L52w (A)", value=data_a["spw_l52"], step=0.5, key="s_l_a_db")
        spw_car_a_raw  = st.number_input("SPW % Ura (A)",  value=data_a["spw_ura"],  step=0.5, key="s_c_a_db")
        matches_l52w_a = st.number_input("Ottelut alustalla L52w (A)", value=data_a["matches_l52"], step=1, key="m_a_db")
        st.subheader("Palautus (RPW %)")
        rpw_l52w_a_raw = st.number_input("RPW % L52w (A)", value=data_a["rpw_l52"], step=0.5, key="r_l_a_db")
        rpw_car_a_raw  = st.number_input("RPW % Ura (A)",  value=data_a["rpw_ura"],  step=0.5, key="r_c_a_db")
    else:
        oletusnimi_a   = valittu_a if valittu_a != "-- Syötä uusi manuaalisesti --" else "Pelaaja A"
        nimi_a         = st.text_input("Pelaajan nimi (A)", value=oletusnimi_a, key="n_a_new")
        elo_a          = st.number_input(f"cElo ({valittu_alusta}) (A)", value=1950, step=10, key="e_a_new")
        ika_a          = st.number_input("Pelaajan ikä (A)", value=25, step=1, key="age_a_new")
        st.subheader("Syöttö (SPW %)")
        spw_l52w_a_raw = st.number_input("SPW % L52w (A)", value=65.0, step=0.5, key="s_l_a_new")
        spw_car_a_raw  = st.number_input("SPW % Ura (A)",  value=64.0, step=0.5, key="s_c_a_new")
        matches_l52w_a = st.number_input("Ottelut alustalla L52w (A)", value=12, step=1, key="m_a_new")
        st.subheader("Palautus (RPW %)")
        rpw_l52w_a_raw = st.number_input("RPW % L52w (A)", value=38.0, step=0.5, key="r_l_a_new")
        rpw_car_a_raw  = st.number_input("RPW % Ura (A)",  value=37.5, step=0.5, key="r_c_a_new")

    if st.button("💾 Tallenna pelaajan A profiili"):
        if tallenna_pelaaja_kantaan(nimi_a, valittu_alusta, ika_a, elo_a,
                                    spw_l52w_a_raw, rpw_l52w_a_raw,
                                    spw_car_a_raw, rpw_car_a_raw, matches_l52w_a):
            st.success(f"Profiili {nimi_a} ({valittu_alusta}) tallennettu!")

    st.subheader("Väsymys (Pelaaja A)")
    lepopaivat_a = st.number_input("Päiviä edellisestä ottelusta (A)", min_value=0, max_value=14, value=2, step=1,
                                   help="0 = pelasi eilen, 1 = yksi lepopäivä jne.")
    setit_a = st.selectbox("Edellisen ottelun settimäärä (A)", [2, 3, 4, 5], index=1,
                           help="Kuinka monta settiä edellinen ottelu kesti")
    erat_7pv_a = st.number_input("Eriä pelattu viimeisen 7 pv (A)", min_value=0, max_value=30, value=6, step=1,
                                 help="Laske yhteen kaikki erät viimeisen 7 päivän otteluista. 2-set ottelu = 2, 3-set = 3 jne.")

    st.subheader("Laadullinen arvio (Pelaaja A)")
    vire_a  = st.slider("Vire / H2H-etu (A) %", min_value=-5.0, max_value=5.0, value=0.0, step=0.5,
                        help="+ jos hyvässä vireessä, H2H-yliote tai nousujohteinen. – jos laskussa tai H2H-tappio.")
    riski_a = st.slider("Loukkaantumisriski / MTO (A) %", min_value=-8.0, max_value=0.0, value=0.0, step=0.5,
                        help="0 = ei riskiä. – jos MTO, loukkaantumisepäily tai vaivautunut.")


# --- PELAAJA B ---
with col2:
    st.header("Pelaaja B")
    valittu_b = st.selectbox("Valitse tietokannasta (B)", pelaajien_nimet, key="sel_b")

    if valittu_b != "-- Syötä uusi manuaalisesti --" and valittu_alusta in tallennetut_pelaajat.get(valittu_b, {}):
        data_b = tallennetut_pelaajat[valittu_b][valittu_alusta]
        nimi_b         = st.text_input("Pelaajan nimi (B)", value=valittu_b, key="n_b_db")
        elo_b          = st.number_input(f"cElo ({valittu_alusta}) (B)", value=data_b["elo"], step=10, key="e_b_db")
        ika_b          = st.number_input("Pelaajan ikä (B)", value=data_b.get("ika", 25), step=1, key="age_b_db")
        st.subheader("Syöttö (SPW %)")
        spw_l52w_b_raw = st.number_input("SPW % L52w (B)", value=data_b["spw_l52"], step=0.5, key="s_l_b_db")
        spw_car_b_raw  = st.number_input("SPW % Ura (B)",  value=data_b["spw_ura"],  step=0.5, key="s_c_b_db")
        matches_l52w_b = st.number_input("Ottelut alustalla L52w (B)", value=data_b["matches_l52"], step=1, key="m_b_db")
        st.subheader("Palautus (RPW %)")
        rpw_l52w_b_raw = st.number_input("RPW % L52w (B)", value=data_b["rpw_l52"], step=0.5, key="r_l_b_db")
        rpw_car_b_raw  = st.number_input("RPW % Ura (B)",  value=data_b["rpw_ura"],  step=0.5, key="r_c_b_db")
    else:
        oletusnimi_b   = valittu_b if valittu_b != "-- Syötä uusi manuaalisesti --" else "Pelaaja B"
        nimi_b         = st.text_input("Pelaajan nimi (B)", value=oletusnimi_b, key="n_b_new")
        elo_b          = st.number_input(f"cElo ({valittu_alusta}) (B)", value=1900, step=10, key="e_b_new")
        ika_b          = st.number_input("Pelaajan ikä (B)", value=25, step=1, key="age_b_new")
        st.subheader("Syöttö (SPW %)")
        spw_l52w_b_raw = st.number_input("SPW % L52w (B)", value=62.0, step=0.5, key="s_l_b_new")
        spw_car_b_raw  = st.number_input("SPW % Ura (B)",  value=63.0, step=0.5, key="s_c_b_new")
        matches_l52w_b = st.number_input("Ottelut alustalla L52w (B)", value=22, step=1, key="m_b_new")
        st.subheader("Palautus (RPW %)")
        rpw_l52w_b_raw = st.number_input("RPW % L52w (B)", value=36.0, step=0.5, key="r_l_b_new")
        rpw_car_b_raw  = st.number_input("RPW % Ura (B)",  value=35.5, step=0.5, key="r_c_b_new")

    if st.button("💾 Tallenna pelaajan B profiili"):
        if tallenna_pelaaja_kantaan(nimi_b, valittu_alusta, ika_b, elo_b,
                                    spw_l52w_b_raw, rpw_l52w_b_raw,
                                    spw_car_b_raw, rpw_car_b_raw, matches_l52w_b):
            st.success(f"Profiili {nimi_b} ({valittu_alusta}) tallennettu!")

    st.subheader("Väsymys (Pelaaja B)")
    lepopaivat_b = st.number_input("Päiviä edellisestä ottelusta (B)", min_value=0, max_value=14, value=2, step=1,
                                   help="0 = pelasi eilen, 1 = yksi lepopäivä jne.")
    setit_b = st.selectbox("Edellisen ottelun settimäärä (B)", [2, 3, 4, 5], index=1,
                           help="Kuinka monta settiä edellinen ottelu kesti")
    erat_7pv_b = st.number_input("Eriä pelattu viimeisen 7 pv (B)", min_value=0, max_value=30, value=6, step=1,
                                 help="Laske yhteen kaikki erät viimeisen 7 päivän otteluista. 2-set ottelu = 2, 3-set = 3 jne.")

    st.subheader("Laadullinen arvio (Pelaaja B)")
    vire_b  = st.slider("Vire / H2H-etu (B) %", min_value=-5.0, max_value=5.0, value=0.0, step=0.5,
                        help="+ jos hyvässä vireessä, H2H-yliote tai nousujohteinen. – jos laskussa tai H2H-tappio.")
    riski_b = st.slider("Loukkaantumisriski / MTO (B) %", min_value=-8.0, max_value=0.0, value=0.0, step=0.5,
                        help="0 = ei riskiä. – jos MTO, loukkaantumisepäily tai vaivautunut.")


st.markdown("---")

# --- MOOTTORI ---

if st.button("Laske Kertoimet", type="primary", use_container_width=True):

    # 1. Elo-perusta
    base_prob_a  = calculate_elo_prob(elo_a, elo_b)
    base_logit_a = to_logit(base_prob_a)

    # 2. Bayesilainen painotus SPW/RPW
    adj_spw_a = calculate_adjusted_stat(spw_l52w_a_raw / 100, spw_car_a_raw / 100, matches_l52w_a)
    adj_rpw_a = calculate_adjusted_stat(rpw_l52w_a_raw / 100, rpw_car_a_raw / 100, matches_l52w_a)
    adj_spw_b = calculate_adjusted_stat(spw_l52w_b_raw / 100, spw_car_b_raw / 100, matches_l52w_b)
    adj_rpw_b = calculate_adjusted_stat(rpw_l52w_b_raw / 100, rpw_car_b_raw / 100, matches_l52w_b)

    # 3. Matchup-etu logit-avaruudessa (log5-menetelmä)
    # Yhdistää SPW ja RPW log-odds-avaruudessa — välttää kaksinkertaislaskennan
    # P(A voittaa pisteen omalla syötöllään B:tä vastaan):
    logit_serve_a  = to_logit(adj_spw_a) + to_logit(1 - adj_rpw_b) - to_logit(0.5)
    logit_serve_b  = to_logit(adj_spw_b) + to_logit(1 - adj_rpw_a) - to_logit(0.5)
    piste_voitto_a = to_prob(logit_serve_a)
    piste_voitto_b = to_prob(logit_serve_b)
    piste_ero      = piste_voitto_a - piste_voitto_b
    matchup_logit  = piste_ero * 16.0

    # 4. Elo-ankkuri: 40% paino — parantaa MAE:ta, korjaa under-dispersion osittain
    elo_logit   = to_logit(base_prob_a)
    pohja_logit = 0.6 * matchup_logit + 0.4 * elo_logit

    # 5. Court speed: skaalautuu syöttöedun suuruuden mukaan (ei binaarinen)
    # Nopea kenttä hyödyttää paremmin syöttävää pelaajaa
    serve_ero   = adj_spw_a - adj_spw_b
    court_adj   = (court_speed - 1.0) * 0.4 * serve_ero * 8.0
    pohja_logit = pohja_logit + court_adj

    # 6. Väsymys — viimeisin ottelu + kumulatiivinen 7pv rasitus + ikäkerroin
    # Ikäkerroin: alle 25 = 1.0, 30 = 1.2, 35 = 1.5, 38+ = 1.8
    def ika_kerroin(ika):
        return max(1.0, 1.0 + (ika - 25) * 0.06) if ika > 25 else 1.0

    ik_a = ika_kerroin(ika_a)
    ik_b = ika_kerroin(ika_b)

    # Viimeisin ottelu: setit + lepopäivät
    viim_a = (setit_a - 2) * 1.5 + max(0, 2 - lepopaivat_a) * 2.0
    viim_b = (setit_b - 2) * 1.5 + max(0, 2 - lepopaivat_b) * 2.0

    # Kumulatiivinen 7pv rasitus: normaali viikko turnauksessa ~6-8 erää
    kum_a = max(0, erat_7pv_a - 6) * 0.8
    kum_b = max(0, erat_7pv_b - 6) * 0.8

    fatigue_a = (viim_a + kum_a) * ik_a
    fatigue_b = (viim_b + kum_b) * ik_b

    fatigue_logit     = ((fatigue_b - fatigue_a) / 100) * 4.0
    fatigue_abs       = (fatigue_a + fatigue_b) / 2
    fatigue_abs_logit = -(fatigue_abs / 100) * 0.5 * abs(pohja_logit)
    fatigue_logit     = fatigue_logit + fatigue_abs_logit

    # 7. Laadulliset säätimet
    kvalit_netto_a = (vire_a + riski_a) / 100
    kvalit_netto_b = (vire_b + riski_b) / 100
    kvalit_logit   = (kvalit_netto_a - kvalit_netto_b) * 4.0

    # 8. Puhdas matemaattinen (ilman laadullisia)
    pure_logit_a = pohja_logit + fatigue_logit
    pure_prob_a  = to_prob(pure_logit_a)
    pure_prob_b  = 1 - pure_prob_a

    # 9. Lopullinen + formaattikorjaus
    # BO5 (Grand Slam) vähentää varianssia — parempi pelaaja voittaa useammin.
    # Kerroin 1.15 on teoreettinen arvio; ideaalitilanteessa kalibroitaisiin GS-datalla.
    final_logit_a = pohja_logit + fatigue_logit + kvalit_logit
    if otteluformaatti == "Grand Slam (Paras 5:stä)":
        final_logit_a = final_logit_a * 1.15
    final_prob_a  = to_prob(final_logit_a)
    final_prob_b  = 1 - final_prob_a

    # 10. Slope-korjaus (kerroin 1/0.54 ≈ 1.85, mitattu malli_vs_markkina.py:stä)
    # Malli litistää todennäköisyyksiä kohti 50% — tämä venyttää ne takaisin.
    # Rajataan ±1.5 logitiin (~82%) jotta ääriarvoissa ei ylikorjata.
    # Käytä tätä early-markkinavertailuun, ei suoraan vetopäätöksiin.
    SLOPE_KORJAUS = 1 / 0.54
    korj_logit_a  = max(-1.5, min(1.5, final_logit_a)) * SLOPE_KORJAUS
    korj_prob_a   = to_prob(korj_logit_a)
    korj_prob_b   = 1 - korj_prob_a
    korj_odds_a   = 1 / korj_prob_a
    korj_odds_b   = 1 / korj_prob_b

    fair_odds_a = 1 / final_prob_a
    fair_odds_b = 1 / final_prob_b
    min_odds_a  = fair_odds_a * (1 + ev_margin / 100)
    min_odds_b  = fair_odds_b * (1 + ev_margin / 100)

    # --- TULOKSET ---
    st.header("📊 Analyysin Tulokset")

    on_kvalit = (vire_a != 0 or riski_a != 0 or vire_b != 0 or riski_b != 0)
    if on_kvalit:
        st.info(f"📐 Puhdas matemaattinen arvio (ilman laadullisia säätöjä): "
                f"**A: {pure_prob_a*100:.1f}%** — **B: {pure_prob_b*100:.1f}%**")

    res1, res2, res3 = st.columns(3)
    res1.metric("Voittotodennäköisyys", f"A: {final_prob_a*100:.1f} %",
                f"B: {final_prob_b*100:.1f} %", delta_color="off")
    res2.metric("Fair Odds (Rajat)", f"A: {fair_odds_a:.2f}",
                f"B: {fair_odds_b:.2f}", delta_color="off")
    res3.metric(f"Minimikerroin (+{ev_margin}% EV)", f"A: {min_odds_a:.2f}",
                f"B: {min_odds_b:.2f}", delta_color="off")

    with st.expander("📖 Kuinka tulkita tuloksia?"):
        st.markdown(f"""
**Voittotodennäköisyys**
Mallin arvio kummankin pelaajan voittomahdollisuudesta. Esim. 65% tarkoittaa että malli arvioi pelaajan voittavan 65 ottelua sadasta vastaavalla tilastoprofiililla.

**Fair Odds (Rajat)**
Teoreettinen "oikea" kerroin ilman vedonvälittäjän marginaalia. Tämä on tärkein luku vetovertailuun:
- Jos vedonvälittäjä tarjoaa **korkeampaa kerrointa** kuin fair odds → mahdollinen value-veto
- Jos vedonvälittäjä tarjoaa **matalampaa kerrointa** → veto on ylihinnoiteltu, älä veto

**Minimikerroin (+{ev_margin}% EV)**
Fair oddsin päälle lisätty turvamarginaali. Käytä tätä kun haluat varmistaa että vedossa on riittävästi edgeä ennen kuin se kannattaa tehdä.
- Veto kannattaa vain jos saat **vähintään tämän kertoimen** vedonvälittäjältä

**Fair Odds vs. Minimikerroin — milloin kumpaa katsot?**
- **Vertailu markkinaan** → katso Fair Odds. Onko tarjottu kerroin yli fair oddsin?
- **Vetopäätös** → katso Minimikerroin. Onko tarjottu kerroin yli minimikertoimen?
        """)

    st.markdown("---")
    st.subheader("📈 Slope-korjattu ennuste (early market -vertailu)")

    selvä_suosikki = min(fair_odds_a, fair_odds_b) < 1.30
    if selvä_suosikki:
        st.warning("Selvä suosikki (fair odds alle 1.30) — slope-korjattu ei ole luotettava, käytä raakaa ennustetta.")
    else:
        st.caption(
            "Malli aliarvioi suosikkeja ja yliarvioi altavastaajia hieman. "
            "Korjattu versio on kalibroitu vastaamaan markkinatasoa — käytä sitä early-kertoimien vertailuun."
        )

    korj_min_a = korj_odds_a * (1 + ev_margin / 100)
    korj_min_b = korj_odds_b * (1 + ev_margin / 100)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Korjattu todennäköisyys", f"A: {korj_prob_a*100:.1f} %",
              f"B: {korj_prob_b*100:.1f} %", delta_color="off")
    k2.metric("Korjattu fair odds", f"A: {korj_odds_a:.2f}",
              f"B: {korj_odds_b:.2f}", delta_color="off")
    k3.metric(f"Korjattu minimikerroin (+{ev_margin}% EV)", f"A: {korj_min_a:.2f}",
              f"B: {korj_min_b:.2f}", delta_color="off")
    siirto_a = korj_prob_a - final_prob_a
    k4.metric("Muutos raakaan", f"A: {siirto_a*100:+.1f} pp",
              f"B: {-siirto_a*100:+.1f} pp", delta_color="normal")

    hc_a_minus, hc_a_plus, hc_b_minus, hc_b_plus = \
        calculate_handicaps_from_match_prob(final_prob_a, otteluformaatti)

    st.markdown("---")
    st.header("🎯 Erätasoitukset (Set Handicaps)")
    hc1, hc2 = st.columns(2)
    hc1.subheader("Pelaaja A")
    hc1.write(f"**-1.5 Erää:** Todennäköisyys {hc_a_minus*100:.1f}% | Raja: {1/hc_a_minus:.2f}")
    hc1.write(f"**+1.5 Erää:** Todennäköisyys {hc_a_plus*100:.1f}% | Raja: {1/hc_a_plus:.2f}")
    hc2.subheader("Pelaaja B")
    hc2.write(f"**-1.5 Erää:** Todennäköisyys {hc_b_minus*100:.1f}% | Raja: {1/hc_b_minus:.2f}")
    hc2.write(f"**+1.5 Erää:** Todennäköisyys {hc_b_plus*100:.1f}% | Raja: {1/hc_b_plus:.2f}")

    with st.expander("Näytä konepellin alle (Matematiikan erittely)"):
        st.write(f"**Elo-perusta (40% paino):** A {base_prob_a*100:.1f}% | logit {elo_logit:+.3f}")
        st.write(f"**Pelaaja A SPW (bayesilainen):** {adj_spw_a*100:.1f}% | RPW: {adj_rpw_a*100:.1f}%")
        st.write(f"**Pelaaja B SPW (bayesilainen):** {adj_spw_b*100:.1f}% | RPW: {adj_rpw_b*100:.1f}%")
        st.write(f"**Pelaaja A L52w otoskoon paino:** {get_bayesian_weight(matches_l52w_a)*100:.0f}%")
        st.write(f"**Pelaaja B L52w otoskoon paino:** {get_bayesian_weight(matches_l52w_b)*100:.0f}%")
        st.write(f"**Matchup-logit (95% paino):** {matchup_logit:+.3f}")
        st.write(f"**Court speed -säätö ({court_speed}):** {court_adj:+.3f}")
        st.write(f"**Pohja-logit:** {pohja_logit:+.3f}")
        st.write(f"**Pelaaja A väsymys:** {fatigue_a:.1f} (setit {setit_a}, lepo {lepopaivat_a}pv, 7pv-erät {erat_7pv_a}, ikäkerroin {ik_a:.2f})")
        st.write(f"**Pelaaja B väsymys:** {fatigue_b:.1f} (setit {setit_b}, lepo {lepopaivat_b}pv, 7pv-erät {erat_7pv_b}, ikäkerroin {ik_b:.2f})")
        st.write(f"**Väsymyssäätö (logit):** {fatigue_logit:+.3f}")
        st.write(f"**Pelaaja A laadullinen:** vire {vire_a:+.1f}% + riski {riski_a:+.1f}% = {(vire_a+riski_a):+.1f}%")
        st.write(f"**Pelaaja B laadullinen:** vire {vire_b:+.1f}% + riski {riski_b:+.1f}% = {(vire_b+riski_b):+.1f}%")
        st.write(f"**Laadullinen säätö (logit):** {kvalit_logit:+.3f}")
        st.write(f"**Puhdas matemaattinen:** A {pure_prob_a*100:.1f}% — B {pure_prob_b*100:.1f}%")
        st.write(f"**Lopullinen (sis. laadulliset):** A {final_prob_a*100:.1f}% — B {final_prob_b*100:.1f}%")
