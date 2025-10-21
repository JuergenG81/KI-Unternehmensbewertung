import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="KI-Unternehmensbewertung (MVP)", layout="wide")

# ---------------------------
# Helper Functions
# ---------------------------
def compute_wacc(cost_of_equity, cost_of_debt, tax_rate, equity_value, debt_value):
    total = max(equity_value + debt_value, 1e-6)
    e_weight = equity_value / total
    d_weight = debt_value / total
    return e_weight * cost_of_equity + d_weight * cost_of_debt * (1 - tax_rate)

def gordon_terminal_value(last_fcff, wacc, g):
    if wacc <= g:
        return np.nan
    return last_fcff * (1 + g) / (wacc - g)

def discount_factor(wacc, t):
    return 1 / ((1 + wacc) ** t)

def fcff_from_drivers(revenue, ebitda_margin, depreciation_ratio, tax_rate, nwc_ratio, capex_ratio):
    ebitda = revenue * ebitda_margin
    depreciation = revenue * depreciation_ratio
    ebit = ebitda - depreciation
    tax = max(0.0, ebit) * tax_rate
    nopat = ebit - tax
    capex = revenue * capex_ratio
    delta_nwc = revenue * nwc_ratio
    fcff = nopat + depreciation - capex - delta_nwc
    return {
        "revenue": revenue,
        "ebitda": ebitda,
        "depreciation": depreciation,
        "ebit": ebit,
        "tax": tax,
        "nopat": nopat,
        "capex": capex,
        "delta_nwc": delta_nwc,
        "fcff": fcff,
    }

def equity_from_ev(ev, debt, cash):
    return ev - debt + cash

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.title("Parameter")

st.sidebar.subheader("Unternehmen")
company_name = st.sidebar.text_input("Name", "Demo GmbH")
country = st.sidebar.text_input("Land", "AT")
industry = st.sidebar.text_input("Branche (ÖNACE/NAICS Freitext)", "Handel")

st.sidebar.subheader("Planung & Treiber")
forecast_years = st.sidebar.number_input("Planungsjahre", 3, 10, 5)
start_revenue = st.sidebar.number_input("Ausgangsumsatz (t-0)", min_value=0.0, value=1_000_000.0, step=10_000.0, format="%.2f")
revenue_growth = st.sidebar.slider("Umsatzwachstum (p.a.)",  -0.50, 1.0, 0.10, step=0.01)
ebitda_margin = st.sidebar.slider("EBITDA-Marge", 0.0, 0.6, 0.18, step=0.01)
depr_ratio = st.sidebar.slider("Abschreibungen / Umsatz", 0.0, 0.2, 0.03, step=0.005)
tax_rate = st.sidebar.slider("Steuersatz", 0.0, 0.6, 0.25, step=0.01)
nwc_ratio = st.sidebar.slider("ΔNWC / Umsatz", -0.2, 0.2, 0.02, step=0.005)
capex_ratio = st.sidebar.slider("CAPEX / Umsatz", 0.0, 0.3, 0.04, step=0.005)

st.sidebar.subheader("Kapitalstruktur")
debt = st.sidebar.number_input("Finanzverbindlichkeiten (Debt)", min_value=0.0, value=300_000.0, step=10_000.0, format="%.2f")
cash = st.sidebar.number_input("Liquide Mittel (Cash)", min_value=0.0, value=100_000.0, step=10_000.0, format="%.2f")

st.sidebar.subheader("Kapitalkosten")
cost_of_equity = st.sidebar.slider("Kosten des Eigenkapitals (Ke)", 0.02, 0.30, 0.12, step=0.005)
cost_of_debt = st.sidebar.slider("Kosten des Fremdkapitals (Kd)", 0.00, 0.20, 0.05, step=0.005)
terminal_growth = st.sidebar.slider("Ewiges Wachstum (g)", -0.05, 0.06, 0.02, step=0.005)

# --- Wiener Verfahren (Sidebar) ---
st.sidebar.subheader("Wiener Verfahren")
use_manual_E = st.sidebar.checkbox("Nachhaltigen Ertrag manuell eingeben", value=False)
manual_E = st.sidebar.number_input("Nachhaltiger Jahresertrag E (vor St.)", min_value=0.0, value=120_000.0, step=1_000.0, format="%.2f")
wien_base = st.sidebar.slider("Basiszinssatz i", 0.00, 0.15, 0.06, step=0.005)
wien_r1 = st.sidebar.slider("Risikozuschlag r₁ (Branche)", 0.00, 0.15, 0.05, step=0.005)
wien_r2 = st.sidebar.slider("Risikozuschlag r₂ (Größe/Stabilität)", 0.00, 0.15, 0.03, step=0.005)
wien_r3 = st.sidebar.slider("Risikozuschlag r₃ (Aussichten/Abhängigkeiten)", 0.00, 0.15, 0.03, step=0.005)

st.sidebar.subheader("Multiplikatoren (optional)")
ev_ebitda_manual = st.sidebar.number_input("EV/EBITDA (Peer Median)", min_value=0.0, value=6.5, step=0.1, format="%.2f")

st.sidebar.subheader("Daten-Upload (optional)")
uploaded = st.sidebar.file_uploader("Historie/Plan (CSV: year,revenue,ebit,depreciation,capex,nwc,cash)", type=["csv"])

ax1.plot(forecast_df["year"], forecast_df["fcff"], marker="o")
ax1.set_xlabel("Jahr")
ax1.set_ylabel("FCFF")
ax1.set_title("Free Cash Flow to Firm (Forecast)")
st.pyplot(fig1)

# ---------------------------
# Multiples
# ---------------------------
st.header("Multiple-Bewertung")
ebitda_base = forecast_df.loc[forecast_df.index[-1], "ebitda"]
ev_multiple = ebitda_base * ev_ebitda_manual if pd.notnull(ebitda_base) else np.nan
equity_multiple = equity_from_ev(ev_multiple, debt, cash)

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("EBITDA (Jahr n)", f"{ebitda_base:,.0f} €")
mcol2.metric("Enterprise Value (Multiple)", f"{ev_multiple:,.0f} €")
mcol3.metric("Equity Value (Multiple)", f"{equity_multiple:,.0f} €")

# ---------------------------
# Wiener Verfahren (Berechnung & Anzeige)
# ---------------------------
st.header("Wiener Verfahren")
wien_rate = wien_base + wien_r1 + wien_r2 + wien_r3

# E bestimmen: manuell ODER konservativ aus Forecast (Durchschnitt EBIT der letzten 3 Jahre, vor Steuern)
if use_manual_E:
    E = manual_E
else:
    try:
        e_last3 = forecast_df.tail(min(3, len(forecast_df)))["ebit"].mean()
        E = max(0.0, float(e_last3))  # vor Steuern, konservativ
    except Exception:
        E = 0.0

if avz_summary is not None and not avz_summary.empty:
    afa_last = float(avz_summary.tail(1)["afa_sum"])
    st.caption(f"Hinweis: AfA laut AVZ im jüngsten Jahr: {afa_last:,.0f} € (Informativ; E bleibt wie bestimmt)")

wien_value = np.nan
if wien_rate > 0:
    wien_value = (E * 100.0) / wien_rate

colw1, colw2, colw3 = st.columns(3)
colw1.metric("Nachhaltiger Ertrag (E, vor St.)", f"{E:,.0f} €")
colw2.metric("Kapitalisierungszinssatz (i + Σr)", f"{wien_rate:.2%}")
colw3.metric("Unternehmenswert (Wiener Verfahren)", f"{wien_value:,.0f} €")

# ---------------------------
# Ergebniszusammenfassung (inkl. Wiener Verfahren)
# ---------------------------
st.header("Ergebniszusammenfassung (inkl. Wiener Verfahren)")
summary_df = pd.DataFrame({
    "Methode": ["DCF", "Multiple", "Wiener Verfahren"],
    "EV": [ev_dcf, ev_multiple, np.nan],  # Wiener ergibt Ertrags-/Equity-Wert, kein EV
    "Equity": [equity_dcf, equity_multiple, wien_value]
})
st.dataframe(summary_df.style.format({"EV":"{:,.0f}", "Equity":"{:,.0f}"}))

# Chart: Equity-Spanne
fig2, ax2 = plt.subplots()
ax2.bar(summary_df["Methode"], summary_df["Equity"])
ax2.set_ylabel("Equity Value")
ax2.set_title("Bewertung nach Methode")
st.pyplot(fig2)

# ---------------------------
# Plausibilisierung (Mittelwert Ertragswert & Substanzwert)
# ---------------------------
if substanzwert is not None and not np.isnan(wien_value):
    mw = (wien_value + substanzwert) / 2.0
    st.info(f"Plausibilisierung (Mittelwert aus Ertragswert & Substanzwert): {mw:,.0f} €")

# ---------------------------
# Sensitivitäten (Tornado Light)
# ---------------------------
st.header("Sensitivität (Einzel-Parameter)")

def recompute_equity(wacc_, growth_, ebitda_margin_):
    rows = []
    revenue = start_revenue
    for t in range(1, int(forecast_years) + 1):
        revenue = revenue * (1 + revenue_growth)
        r = fcff_from_drivers(revenue, ebitda_margin_, depr_ratio, tax_rate, nwc_ratio, capex_ratio)
        r["year"] = t
        r["discount_factor"] = discount_factor(wacc_, t)
        r["pv_fcff"] = r["fcff"] * r["discount_factor"]
        rows.append(r)
    df = pd.DataFrame(rows)
    last_fcff = df.iloc[-1]["fcff"]
    tv = gordon_terminal_value(last_fcff, wacc_, growth_)
    pv_tv = tv * discount_factor(wacc_, int(forecast_years))
    ev = df["pv_fcff"].sum() + pv_tv
    return equity_from_ev(ev, debt, cash)

# +/- 10% Punkte auf EBITDA-Marge
eb_low = recompute_equity(wacc, terminal_growth, max(0.0, ebitda_margin - 0.10))
eb_high = recompute_equity(wacc, terminal_growth, min(0.6, ebitda_margin + 0.10))
# +/- 100 bp auf WACC
wacc_low = recompute_equity(max(0.001, wacc - 0.01), terminal_growth, ebitda_margin)
wacc_high = recompute_equity(wacc + 0.01, terminal_growth, ebitda_margin)
# +/- 50 bp auf g
g_low = recompute_equity(wacc, terminal_growth - 0.005, ebitda_margin)

"Historie/Plan (CSV: year,revenue,ebitda,ebit,depreciation,capex,nwc,debt,cash)",
    type=["csv"]
)

# ---------------------------
# Header & Intro
# ---------------------------
st.title("KI-Unternehmensbewertung — MVP")
st.caption("DCF + Multiples + Wiener Verfahren (mit AVZ-Upload) — inkl. Sensitivitäten und Report.")

# ---------------------------
# AVZ-Upload (optional)
# ---------------------------
st.header("AVZ-Upload (optional) – Investitionen, AfA & Substanzwert")
avz_file = st.file_uploader(
    "Anlagenverzeichnis (CSV: jahr,wirtschaftsgut,zugang,afa,abgang,restbuchwert)",
    type=["csv"],
    key="avz"
)

avz_df = None
avz_summary = None
substanzwert = None

if avz_file is not None:
    try:
        avz_df = pd.read_csv(avz_file)
        required_cols = {"jahr","wirtschaftsgut","zugang","afa","abgang","restbuchwert"}
        avz_df.columns = [c.lower() for c in avz_df.columns]
        if not required_cols.issubset(set(avz_df.columns)):
            st.error("CSV-Spalten erwartet: jahr,wirtschaftsgut,zugang,afa,abgang,restbuchwert")
        else:
            for c in ["zugang","afa","abgang","restbuchwert"]:
                avz_df[c] = pd.to_numeric(avz_df[c], errors="coerce").fillna(0.0)

            avz_summary = avz_df.groupby("jahr").agg(
                capex_avz=("zugang","sum"),
                afa_sum=("afa","sum"),
                abgang_sum=("abgang","sum"),
                restwert_sum=("restbuchwert","sum")
            ).reset_index().sort_values("jahr")

            st.success("AVZ erfolgreich geladen und aggregiert.")
            st.dataframe(avz_summary.style.format({
                "capex_avz":"{:,.0f}",
                "afa_sum":"{:,.0f}",
                "restwert_sum":"{:,.0f}"
            }))

            # Substanzwert-Proxy: Restbuchwerte letztes Jahr
            substanzwert = float(avz_summary.tail(1)["restwert_sum"]) if not avz_summary.empty else None
            if substanzwert is not None:
                st.info(f"Substanzwert (Restbuchwerte letztes Jahr): {substanzwert:,.0f} €")
    except Exception as e:
        st.error(f"AVZ konnte nicht gelesen werden: {e}")

# ---------------------------
# DCF-Bewertung (Forecast & EV/Equity)
# ---------------------------
st.header("DCF-Bewertung")

# Equity/ Debt placeholders für WACC (vereinfachter Start: Equity ~ start_revenue)
equity_guess = max(start_revenue, 1.0)
wacc = compute_wacc(cost_of_equity, cost_of_debt, tax_rate, equity_guess, debt)

rows = []
revenue = start_revenue
for t in range(1, int(forecast_years) + 1):
    revenue = revenue * (1 + revenue_growth)
    row = fcff_from_drivers(
        revenue=revenue,
        ebitda_margin=ebitda_margin,
        depreciation_ratio=depr_ratio,
        tax_rate=tax_rate,
        nwc_ratio=nwc_ratio,
        capex_ratio=capex_ratio,
    )
    row["year"] = t
    row["discount_factor"] = discount_factor(wacc, t)
    row["pv_fcff"] = row["fcff"] * row["discount_factor"]
    rows.append(row)

forecast_df = pd.DataFrame(
    rows,
    columns=[
        "year","revenue","ebitda","depreciation","ebit","tax",
        "nopat","capex","delta_nwc","fcff","discount_factor","pv_fcff"
    ]
)

last_fcff = forecast_df.loc[forecast_df.index[-1], "fcff"]
tv = gordon_terminal_value(last_fcff, wacc, terminal_growth)
pv_tv = tv * discount_factor(wacc, int(forecast_years))

pv_fcff_sum = forecast_df["pv_fcff"].sum()
ev_dcf = pv_fcff_sum + pv_tv
equity_dcf = equity_from_ev(ev_dcf, debt, cash)

col1, col2, col3 = st.columns(3)
col1.metric("WACC", f"{wacc:.2%}")
col2.metric("Enterprise Value (DCF)", f"{ev_dcf:,.0f} €")
col3.metric("Equity Value (DCF)", f"{equity_dcf:,.0f} €")

st.subheader("Prognose & Barwerte (FCFF)")
st.dataframe(forecast_df.style.format({
    "revenue":"{:,.0f}",
    "ebitda":"{:,.0f}",
    "depreciation":"{:,.0f}",
    "ebit":"{:,.0f}",
    "tax":"{:,.0f}",
    "nopat":"{:,.0f}",
    "capex":"{:,.0f}",
    "delta_nwc":"{:,.0f}",
    "fcff":"{:,.0f}",
    "discount_factor":"{:.3f}",
    "pv_fcff":"{:,.0f}",
}))

# Chart: FCFF Verlauf
st.subheader("FCFF Verlauf")
fig1, ax1 = plt.subplots()

ax1.plot(forecast_df["year"], forecast_df["fcff"], marker="o")
ax1.set_xlabel("Jahr")
ax1.set_ylabel("FCFF")
ax1.set_title("Free Cash Flow to Firm (Forecast)")
st.pyplot(fig1)

# ---------------------------
# Multiples
# ---------------------------
st.header("Multiple-Bewertung")
ebitda_base = forecast_df.loc[forecast_df.index[-1], "ebitda"]
ev_multiple = ebitda_base * ev_ebitda_manual if pd.notnull(ebitda_base) else np.nan
equity_multiple = equity_from_ev(ev_multiple, debt, cash)

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("EBITDA (Jahr n)", f"{ebitda_base:,.0f} €")
mcol2.metric("Enterprise Value (Multiple)", f"{ev_multiple:,.0f} €")
mcol3.metric("Equity Value (Multiple)", f"{equity_multiple:,.0f} €")

# ---------------------------
# Wiener Verfahren (Berechnung & Anzeige)
# ---------------------------
st.header("Wiener Verfahren")
wien_rate = wien_base + wien_r1 + wien_r2 + wien_r3

# E bestimmen: manuell ODER konservativ aus Forecast (Durchschnitt EBIT der letzten 3 Jahre, vor Steuern)
if use_manual_E:
    E = manual_E
else:
    try:
        e_last3 = forecast_df.tail(min(3, len(forecast_df)))["ebit"].mean()
        E = max(0.0, float(e_last3))  # vor Steuern, konservativ
    except Exception:
        E = 0.0

if avz_summary is not None and not avz_summary.empty:
    afa_last = float(avz_summary.tail(1)["afa_sum"])
    st.caption(f"Hinweis: AfA laut AVZ im jüngsten Jahr: {afa_last:,.0f} € (Informativ; E bleibt wie bestimmt)")

wien_value = np.nan
if wien_rate > 0:
    wien_value = (E * 100.0) / wien_rate

colw1, colw2, colw3 = st.columns(3)
colw1.metric("Nachhaltiger Ertrag (E, vor St.)", f"{E:,.0f} €")
colw2.metric("Kapitalisierungszinssatz (i + Σr)", f"{wien_rate:.2%}")
colw3.metric("Unternehmenswert (Wiener Verfahren)", f"{wien_value:,.0f} €")

# ---------------------------
# Ergebniszusammenfassung (inkl. Wiener Verfahren)
# ---------------------------
st.header("Ergebniszusammenfassung (inkl. Wiener Verfahren)")
summary_df = pd.DataFrame({
    "Methode": ["DCF", "Multiple", "Wiener Verfahren"],
    "EV": [ev_dcf, ev_multiple, np.nan],  # Wiener ergibt Ertrags-/Equity-Wert, kein EV
    "Equity": [equity_dcf, equity_multiple, wien_value]
})
st.dataframe(summary_df.style.format({"EV":"{:,.0f}", "Equity":"{:,.0f}"}))

# Chart: Equity-Spanne
fig2, ax2 = plt.subplots()
ax2.bar(summary_df["Methode"], summary_df["Equity"])
ax2.set_ylabel("Equity Value")
ax2.set_title("Bewertung nach Methode")
st.pyplot(fig2)

# ---------------------------
# Plausibilisierung (Mittelwert Ertragswert & Substanzwert)
# ---------------------------
if substanzwert is not None and not np.isnan(wien_value):
    mw = (wien_value + substanzwert) / 2.0
    st.info(f"Plausibilisierung (Mittelwert aus Ertragswert & Substanzwert): {mw:,.0f} €")

# ---------------------------
# Sensitivitäten (Tornado Light)
# ---------------------------
st.header("Sensitivität (Einzel-Parameter)")

def recompute_equity(wacc_, growth_, ebitda_margin_):
    rows = []
    revenue = start_revenue
    for t in range(1, int(forecast_years) + 1):
        revenue = revenue * (1 + revenue_growth)
        r = fcff_from_drivers(revenue, ebitda_margin_, depr_ratio, tax_rate, nwc_ratio, capex_ratio)
        r["year"] = t
        r["discount_factor"] = discount_factor(wacc_, t)
        r["pv_fcff"] = r["fcff"] * r["discount_factor"]
        rows.append(r)
    df = pd.DataFrame(rows)
    last_fcff = df.iloc[-1]["fcff"]
    tv = gordon_terminal_value(last_fcff, wacc_, growth_)
    pv_tv = tv * discount_factor(wacc_, int(forecast_years))
    ev = df["pv_fcff"].sum() + pv_tv
    return equity_from_ev(ev, debt, cash)

# +/- 10% Punkte auf EBITDA-Marge
eb_low = recompute_equity(wacc, terminal_growth, max(0.0, ebitda_margin - 0.10))
eb_high = recompute_equity(wacc, terminal_growth, min(0.6, ebitda_margin + 0.10))
# +/- 100 bp auf WACC
wacc_low = recompute_equity(max(0.001, wacc - 0.01), terminal_growth, ebitda_margin)
wacc_high = recompute_equity(wacc + 0.01, terminal_growth, ebitda_margin)
# +/- 50 bp auf g
g_low = recompute_equity(wacc, terminal_growth - 0.005, ebitda_margin)

"Historie/Plan (CSV: year,revenue,ebitda,ebit,depreciation,capex,nwc,debt,cash)",
    type=["csv"]

# ---------------------------
# Header & Intro
# ---------------------------
st.title("KI-Unternehmensbewertung — MVP")
st.caption("DCF + Multiples + Wiener Verfahren (mit AVZ-Upload) — inkl. Sensitivitäten und Report.")

# ---------------------------
# AVZ-Upload (optional)
# ---------------------------
st.header("AVZ-Upload (optional) – Investitionen, AfA & Substanzwert")
avz_file = st.file_uploader(
    "Anlagenverzeichnis (CSV: jahr,wirtschaftsgut,zugang,afa,abgang,restbuchwert)",
    type=["csv"],
    key="avz"
)

avz_df = None
avz_summary = None
substanzwert = None

if avz_file is not None:
    try:
        avz_df = pd.read_csv(avz_file)
        required_cols = {"jahr","wirtschaftsgut","zugang","afa","abgang","restbuchwert"}
        avz_df.columns = [c.lower() for c in avz_df.columns]
        if not required_cols.issubset(set(avz_df.columns)):
            st.error("CSV-Spalten erwartet: jahr,wirtschaftsgut,zugang,afa,abgang,restbuchwert")
        else:
            for c in ["zugang","afa","abgang","restbuchwert"]:
                avz_df[c] = pd.to_numeric(avz_df[c], errors="coerce").fillna(0.0)

            avz_summary = avz_df.groupby("jahr").agg(
                capex_avz=("zugang","sum"),
                afa_sum=("afa","sum"),
                abgang_sum=("abgang","sum"),
                restwert_sum=("restbuchwert","sum")
            ).reset_index().sort_values("jahr")

            st.success("AVZ erfolgreich geladen und aggregiert.")
            st.dataframe(avz_summary.style.format({
                "capex_avz":"{:,.0f}",
                "afa_sum":"{:,.0f}",
                "restwert_sum":"{:,.0f}"
            }))

            # Substanzwert-Proxy: Restbuchwerte letztes Jahr
            substanzwert = float(avz_summary.tail(1)["restwert_sum"]) if not avz_summary.empty else None
            if substanzwert is not None:
                st.info(f"Substanzwert (Restbuchwerte letztes Jahr): {substanzwert:,.0f} €")
    except Exception as e:
        st.error(f"AVZ konnte nicht gelesen werden: {e}")

# ---------------------------
# DCF-Bewertung (Forecast & EV/Equity)
# ---------------------------
st.header("DCF-Bewertung")

# Equity/ Debt placeholders für WACC (vereinfachter Start: Equity ~ start_revenue)
equity_guess = max(start_revenue, 1.0)
wacc = compute_wacc(cost_of_equity, cost_of_debt, tax_rate, equity_guess, debt)

rows = []
revenue = start_revenue
for t in range(1, int(forecast_years) + 1):
    revenue = revenue * (1 + revenue_growth)
    row = fcff_from_drivers(
        revenue=revenue,
        ebitda_margin=ebitda_margin,
        depreciation_ratio=depr_ratio,
        tax_rate=tax_rate,
        nwc_ratio=nwc_ratio,
        capex_ratio=capex_ratio,
    )
    row["year"] = t
    row["discount_factor"] = discount_factor(wacc, t)
    row["pv_fcff"] = row["fcff"] * row["discount_factor"]
    rows.append(row)

forecast_df = pd.DataFrame(
    rows,
    columns=[
        "year","revenue","ebitda","depreciation","ebit","tax",
        "nopat","capex","delta_nwc","fcff","discount_factor","pv_fcff"
    ]
)

last_fcff = forecast_df.loc[forecast_df.index[-1], "fcff"]
tv = gordon_terminal_value(last_fcff, wacc, terminal_growth)
pv_tv = tv * discount_factor(wacc, int(forecast_years))

pv_fcff_sum = forecast_df["pv_fcff"].sum()
ev_dcf = pv_fcff_sum + pv_tv
equity_dcf = equity_from_ev(ev_dcf, debt, cash)

col1, col2, col3 = st.columns(3)
col1.metric("WACC", f"{wacc:.2%}")
col2.metric("Enterprise Value (DCF)", f"{ev_dcf:,.0f} €")
col3.metric("Equity Value (DCF)", f"{equity_dcf:,.0f} €")

st.subheader("Prognose & Barwerte (FCFF)")
st.dataframe(forecast_df.style.format({
    "revenue":"{:,.0f}",
    "ebitda":"{:,.0f}",
    "depreciation":"{:,.0f}",
    "ebit":"{:,.0f}",
    "tax":"{:,.0f}",
    "nopat":"{:,.0f}",
    "capex":"{:,.0f}",
    "delta_nwc":"{:,.0f}",
    "fcff":"{:,.0f}",
    "discount_factor":"{:.3f}",
    "pv_fcff":"{:,.0f}",
}))

# Chart: FCFF Verlauf
st.subheader("FCFF Verlauf")
fig1, ax1 = plt.subplots()

g_high = recompute_equity(wacc, terminal_growth + 0.005, ebitda_margin)

sens_df = pd.DataFrame({
    "Parameter": ["EBITDA-Marge ↓", "EBITDA-Marge ↑", "WACC ↓", "WACC ↑", "g ↓", "g ↑"],
    "Equity": [eb_low, eb_high, wacc_low, wacc_high, g_low, g_high]
})
st.dataframe(sens_df.style.format({"Equity":"{:,.0f}"}))

fig3, ax3 = plt.subplots()
ax3.bar(sens_df["Parameter"], sens_df["Equity"])
ax3.set_title("Sensitivität (Änderung Einzel-Parameter)")
ax3.set_ylabel("Equity Value")
plt.xticks(rotation=20)
st.pyplot(fig3)

# ---------------------------
# Report Export (HTML)
# ---------------------------
st.header("Report")
report_lines = []
report_lines.append(f"<h1>Bewertungsreport — {company_name}</h1>")
report_lines.append(f"<p><b>Datum:</b> {date.today().isoformat()} | <b>Land:</b> {country} | <b>Branche:</b> {industry}</p>")
report_lines.append("<h2>Parameter</h2>")
report_lines.append(f"<ul>"
                    f"<li>WACC: {wacc:.2%}</li>"
                    f"<li>g: {terminal_growth:.2%}</li>"
                    f"<li>Ke: {cost_of_equity:.2%}</li>"
                    f"<li>Kd: {cost_of_debt:.2%}</li>"
                    f"<li>Debt: {debt:,.0f} €</li>"
                    f"<li>Cash: {cash:,.0f} €</li>"
                    f"</ul>")
report_lines.append("<h2>Ergebnisse (Zusammenfassung)</h2>")
report_lines.append(summary_df.to_html(index=False))

# Wiener-Verfahren im Report
report_lines.append("<h2>Wiener Verfahren</h2>")
report_lines.append(f"<p>E (nachhaltiger Ertrag, vor St.): {E:,.0f} €<br>"
                    f"i + Σr: {wien_rate:.2%}<br>"
                    f"Wert: {wien_value:,.0f} €</p>")
if substanzwert is not None and not np.isnan(wien_value):
    mw = (wien_value + substanzwert) / 2.0
    report_lines.append(f"<p>Plausibilisierung (Mittelwert mit Substanzwert {substanzwert:,.0f} €): {mw:,.0f} €</p>")

# Forecast-Tabelle im Report
report_lines.append("<h2>Forecast (FCFF)</h2>")
report_lines.append(forecast_df.to_html(index=False))

html_report = "\n".join(report_lines)
st.download_button(
    label="Report als HTML herunterladen",
    data=html_report,
    file_name=f"Bewertungsreport_{company_name.replace(' ', '_')}.html",
    mime="text/html"
)

st.caption("MVP — Keine Anlageberatung. Ergebnisse sind modell- und annahmenabhängig.")
