import gspread
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from oauth2client.service_account import ServiceAccountCredentials
import json
from datetime import datetime

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Dashboard Portefeuille", layout="wide")
st.markdown("<h1 style='text-align: left; font-size: 30px;'>📊 Dashboard Portefeuille - FBM</h1>", unsafe_allow_html=True)

# -----------------------
# Google Sheets config via Streamlit Secrets
# -----------------------
SHEET_NAME = "transactions_dashboard"   # Nom de ta Google Sheet
SCOPE = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

# ⚠️ Utilisation des secrets Streamlit Cloud
creds_dict = st.secrets["google_service_account"]
CREDS = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
client = gspread.authorize(CREDS)
sheet = client.open(SHEET_NAME).sheet1

EXPECTED_COLS = ["Date","Type","Ticker","Quantité","Prix","PnL réalisé (€/$)","PnL réalisé (%)"]

# -----------------------
# Helpers
# -----------------------
@st.cache_data(ttl=60)
def fetch_last_close(tickers):
    if not tickers:
        return {}
    tickers = list(set(tickers))
    try:
        data = yf.download(tickers, period="5d", progress=False, threads=True)
        closes = {}
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            close_df = data['Close']
            for tk in tickers:
                if tk in close_df.columns:
                    series = close_df[tk].dropna()
                    closes[tk] = float(series.iloc[-1]) if not series.empty else None
                else:
                    closes[tk] = None
        else:
            last = data['Close'].dropna()
            for tk in tickers:
                closes[tk] = float(last.iloc[-1]) if not last.empty else None
        return closes
    except Exception:
        closes = {}
        for tk in tickers:
            try:
                t = yf.Ticker(tk)
                hist = t.history(period="5d")
                if not hist.empty:
                    closes[tk] = float(hist['Close'].iloc[-1])
                else:
                    closes[tk] = None
            except:
                closes[tk] = None
        return closes

def load_transactions():
    try:
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None
        # Convert Date column to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Erreur lecture Google Sheet: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def save_transactions(df):
    try:
        sheet.clear()
        sheet.append_row(EXPECTED_COLS)
        for row in df.values.tolist():
            # Convert datetime to ISO string pour Google Sheets
            row_copy = [r.isoformat() if isinstance(r, (datetime, pd.Timestamp)) else r for r in row]
            sheet.append_row(row_copy)
    except Exception as e:
        st.error(f"Erreur écriture Google Sheet: {e}")

def format_pct(val):
    if pd.isna(val):
        return ""
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"

# -----------------------
# State init
# -----------------------
if "transactions" not in st.session_state:
    st.session_state.transactions = load_transactions().to_dict('records')

# -----------------------
# Onglets
# -----------------------
tab1, tab2 = st.tabs(["💰 Transactions", "📂 Portefeuille"])

# -----------------------
# Onglet 1 : Saisie Transactions
# -----------------------
with tab1:
    st.header("Ajouter une transaction")

    type_tx = st.selectbox("Type", ["Achat", "Vente", "Dépot €"])

    tickers_existants = sorted(set([tx["Ticker"] for tx in st.session_state.transactions if tx["Ticker"] and tx["Ticker"] != "CASH"]))

    ticker = st.selectbox(
        "Ticker (ex: AAPL, BTC-USD)", 
        options=[""] + tickers_existants,
        index=0,
        format_func=lambda x: x if x != "" else ""
    )

    if ticker == "":
        ticker = st.text_input("Nouveau ticker (ex: AAPL)").upper().strip()

    quantite = st.number_input("Quantité", min_value=0.0001, step=0.0001, format="%.4f")
    prix = st.number_input("Prix (€/$)", min_value=0.0, step=0.01, format="%.2f")

    date_input = st.date_input("Date de transaction")
    
    if st.button("➕ Ajouter Transaction"):
        try:
            date_tx = pd.to_datetime(date_input)
            st.success(f"Date enregistrée : {date_tx}")
        except Exception as e:
            st.error(f"Format de date invalide ({e}). Utilisation de la date actuelle.")
            date_tx = datetime.now()

        df_hist = pd.DataFrame(st.session_state.transactions)
        if df_hist.empty:
            df_hist = pd.DataFrame(columns=EXPECTED_COLS)

        transaction = None

        if type_tx == "Dépot €":
            transaction = {
                "Date": date_input,
                "Type": "Dépot €",
                "Ticker": "CASH",
                "Quantité": 1,
                "Prix": round(prix,2),
                "PnL réalisé (€/$)": 0.0,
                "PnL réalisé (%)": 0.0
            }
        else:
            if not ticker:
                st.error("Ticker requis pour Achat/Vente.")
            elif quantite <= 0 or prix <= 0:
                st.error("Quantité et prix doivent être > 0.")
            else:
                ticker = ticker.upper()
                if type_tx == "Achat":
                    transaction = {
                        "Date": date_input,
                        "Type": "Achat",
                        "Ticker": ticker,
                        "Quantité": quantite,
                        "Prix": round(prix,2),
                        "PnL réalisé (€/$)": 0.0,
                        "PnL réalisé (%)": 0.0
                    }
                elif type_tx == "Vente":
                    df_pos = df_hist[df_hist["Ticker"] == ticker]
                    qty_pos = df_pos["Quantité"].sum() if not df_pos.empty else 0
                    if qty_pos < quantite:
                        st.error("❌ Pas assez de titres pour vendre.")
                    else:
                        achats = df_pos[df_pos["Quantité"] > 0]
                        total_qty_achats = achats["Quantité"].sum() if not achats.empty else 0
                        prix_moyen = (achats["Quantité"] * achats["Prix"]).sum() / total_qty_achats if total_qty_achats > 0 else 0
                        pnl_real = (prix - prix_moyen) * quantite
                        pnl_pct = ((prix - prix_moyen) / prix_moyen * 100) if prix_moyen != 0 else 0
                        transaction = {
                            "Date": date_input,
                            "Type": "Vente",
                            "Ticker": ticker,
                            "Quantité": -abs(quantite),
                            "Prix": round(prix,2),
                            "PnL réalisé (€/$)": round(pnl_real,2),
                            "PnL réalisé (%)": round(pnl_pct,2)
                        }

        if transaction:
            st.session_state.transactions.append(transaction)
            df_save = pd.DataFrame(st.session_state.transactions)
            save_transactions(df_save)
            st.success(f"{type_tx} enregistré : {transaction['Ticker']}")

    st.subheader("Historique des transactions")
    if st.session_state.transactions:
        df_tx = pd.DataFrame(st.session_state.transactions)
        df_tx["Date"] = pd.to_datetime(df_tx["Date"], errors="coerce")
        st.dataframe(
            df_tx.sort_values(by="Date", ascending=False).reset_index(drop=True).head(200),
            use_container_width=True
        )
    else:
        st.info("Aucune transaction enregistrée.")

# -----------------------
# Onglet 2 : Portefeuille consolidé
# -----------------------
with tab2:
    st.header("Portefeuille consolidé")

    if not st.session_state.transactions:
        st.info("Aucune transaction pour calculer le portefeuille.")
    else:
        df_tx = pd.DataFrame(st.session_state.transactions)
        df_tx["Date"] = pd.to_datetime(df_tx["Date"], errors="coerce")
        df_tx = df_tx.sort_values(by="Date")

        total_depots = df_tx[df_tx["Type"] == "Dépot €"]["Prix"].sum() if not df_tx[df_tx["Type"] == "Dépot €"].empty else 0.0
        total_achats = (df_tx[df_tx["Type"] == "Achat"]["Quantité"] * df_tx[df_tx["Type"] == "Achat"]["Prix"]).sum() if not df_tx[df_tx["Type"] == "Achat"].empty else 0.0
        ventes = df_tx[df_tx["Type"] == "Vente"]
        total_ventes = ((-ventes["Quantité"]) * ventes["Prix"]).sum() if not ventes.empty else 0.0

        cash = total_depots + total_ventes - total_achats

        df_actifs = df_tx[df_tx["Ticker"] != "CASH"]
        portefeuille = pd.DataFrame()
        if not df_actifs.empty:
            grp = df_actifs.groupby("Ticker", as_index=False).agg({"Quantité": "sum"})
            prix_moy = {}
            for tk in grp["Ticker"]:
                df_tk = df_actifs[df_actifs["Ticker"] == tk]
                achats = df_tk[df_tk["Quantité"] > 0]
                prix_moy[tk] = (achats["Quantité"] * achats["Prix"]).sum() / achats["Quantité"].sum() if not achats.empty else 0.0

            tickers = [t for t in grp["Ticker"] if grp.loc[grp["Ticker"]==t, "Quantité"].values[0] != 0]
            closes = fetch_last_close(tickers)

            rows = []
            for t in tickers:
                qty = grp.loc[grp["Ticker"]==t, "Quantité"].values[0]
                avg_cost = prix_moy.get(t, 0.0)
                current = closes.get(t, None)
                if current is None:
                    valeur, pnl_abs, pnl_pct = None, None, None
                else:
                    valeur = current * qty
                    pnl_abs = (current - avg_cost) * qty
                    pnl_pct = ((current - avg_cost)/avg_cost*100) if avg_cost != 0 else None
                rows.append({
                    "Ticker": t,
                    "Quantité nette": qty,
                    "Prix moyen pondéré": round(avg_cost,2),
                    "Prix actuel": round(current,2) if current is not None else None,
                    "Valeur totale": round(valeur,2) if valeur is not None else None,
                    "PnL latent (€/$)": round(pnl_abs,2) if pnl_abs is not None else None,
                    "PnL latent (%)": round(pnl_pct,2) if pnl_pct is not None else None
                })

            portefeuille = pd.DataFrame(rows)

        total_valeur_actifs = portefeuille["Valeur totale"].sum() if not portefeuille.empty else 0.0
        total_pnl_latent = portefeuille["PnL latent (€/$)"].sum() if not portefeuille.empty else 0.0
        total_pnl_real = df_tx["PnL réalisé (€/$)"].sum() if not df_tx.empty else 0.0
        total_patrimoine = cash + total_valeur_actifs

        k1, k2, k3, k4 = st.columns(4)
        pct_cash = (total_valeur_actifs != 0 and cash / total_valeur_actifs * 100 or 0)
        pct_valeur = (total_valeur_actifs != 0 and total_pnl_latent / total_valeur_actifs * 100 or 0)
        pct_pnl_latent = (total_valeur_actifs != 0 and total_pnl_latent / total_valeur_actifs * 100 or 0)
        pct_pnl_real = (total_valeur_actifs != 0 and total_pnl_real / total_valeur_actifs * 100 or 0)

        k1.metric("💵 Liquidités (est.)", f"{cash:,.2f} €", delta=f"{pct_cash:.2f}%")
        k2.metric("📊 Valeur Actifs", f"{total_valeur_actifs:,.2f} €", delta=f"{pct_valeur:.2f}%")
        k3.metric("📈 PnL Latent", f"{total_pnl_latent:,.2f} €", delta=f"{pct_pnl_latent:.2f}%")
        k4.metric("✅ PnL Réalisé Total", f"{total_pnl_real:,.2f} €", delta=f"{pct_pnl_real:.2f}%")

        st.write(f"Total dépôts : {total_depots:,.2f} € — Total achats : {total_achats:,.2f} € — Total ventes : {total_ventes:,.2f} €")

        if not portefeuille.empty:
            st.dataframe(portefeuille.sort_values(by="Valeur totale", ascending=False).reset_index(drop=True), use_container_width=True)
            try:
                fig = px.pie(portefeuille.dropna(subset=["Valeur totale"]), values="Valeur totale", names="Ticker", title="Répartition du portefeuille")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Impossible d'afficher la répartition (données manquantes).")

            if portefeuille["PnL latent (€/$)"].notna().any():
                fig2 = px.bar(portefeuille.dropna(subset=["PnL latent (€/$)"]), x="Ticker", y="PnL latent (€/$)", text="PnL latent (%)", title="PnL latent par ticker")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune position ouverte (hors CASH).")

        df_ventes = df_tx[df_tx["Type"] == "Vente"].copy()
        if not df_ventes.empty:
            df_ventes = df_ventes.sort_values(by="Date")
            df_ventes["Cumul PnL réalisé"] = df_ventes["PnL réalisé (€/$)"].cumsum()
            fig3 = px.line(df_ventes, x="Date", y="Cumul PnL réalisé", title="PnL Réalisé Cumulatif")
            st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Voir transactions détaillées"):
            st.dataframe(df_tx.reset_index(drop=True), use_container_width=True)
