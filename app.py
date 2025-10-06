import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date
from google.oauth2.service_account import Credentials
from streamlit_elements import elements, mui

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Dashboard Portefeuille", layout="wide")
st.markdown("<h1 style='text-align: left; font-size: 30px;'>📊 Dashboard Portefeuille - FBM</h1>", unsafe_allow_html=True)

SHEET_NAME = "transactions_dashboard"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# -----------------------
# Google Sheets Auth (robuste)
# -----------------------
creds_info = None
sheet = None
try:
    creds_info = st.secrets["google_service_account"]
    credentials = Credentials.from_service_account_info(creds_info, scopes=SCOPE)
    client = gspread.authorize(credentials)
    sheet = client.open(SHEET_NAME).sheet1
except Exception as e:
    st.error("Erreur d'authentification Google Sheets — vérifie st.secrets et le partage de la feuille.")
    st.exception(e)
    sheet = None

# Colonnes attendues
EXPECTED_COLS = ["Date","Profil","Type","Ticker","Quantité","Prix","PnL réalisé (€/$)","PnL réalisé (%)", "Frais (€/$)"]

# -----------------------
# Helpers
# -----------------------
@st.cache_data(ttl=60)
def fetch_last_close(tickers):
    closes = {}
    if not tickers:
        return closes
    tickers = list(set(tickers))
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            hist = t.history(period="7d")
            if not hist.empty and 'Close' in hist.columns:
                closes[tk] = float(hist['Close'].dropna().iloc[-1])
            else:
                closes[tk] = None
        except Exception:
            closes[tk] = None
    return closes

def parse_float(val):
    if val is None or val == "":
        return 0.0
    try:
        return float(str(val).replace(",", "."))
    except ValueError:
        return 0.0

def load_transactions():
    if sheet is None:
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        # Ensure expected cols exist
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None

        # Parse dates (Google Sheets may store as datetime or string)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Normalize numeric columns
        num_cols = ["Quantité", "Prix", "Frais (€/$)", "PnL réalisé (€/$)", "PnL réalisé (%)"]
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False).replace(["", "None", "nan"], "0").astype(float)

        return df
    except Exception as e:
        st.error(f"Erreur lecture Google Sheet: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def save_transactions(df):
    """
    Sauvegarde en Google Sheet.
    On écrit les dates en ISO (YYYY-MM-DD) et on utilise value_input_option='USER_ENTERED'
    pour que Google Sheets les interprète comme dates (cell format date).
    """
    if sheet is None:
        st.error("Pas de connexion à Google Sheets : impossible d'enregistrer.")
        return
    try:
        # Prepare rows with header
        rows = []
        for _, r in df.iterrows():
            row = []
            for c in EXPECTED_COLS:
                val = r.get(c, "")
                if pd.isna(val) or val is None:
                    row.append("")
                elif isinstance(val, (pd.Timestamp, datetime, date)):
                    # Write ISO date (Sheets will convert with USER_ENTERED)
                    row.append(val.strftime("%Y-%m-%d"))
                elif isinstance(val, (float, int)):
                    # Write numeric as plain string (USER_ENTERED will parse floats)
                    row.append(str(val))
                else:
                    row.append(str(val))
            rows.append(row)

        # Clear and write (header + rows) using USER_ENTERED so dates become date cells
        sheet.clear()
        sheet.append_row(EXPECTED_COLS, value_input_option='USER_ENTERED')
        if rows:
            sheet.append_rows(rows, value_input_option='USER_ENTERED')
    except Exception as e:
        st.error(f"Erreur écriture Google Sheet: {e}")

def format_pct(val):
    if pd.isna(val) or val is None:
        return ""
    try:
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.2f}%"
    except Exception:
        return ""

# -----------------------
# State init
# -----------------------
if "transactions" not in st.session_state:
    df = load_transactions()
    st.session_state.transactions = df.to_dict('records')

# -----------------------
# Onglets
# -----------------------
tab1, tab2, tab3 = st.tabs(["💰 Transactions", "📂 Portefeuille", "📊 Répartition"])

# -----------------------
# Onglet 1 : Saisie Transactions avec Typeahead Yahoo Finance
# -----------------------
with tab1:
    st.header("Ajouter une transaction")

    profil = st.selectbox("Portefeuille / Profil", ["Gas", "Marc"])
    type_tx = st.selectbox("Type", ["Achat", "Vente", "Dépot €"])

    st.markdown("### 🔍 Recherche de titre (Ticker ou Nom d’entreprise)")

    # Session vars pour autocomplete
    if "ticker_query" not in st.session_state:
        st.session_state.ticker_query = ""
    if "ticker_suggestions" not in st.session_state:
        st.session_state.ticker_suggestions = []
    if "ticker_selected" not in st.session_state:
        st.session_state.ticker_selected = ""

    def get_yf_suggestions(query: str):
        """Récupère les tickers correspondant à la recherche.
        yfinance propose `yf.search` dans certaines versions ; en cas d'échec on renvoie [].
        """
        try:
            res = yf.search(query)
            if not res or "quotes" not in res:
                return []
            suggestions = []
            for item in res["quotes"]:
                symbol = item.get("symbol")
                name = item.get("shortname", item.get("longname", ""))
                exch = item.get("exchange", "")
                if symbol and name:
                    suggestions.append(f"{symbol} — {name} ({exch})")
            return suggestions[:15]
        except Exception:
            return []

    # Autocomplete React via streamlit-elements
    with elements("ticker_autocomplete"):
        def on_input_change(event, value):
            st.session_state.ticker_query = value or ""
            if len(st.session_state.ticker_query) >= 2:
                st.session_state.ticker_suggestions = get_yf_suggestions(st.session_state.ticker_query)
            else:
                st.session_state.ticker_suggestions = []

        def on_select(event, value):
            st.session_state.ticker_selected = value.split(" — ")[0] if value else ""

        mui.Autocomplete(
            options=st.session_state.ticker_suggestions,
            value=st.session_state.ticker_selected,
            onChange=on_select,
            onInputChange=on_input_change,
            renderInput=lambda params: mui.TextField(
                **params,
                label="Rechercher un titre (ex : AAPL, TSLA, FP.PA)",
                variant="outlined",
                fullWidth=True
            ),
            sx={"width": "100%"},
        )

    if st.session_state.ticker_selected:
        st.success(f"✅ Ticker sélectionné : {st.session_state.ticker_selected}")
    ticker_selected = st.session_state.ticker_selected or None

    # Saisie transaction
    quantite_input = st.text_input("Quantité (pour Achat/Vente) ou laisser 0 pour Dépot", "0")
    prix_input = st.text_input("Prix (€/$) — pour dépôt saisir le montant du dépôt ici", "0")
    frais_input = st.text_input("Frais (€/$)", "0")
    date_input = st.date_input("Date de transaction", value=datetime.today())

    if st.button("➕ Ajouter Transaction"):
        quantite = parse_float(quantite_input)
        prix = parse_float(prix_input)
        frais = parse_float(frais_input)

        # Validation
        if type_tx in ("Achat", "Vente") and not ticker_selected:
            st.error("Ticker requis pour Achat/Vente.")
        elif type_tx == "Dépot €" and prix <= 0:
            st.error("Montant du dépôt doit être > 0.")
        elif type_tx in ("Achat","Vente") and (quantite <= 0 or prix <= 0):
            st.error("Quantité et prix doivent être > 0 pour Achat/Vente.")
        else:
            df_hist = pd.DataFrame(st.session_state.transactions) if st.session_state.transactions else pd.DataFrame(columns=EXPECTED_COLS)
            if "Profil" not in df_hist.columns:
                df_hist["Profil"] = "Gas"

            try:
                date_tx = pd.to_datetime(date_input)
            except Exception:
                date_tx = pd.Timestamp.now()

            ticker = ticker_selected if ticker_selected else "CASH"
            transaction = None

            if type_tx == "Dépot €":
                # On considère `prix` comme le montant du dépôt
                transaction = {
                    "Profil": profil,
                    "Date": date_tx,
                    "Type": "Dépot €",
                    "Ticker": "CASH",
                    "Quantité": round(prix, 2),  # montant en €
                    "Prix": 1.0,
                    "Frais (€/$)": round(frais, 2),
                    "PnL réalisé (€/$)": 0.0,
                    "PnL réalisé (%)": 0.0
                }
            elif type_tx == "Achat":
                transaction = {
                    "Profil": profil,
                    "Date": date_tx,
                    "Type": "Achat",
                    "Ticker": ticker,
                    "Quantité": round(quantite, 8),
                    "Prix": round(prix, 8),
                    "Frais (€/$)": round(frais, 8),
                    "PnL réalisé (€/$)": 0.0,
                    "PnL réalisé (%)": 0.0
                }
            elif type_tx == "Vente":
                df_pos = df_hist[(df_hist["Ticker"] == ticker) & (df_hist["Profil"] == profil)]
                qty_pos = df_pos["Quantité"].sum() if not df_pos.empty else 0
                if qty_pos < quantite:
                    st.error("❌ Pas assez de titres pour vendre.")
                else:
                    achats = df_pos[df_pos["Quantité"] > 0]
                    total_qty_achats = achats["Quantité"].sum() if not achats.empty else 0
                    prix_moyen = ((achats["Quantité"] * achats["Prix"]).sum() / total_qty_achats) if total_qty_achats > 0 else 0
                    pnl_real = (prix - prix_moyen) * quantite - frais
                    pnl_pct = ((prix - prix_moyen) / prix_moyen * 100) if prix_moyen != 0 else 0.0
                    transaction = {
                        "Profil": profil,
                        "Date": date_tx,
                        "Type": "Vente",
                        "Ticker": ticker,
                        "Quantité": -abs(quantite),
                        "Prix": round(prix, 8),
                        "Frais (€/$)": round(frais, 8),
                        "PnL réalisé (€/$)": round(pnl_real, 2),
                        "PnL réalisé (%)": round(pnl_pct, 2)
                    }

            if transaction:
                st.session_state.transactions.append(transaction)
                df_save = pd.DataFrame(st.session_state.transactions)
                for col in EXPECTED_COLS:
                    if col not in df_save.columns:
                        df_save[col] = None
                df_save = df_save[EXPECTED_COLS]
                save_transactions(df_save)
                st.success(f"{type_tx} enregistré : {transaction['Ticker']}")

    # Historique
    st.subheader("Historique des transactions (dernier 200)")
    if st.session_state.transactions:
        df_tx = pd.DataFrame(st.session_state.transactions)
        df_tx["Date"] = pd.to_datetime(df_tx["Date"], errors="coerce")
        st.dataframe(df_tx.sort_values(by="Date", ascending=False).reset_index(drop=True).head(200), width='stretch')
        # Export CSV
        csv = df_tx.to_csv(index=False)
        st.download_button("⬇️ Export CSV", data=csv, file_name="transactions.csv", mime="text/csv")
    else:
        st.info("Aucune transaction enregistrée.")

# ----------------------- Onglet 2 : Portefeuille consolidé -----------------------
with tab2:
    st.header("Portefeuille consolidé")

    if not st.session_state.transactions:
        st.info("Aucune transaction pour calculer le portefeuille.")
    else:
        df_tx = pd.DataFrame(st.session_state.transactions)
        df_tx["Date"] = pd.to_datetime(df_tx["Date"], errors="coerce")
        df_tx = df_tx.sort_values(by="Date")

        # Calculs cash / achats / ventes
        total_depots = df_tx[df_tx["Type"] == "Dépot €"]["Quantité"].sum() if not df_tx[df_tx["Type"] == "Dépot €"].empty else 0.0
        total_achats = (df_tx[df_tx["Type"] == "Achat"]["Quantité"] * df_tx[df_tx["Type"] == "Achat"]["Prix"]).sum() if not df_tx[df_tx["Type"] == "Achat"].empty else 0.0
        ventes = df_tx[df_tx["Type"] == "Vente"]
        total_ventes = ((-ventes["Quantité"]) * ventes["Prix"]).sum() if not ventes.empty else 0.0
        total_frais = df_tx["Frais (€/$)"].sum() if "Frais (€/$)" in df_tx.columns else 0.0

        cash = total_depots + total_ventes - total_achats - total_frais

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
                valeur = (current * qty) if current is not None else None
                pnl_abs = ((current - avg_cost) * qty) if current is not None else None
                pnl_pct = ((current - avg_cost) / avg_cost * 100) if avg_cost not in (0, None) and current is not None and avg_cost != 0 else None
                rows.append({
                    "Ticker": t,
                    "Quantité nette": qty,
                    "Prix moyen pondéré": round(avg_cost, 2) if avg_cost is not None else None,
                    "Prix actuel": round(current, 2) if current is not None else None,
                    "Valeur totale": round(valeur, 2) if valeur is not None else None,
                    "PnL latent (€/$)": round(pnl_abs, 2) if pnl_abs is not None else None,
                    "PnL latent (%)": round(pnl_pct, 2) if pnl_pct is not None else None
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

        st.write(f"Total dépôts : {total_depots:,.2f} € — Total achats : {total_achats:,.2f} € — Total ventes : {total_ventes:,.2f} € — Frais : {total_frais:,.2f} €")

        if not portefeuille.empty:
            st.dataframe(portefeuille.sort_values(by="Valeur totale", ascending=False).reset_index(drop=True), width='stretch')
            try:
                fig = px.pie(portefeuille.dropna(subset=["Valeur totale"]), values="Valeur totale", names="Ticker", title="Répartition du portefeuille")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Impossible d'afficher la répartition (données manquantes).")

            if portefeuille["PnL latent (€/$)"].notna().any():
                fig2 = px.bar(portefeuille.dropna(subset=["PnL latent (€/$)"]), x="Ticker", y="PnL latent (€/$)",
                              text="PnL latent (%)", title="PnL latent par ticker")
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
            st.dataframe(df_tx.reset_index(drop=True), width='stretch')

# ----------------------- Onglet 3 : Répartition par Profil -----------------------
with tab3:
    st.header("Comparatif portefeuilles individuels")
    profils = ["Gas","Marc"]
    cols = st.columns(len(profils))

    for i, p in enumerate(profils):
        df_p = pd.DataFrame(st.session_state.transactions)
        if "Profil" not in df_p.columns:
            df_p["Profil"] = "Gas"
        df_p = df_p[df_p["Profil"]==p]
        if df_p.empty:
            cols[i].info(f"Aucune transaction pour {p}")
            continue

        df_actifs = df_p[df_p["Ticker"] != "CASH"]
        portefeuille = pd.DataFrame()
        if not df_actifs.empty:
            grp = df_actifs.groupby("Ticker", as_index=False).agg({"Quantité":"sum"})
            prix_moy = {}
            for tk in grp["Ticker"]:
                df_tk = df_actifs[df_actifs["Ticker"]==tk]
                achats = df_tk[df_tk["Quantité"] > 0]
                prix_moy[tk] = (achats["Quantité"]*achats["Prix"]).sum()/achats["Quantité"].sum() if not achats.empty else 0.0

            tickers = [t for t in grp["Ticker"] if grp.loc[grp["Ticker"]==t,"Quantité"].values[0] != 0]
            closes = fetch_last_close(tickers)

            rows = []
            for t in tickers:
                qty = grp.loc[grp["Ticker"]==t,"Quantité"].values[0]
                avg_cost = prix_moy.get(t,0.0)
                current = closes.get(t,None)
                valeur = (current*qty) if current is not None else None
                pnl_abs = ((current-avg_cost)*qty) if current is not None else None
                pnl_pct = ((current-avg_cost)/avg_cost*100) if avg_cost not in (0,None) and current is not None and avg_cost != 0 else None
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

        total_valeur = portefeuille["Valeur totale"].sum() if not portefeuille.empty else 0.0
        cols[i].metric(f"{p} : Valeur portefeuilles", f"{total_valeur:,.2f} €")
        if not portefeuille.empty:
            fig = px.pie(portefeuille.dropna(subset=["Valeur totale"]), values="Valeur totale", names="Ticker", title=f"{p} Répartition")
            cols[i].plotly_chart(fig)
