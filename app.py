import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime
from google.oauth2.service_account import Credentials

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Dashboard Portefeuille", layout="wide")
st.markdown("<h1 style='text-align: left; font-size: 30px;'>📊 Dashboard Portefeuille - FBM</h1>", unsafe_allow_html=True)

SHEET_NAME = "transactions_dashboard"   # Nom de ta Google Sheet
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_info = st.secrets["google_service_account"]
credentials = Credentials.from_service_account_info(creds_info, scopes=SCOPE)
client = gspread.authorize(credentials)
sheet = client.open(SHEET_NAME).sheet1

# ---------- Auth Google Sheets (robuste) ----------
# Dans Streamlit Cloud : st.secrets["google_service_account"] doit contenir tout l'objet JSON du service account (dict)
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
    """Récupère le dernier cours de clôture par ticker.
       Utilise yfinance ticker.history en fallback pour plus de robustesse."""
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

def load_transactions():
    if sheet is None:
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None
        # Convert Date column to datetime (keep NaT si invalide)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # ✅ Conversion sécurisée des colonnes numériques
        num_cols = ["Quantité", "Prix", "Frais (€/$)", "PnL réalisé (€/$)", "PnL réalisé (%)"]
        for col in num_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)                     # forcer en string
                    .str.replace(",", ".", regex=False)  # remplacer virgule par point
                    .replace("", "0")                # vides → 0
                    .astype(float)                   # reconvertir en float
                )

        return df
    except Exception as e:
        st.error(f"Erreur lecture Google Sheet: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def save_transactions(df):
    """Ecrit la table entière en une seule opération (plus rapide / moins d'APIs)."""
    if sheet is None:
        st.error("Pas de connexion à Google Sheets : impossible d'enregistrer.")
        return
    try:
        # Prépare les rows (converts dates en ISO)
        rows = []
        for _, r in df.iterrows():
            row = []
            for c in EXPECTED_COLS:
                val = r.get(c, "")
                if pd.isna(val):
                    row.append("")
                elif isinstance(val, (pd.Timestamp, datetime)):
                    row.append(val.isoformat())
                else:
                    row.append(val)
            rows.append(row)
        # Update sheet in bulk
        sheet.clear()
        sheet.append_row(EXPECTED_COLS)
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
# Fonction utilitaire pour convertir nombre avec virgule
def parse_float(val):
    if val is None or val == "":
        return 0.0
    try:
        return float(str(val).replace(",", "."))
    except ValueError:
        return 0.0

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

# ----------------------- Onglet 1 : Saisie Transactions -----------------------
# Fonction utilitaire pour convertir nombre avec virgule
def parse_float(val):
    if val is None or val == "":
        return 0.0
    try:
        return float(str(val).replace(",", "."))
    except ValueError:
        return 0.0

# ----------------------- Onglet 1 : Saisie Transactions -----------------------
with tab1:
    st.header("Ajouter une transaction")

    profil = st.selectbox("Portefeuille / Profil", ["Gas", "Marc"])
    type_tx = st.selectbox("Type", ["Achat", "Vente", "Dépot €"])

    tickers_existants = sorted(set([tx.get("Ticker") for tx in st.session_state.transactions if tx.get("Ticker") and tx.get("Ticker") != "CASH"]))

    ticker_choice = st.selectbox(
        "Ticker (ex: AAPL, BTC-USD)", 
        options=[""] + tickers_existants,
        index=0,
        format_func=lambda x: x if x != "" else ""
    )

    if ticker_choice == "":
        ticker_input = st.text_input("Nouveau ticker (ex: AAPL)").upper().strip()
        ticker = ticker_input
    else:
        ticker = ticker_choice

    quantite = st.text_input("Quantité", "0")
    prix = st.text_input("Prix (€/$)", "0")
    frais = st.text_input("Frais (€/$)", "0")
    date_input = st.date_input("Date de transaction", value=datetime.today())

    if st.button("➕ Ajouter Transaction"):
        # Conversion propre
        quantite = parse_float(quantite)
        prix = parse_float(prix)
        frais = parse_float(frais)

        # Validation
        if type_tx in ("Achat", "Vente") and (not ticker):
            st.error("Ticker requis pour Achat/Vente.")
        elif type_tx == "Dépot €" and prix <= 0:
            st.error("Prix doit être > 0 pour un dépôt.")
        elif type_tx in ("Achat","Vente") and (quantite <= 0 or prix <= 0):
            st.error("Quantité et prix doivent être > 0 pour Achat/Vente.")
        else:
            # Charger historique
            df_hist = pd.DataFrame(st.session_state.transactions) if st.session_state.transactions else pd.DataFrame(columns=EXPECTED_COLS)
            
            # ✅ S'assurer que la colonne "Profil" existe
            if "Profil" not in df_hist.columns:
                df_hist["Profil"] = "Gas"  # valeur par défaut pour anciennes transactions

            # Normaliser date
            try:
                date_tx = pd.to_datetime(date_input)
            except Exception:
                date_tx = pd.Timestamp.now()

            transaction = None
            if type_tx == "Dépot €":
                transaction = {
                    "Profil": profil,
                    "Date": date_tx,
                    "Type": "Dépot €",
                    "Ticker": "CASH",
                    "Quantité": quantite,
                    "Prix": 1,
                    "Frais (€/$)": round(frais,2),
                    "PnL réalisé (€/$)": 0.0,
                    "PnL réalisé (%)": 0.0
                }
            elif type_tx == "Achat":
                transaction = {
                    "Profil": profil,
                    "Date": date_tx,
                    "Type": "Achat",
                    "Ticker": ticker.upper(),
                    "Quantité": quantite,
                    "Prix": round(prix,2),
                    "Frais (€/$)": round(frais,2),
                    "PnL réalisé (€/$)": 0.0,
                    "PnL réalisé (%)": 0.0
                }
            elif type_tx == "Vente":
                df_pos = df_hist[(df_hist["Ticker"] == ticker) & (df_hist["Profil"] == profil)]
                qty_pos = df_pos["Quantité"].sum() if not df_pos.empty else 0
                if qty_pos < quantite:
                    st.error("❌ Pas assez de titres pour vendre.")
                    transaction = None
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
                        "Ticker": ticker.upper(),
                        "Quantité": -abs(quantite),
                        "Prix": round(prix,2),
                        "Frais (€/$)": round(frais,2),
                        "PnL réalisé (€/$)": round(pnl_real,2),
                        "PnL réalisé (%)": round(pnl_pct,2)
                    }

            if transaction:
                st.session_state.transactions.append(transaction)
                df_save = pd.DataFrame(st.session_state.transactions)
                # ✅ Respect de l'ordre des colonnes
                for col in EXPECTED_COLS:
                    if col not in df_save.columns:
                        df_save[col] = None
                df_save = df_save[EXPECTED_COLS]
                save_transactions(df_save)
                st.success(f"{type_tx} enregistré : {transaction['Ticker']}")

    st.subheader("Historique des transactions")
    if st.session_state.transactions:
        df_tx = pd.DataFrame(st.session_state.transactions)
        df_tx["Date"] = pd.to_datetime(df_tx["Date"], errors="coerce")
        st.dataframe(df_tx.sort_values(by="Date", ascending=False).reset_index(drop=True).head(200), width='stretch')
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

        # Calcul du cash
        total_depots = df_tx[df_tx["Type"] == "Dépot €"]["Quantité"].sum() if not df_tx[df_tx["Type"] == "Dépot €"].empty else 0.0
        total_achats = (df_tx[df_tx["Type"] == "Achat"]["Quantité"] * df_tx[df_tx["Type"] == "Achat"]["Prix"]).sum() if not df_tx[df_tx["Type"] == "Achat"].empty else 0.0
        ventes = df_tx[df_tx["Type"] == "Vente"]
        total_ventes = ((-ventes["Quantité"]) * ventes["Prix"]).sum() if not ventes.empty else 0.0
        total_frais = df_tx["Frais"].sum() if "Frais" in df_tx.columns else 0.0

        cash = total_depots + total_ventes - total_achats - total_frais

        # Calcul des positions
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

            # ---------------- Fetch latest close
            closes = {}
            for tk in tickers:
                try:
                    t = yf.Ticker(tk)
                    hist = t.history(period="30d", interval="1d")
                    closes[tk] = float(hist['Close'].dropna().iloc[-1]) if not hist.empty else None
                except Exception:
                    closes[tk] = None
                    st.warning(f"Impossible de récupérer le prix pour {tk}")

            # Construction du portefeuille
            rows = []
            for t in tickers:
                qty = grp.loc[grp["Ticker"]==t, "Quantité"].values[0]
                avg_cost = prix_moy.get(t, 0.0)
                current = closes.get(t, None)
                valeur = (current * qty) if current is not None else None
                pnl_abs = ((current - avg_cost) * qty) if current is not None else None
                pnl_pct = ((current - avg_cost)/avg_cost*100) if avg_cost not in (0,None) and current is not None else None
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

        # Totaux
        total_valeur_actifs = portefeuille["Valeur totale"].sum() if not portefeuille.empty else 0.0
        total_pnl_latent = portefeuille["PnL latent (€/$)"].sum() if not portefeuille.empty else 0.0
        total_pnl_real = df_tx["PnL réalisé (€/$)"].sum() if not df_tx.empty else 0.0
        total_patrimoine = cash + total_valeur_actifs

        # Affichage métriques
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

        # Dataframe et graphiques
        if not portefeuille.empty:
            st.dataframe(portefeuille.sort_values(by="Valeur totale", ascending=False).reset_index(drop=True), width='stretch')

            try:
                fig = px.pie(portefeuille.dropna(subset=["Valeur totale"]), values="Valeur totale", names="Ticker", title="Répartition du portefeuille")
                st.plotly_chart(fig, width='stretch')
            except Exception:
                st.write("Impossible d'afficher la répartition (données manquantes).")

            if portefeuille["PnL latent (€/$)"].notna().any():
                fig2 = px.bar(portefeuille.dropna(subset=["PnL latent (€/$)"]), x="Ticker", y="PnL latent (€/$)",
                              text="PnL latent (%)", title="PnL latent par ticker")
                st.plotly_chart(fig2, width='stretch')
        else:
            st.info("Aucune position ouverte (hors CASH).")

        # Graphique PnL cumulatif ventes
        df_ventes = df_tx[df_tx["Type"] == "Vente"].copy()
        if not df_ventes.empty:
            df_ventes = df_ventes.sort_values(by="Date")
            df_ventes["Cumul PnL réalisé"] = df_ventes["PnL réalisé (€/$)"].cumsum()
            fig3 = px.line(df_ventes, x="Date", y="Cumul PnL réalisé", title="PnL Réalisé Cumulatif")
            st.plotly_chart(fig3, width='stretch')

        # Expander transactions détaillées
        with st.expander("Voir transactions détaillées"):
            st.dataframe(df_tx.reset_index(drop=True), width='stretch')

# ----------------------- Onglet 3 : Répartition par Profil -----------------------
with tab3:
    st.header("Comparatif portefeuilles individuels")
    profils = ["Gas","Marc"]
    cols = st.columns(len(profils))
    
    for i, p in enumerate(profils):
        df_p = pd.DataFrame(st.session_state.transactions)

# ✅ S'assurer que la colonne "Profil" existe
        if "Profil" not in df_p.columns:
            df_p["Profil"] = "Gas"  # valeur par défaut pour les anciennes transactions
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

            tickers = [t for t in grp["Ticker"] if grp.loc[grp["Ticker"]==t,"Quantité"].values[0]!=0]
            closes = fetch_last_close(tickers)
            rows = []
            for t in tickers:
                qty = grp.loc[grp["Ticker"]==t,"Quantité"].values[0]
                avg_cost = prix_moy.get(t,0.0)
                current = closes.get(t,None)
                valeur = (current*qty) if current is not None else None
                pnl_abs = ((current-avg_cost)*qty) if current is not None else None
                pnl_pct = ((current-avg_cost)/avg_cost*100) if avg_cost not in (0,None) and current is not None else None
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
        total_pnl_latent = portefeuille["PnL latent (€/$)"].sum() if not portefeuille.empty else 0.0
        total_pnl_real = df_p["PnL réalisé (€/$)"].sum() if not df_p.empty else 0.0
        total_depots = df_p[df_p["Type"]=="Dépot €"]["Quantité"].sum() if not df_p[df_p["Type"]=="Dépot €"].empty else 0.0
        total_achats = (df_p[df_p["Type"]=="Achat"]["Quantité"]*df_p[df_p["Type"]=="Achat"]["Prix"]).sum() if not df_p[df_p["Type"]=="Achat"].empty else 0.0
        total_ventes = ((-df_p[df_p["Type"]=="Vente"]["Quantité"])*df_p[df_p["Type"]=="Vente"]["Prix"]).sum() if not df_p[df_p["Type"]=="Vente"].empty else 0.0
        total_frais = df_p["Frais"].sum() if "Frais" in df_p.columns else 0.0
        cash = total_depots + total_ventes - total_achats - total_frais
        
        cols[i].metric(f"💵 Liquidités {p}", f"{cash:,.2f} €")
        cols[i].metric(f"📊 Valeur Actifs {p}", f"{total_valeur:,.2f} €")
        cols[i].metric(f"📈 PnL latent {p}", f"{total_pnl_latent:,.2f} €")
        cols[i].metric(f"✅ PnL réalisé {p}", f"{total_pnl_real:,.2f} €")
        
        if not portefeuille.empty:
            cols[i].dataframe(portefeuille.sort_values(by="Valeur totale",ascending=False).reset_index(drop=True), width='stretch')
