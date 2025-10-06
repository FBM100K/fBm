import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date
from google.oauth2.service_account import Credentials
import requests
import uuid
import time

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Dashboard Portefeuille", layout="wide")
st.markdown("<h1 style='text-align: left; font-size: 30px;'>📊 Dashboard Portefeuille - FBM</h1>", unsafe_allow_html=True)

SHEET_NAME = "transactions_dashboard"
# Schéma étendu : ajouté id et DateTime_UTC, Devise, Source, Note
EXPECTED_COLS = [
    "Date",
    "Profil",
    "Type",
    "Ticker",
    "Quantité",
    "Prix_unitaire",
    "Devise",
    "Frais (€/$)",
    "PnL réalisé (€/$)",
    "PnL réalisé (%)",
    "Note"
]

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# -----------------------
# Google Sheets Auth (robuste)
# -----------------------
creds_info = None
sheet = None
gc_client = None
try:
    creds_info = st.secrets["google_service_account"]
    credentials = Credentials.from_service_account_info(creds_info, scopes=SCOPE)
    gc_client = gspread.authorize(credentials)
    sh = gc_client.open(SHEET_NAME)
    sheet = sh.sheet1
except Exception as e:
    st.error("Erreur d'authentification Google Sheets — vérifie st.secrets et le partage de la feuille.")
    st.exception(e)
    sheet = None
    sh = None
    gc_client = None

# -----------------------
# Helpers utilitaires
# -----------------------
def generate_uuid():
    return str(uuid.uuid4())

def now_utc_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_float(val):
    """Parsing robuste des valeurs numériques venant de forms / sheets."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == "":
        return 0.0
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0

def safe_str(val):
    if pd.isna(val) or val is None:
        return ""
    return str(val)

# -----------------------
# Lecture / écriture transactions (Google Sheet)
# -----------------------
@st.cache_data(ttl=30)
def load_transactions_from_sheet():
    """
    Charge la table depuis Google Sheets et normalise les colonnes.
    Retourne un DataFrame avec toutes les colonnes EXPECTED_COLS (ajoute si manquantes).
    """
    if sheet is None:
        # retourne df vide avec les colonnes attendues
        return pd.DataFrame(columns=EXPECTED_COLS)

    try:
        records = sheet.get_all_records()
        df = pd.DataFrame(records)

        # Ajouter colonnes manquantes
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None

        # Normaliser types
        # Date -> datetime.date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

        # DateTime_UTC -> datetime
        if "DateTime_UTC" in df.columns:
            df["DateTime_UTC"] = pd.to_datetime(df["DateTime_UTC"], errors="coerce")

        # Numeric columns
        numeric_cols = ["Quantité", "Prix_unitaire", "Frais (€/$)", "PnL réalisé (€/$)", "PnL réalisé (%)"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(["", "None", "nan"], "0").str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Devise default
        if "Devise" not in df.columns:
            df["Devise"] = "EUR"

        # Profil default
        if "Profil" not in df.columns:
            df["Profil"] = "Gas"

        # Type normalisation
        df["Type"] = df["Type"].fillna("Achat")

        # Ensure ordering of columns follows EXPECTED_COLS
        df = df.reindex(columns=EXPECTED_COLS)

        return df
    except Exception as e:
        st.error(f"Erreur lecture Google Sheet: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def backup_sheet(sh_obj):
    """Essaye de créer un onglet backup_{timestamp} contenant snapshot courant (best effort)."""
    if sh_obj is None:
        return
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"backup_{ts}"
        # si nom existe déjà, suffixe
        existing_titles = [w.title for w in sh_obj.worksheets()]
        if name in existing_titles:
            name = name + "_" + str(int(time.time()))
        # create worksheet (rows/cols minimal)
        new_ws = sh_obj.add_worksheet(title=name, rows="1000", cols="20")
        # write current data
        df_cur = load_transactions_from_sheet()
        values = [EXPECTED_COLS] + df_cur.fillna("").astype(str).values.tolist()
        new_ws.update("A1", values, value_input_option="USER_ENTERED")
    except Exception:
        # best-effort, ne pas planter l'app si backup échoue
        pass

def save_transactions_to_sheet(df):
    """
    Écriture atomique dans la Sheet : on met à jour A1 avec la totalité des valeurs.
    Avant écriture on effectue un backup best-effort.
    """
    if sheet is None or sh is None:
        st.error("Pas de connexion à Google Sheets : impossible d'enregistrer.")
        return False

    # Normaliser le df pour l'export
    df_out = df.copy()
    # Convertir Date en ISO (YYYY-MM-DD) pour que Sheets le détecte comme date via USER_ENTERED
    if "Date" in df_out.columns:
        df_out["Date"] = df_out["Date"].apply(lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) and isinstance(d, (date, pd.Timestamp)) else (d if d else ""))
    if "DateTime_UTC" in df_out.columns:
        df_out["DateTime_UTC"] = df_out["DateTime_UTC"].apply(lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(dt) and isinstance(dt, (datetime, pd.Timestamp)) else (dt if dt else ""))

    # Assure toutes les colonnes attendues présentes
    for c in EXPECTED_COLS:
        if c not in df_out.columns:
            df_out[c] = ""

    # Formater lignes
    values = [EXPECTED_COLS] + df_out[EXPECTED_COLS].fillna("").astype(str).values.tolist()

    try:
        # backup best-effort
        backup_sheet(sh)

        # écriture atomique : update A1
        sheet.update("A1", values, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        st.error(f"Erreur écriture Google Sheet: {e}")
        return False

# -----------------------
# Fetch prices - batch (optimisé)
# -----------------------
@st.cache_data(ttl=60)
def fetch_last_close_batch(tickers):
    """
    Retourne dict {ticker: last_close_or_None}.
    Utilise yfinance.download en batch, fallback per-ticker si nécessaire.
    """
    result = {}
    if not tickers:
        return result
    # filtrer tickers invalides et 'CASH'
    tickers = sorted({t.strip().upper() for t in tickers if t and str(t).strip().upper() != "CASH"})
    if not tickers:
        return result
    try:
        # yfinance.download supporte une liste; group_by='ticker' renvoie MultiIndex quand multi tickers
        data = yf.download(tickers, period="7d", progress=False, threads=True, group_by='ticker', auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    ser = data[t]['Close'].dropna()
                    result[t] = float(ser.iloc[-1]) if not ser.empty else None
                except Exception:
                    result[t] = None
        else:
            # single ticker case
            try:
                ser = data['Close'].dropna()
                result[tickers[0]] = float(ser.iloc[-1]) if not ser.empty else None
            except Exception:
                result[tickers[0]] = None
        return result
    except Exception:
        # fallback per ticker (plus lent)
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="7d")
                result[t] = float(hist['Close'].dropna().iloc[-1]) if not hist.empty else None
            except Exception:
                result[t] = None
        return result

def format_pct(val):
    if pd.isna(val) or val is None:
        return ""
    try:
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.2f}%"
    except Exception:
        return ""

# -----------------------
# State init (DataFrame centralisé)
# -----------------------
if "df_transactions" not in st.session_state:
    df_loaded = load_transactions_from_sheet()
    st.session_state.df_transactions = df_loaded

# util: rafraichir depuis sheet
def refresh_from_sheet():
    st.session_state.df_transactions = load_transactions_from_sheet()
    st.experimental_rerun()

# -----------------------
# Onglets
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["💰 Transactions", "📂 Portefeuille", "📊 Répartition", "Calendrier"])

# -----------------------
# Onglet 1 : Saisie Transactions avec recherche Alpha Vantage
# -----------------------
with tab1:
    st.header("Ajouter une transaction")

    profil = st.selectbox("Portefeuille / Profil", ["Gas", "Marc"], index=0)
    type_tx = st.selectbox("Type", ["Achat", "Vente", "Dépot €", "Dividende", "Frais"], index=0)

    st.markdown("Recherche de titre (Ticker ou Nom d’entreprise)")

    # ---- Initialisation variables session ----
    if "ticker_query" not in st.session_state:
        st.session_state.ticker_query = ""
    if "ticker_suggestions" not in st.session_state:
        st.session_state.ticker_suggestions = []
    if "ticker_selected" not in st.session_state:
        st.session_state.ticker_selected = ""

    # ---- Alpha Vantage config (optionnel) ----
    ALPHA_VANTAGE_API_KEY = None
    try:
        ALPHA_VANTAGE_API_KEY = st.secrets["alpha_vantage"]["api_key"]
    except Exception:
        ALPHA_VANTAGE_API_KEY = None

    def get_alpha_vantage_suggestions(query: str):
        """Recherche de tickers via Alpha Vantage (best effort)."""
        if not ALPHA_VANTAGE_API_KEY or not query or len(query) < 2:
            return []
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": query,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            matches = data.get("bestMatches", [])
            suggestions = []
            for m in matches:
                symbol = m.get("1. symbol", "")
                name = m.get("2. name", "")
                region = m.get("4. region", "")
                if symbol and name:
                    suggestions.append(f"{symbol} — {name} ({region})")
            return suggestions[:15]
        except Exception:
            return []

    # ---- Interface recherche simple ----
    query = st.text_input("Entrez un nom ou ticker à rechercher :", value=st.session_state.ticker_query)
    if st.button("🔎 Rechercher"):
        st.session_state.ticker_query = query
        if query:
            st.session_state.ticker_suggestions = get_alpha_vantage_suggestions(query)
            if st.session_state.ticker_suggestions:
                sel = st.selectbox("Résultats trouvés :", st.session_state.ticker_suggestions, key=f"ticker_selectbox_{generate_uuid()}")
                st.session_state.ticker_selected = sel.split(" — ")[0]
            else:
                st.warning("Aucun résultat trouvé (ou clé AlphaVantage manquante).")
        else:
            st.info("Veuillez saisir un mot-clé pour lancer la recherche.")

    # ---- Feedback sélection ----
    if st.session_state.ticker_selected:
        st.success(f"✅ Ticker sélectionné : {st.session_state.ticker_selected}")

    ticker_selected = st.session_state.ticker_selected or None

    # ---- Saisie transaction ----
    quantite_input = st.text_input("Quantité", "0")
    prix_input = st.text_input("Prix unitaire (€/$)", "0")
    frais_input = st.text_input("Frais (€/$)", "0")
    date_input = st.date_input("Date de transaction", value=datetime.today())
    devise = st.selectbox("Devise", ["EUR", "USD"], index=0)
    note = st.text_input("Note (optionnel)", "")

    if st.button("➕ Ajouter Transaction"):
        quantite = parse_float(quantite_input)
        prix = parse_float(prix_input)
        frais = parse_float(frais_input)

        # Validation basique
        if type_tx in ("Achat", "Vente") and not ticker_selected:
            st.error("Ticker requis pour Achat/Vente.")
        elif type_tx == "Dépot €" and prix <= 0 and quantite == 0:
            # dépôt : on autorise quantité comme montant si tu utilises Quantité pour montant
            st.error("Prix ou Quantité doit être > 0 pour un dépôt.")
        elif type_tx in ("Achat", "Vente") and (quantite <= 0 or prix <= 0):
            st.error("Quantité et prix doivent être > 0 pour Achat/Vente.")
        else:
            df_hist = st.session_state.df_transactions.copy() if st.session_state.df_transactions is not None else pd.DataFrame(columns=EXPECTED_COLS)
            # Ensure required cols
            for c in EXPECTED_COLS:
                if c not in df_hist.columns:
                    df_hist[c] = None

            try:
                date_tx = pd.to_datetime(date_input).date()
            except Exception:
                date_tx = datetime.utcnow().date()

            ticker = ticker_selected if ticker_selected else "CASH"
            tx_id = generate_uuid()
            dt_write = datetime.utcnow()

            transaction = {
                "Date": date_tx,
                "Profil": profil,
                "Type": type_tx,
                "Ticker": ticker,
                "Quantité": 0.0,
                "Prix_unitaire": 0.0,
                "Devise": devise,
                "Frais (€/$)": round(frais, 2),
                "PnL réalisé (€/$)": 0.0,
                "PnL réalisé (%)": 0.0,
                "Note": note
            }

            if type_tx == "Dépot €":
                # On stocke dépôt comme CASH avec Quantité = montant (ou tu peux utiliser Prix_unitaire)
                transaction["Ticker"] = "CASH"
                # utiliser Quantité pour le montant du dépôt pour compatibilité précédente
                transaction["Quantité"] = round(quantite if quantite > 0 else prix, 2)
                transaction["Prix_unitaire"] = 1.0
            elif type_tx == "Achat":
                transaction["Quantité"] = round(quantite, 6)
                transaction["Prix_unitaire"] = round(prix, 6)
            elif type_tx == "Vente":
                # Vérifier quantité disponible
                df_pos = df_hist[(df_hist["Ticker"] == ticker) & (df_hist["Profil"] == profil)]
                qty_pos = df_pos["Quantité"].sum() if not df_pos.empty else 0.0
                if qty_pos < quantite:
                    st.error("❌ Pas assez de titres pour vendre.")
                    transaction = None
                else:
                    # calcul PnL réalisé basique basé sur prix moyen d'achat
                    achats = df_pos[df_pos["Quantité"] > 0]
                    total_qty_achats = achats["Quantité"].sum() if not achats.empty else 0.0
                    prix_moyen = ((achats["Quantité"] * achats["Prix_unitaire"]).sum() / total_qty_achats) if total_qty_achats > 0 else 0.0
                    pnl_real = (prix - prix_moyen) * quantite - frais
                    pnl_pct = ((prix - prix_moyen) / prix_moyen * 100) if prix_moyen != 0 else 0.0
                    transaction["Quantité"] = -abs(round(quantite, 6))
                    transaction["Prix_unitaire"] = round(prix, 6)
                    transaction["PnL réalisé (€/$)"] = round(pnl_real, 2)
                    transaction["PnL réalisé (%)"] = round(pnl_pct, 2)

            if transaction:
                # Append to DataFrame and save
                df_new = pd.concat([df_hist, pd.DataFrame([transaction])], ignore_index=True)
                # Reindex to expected columns
                for c in EXPECTED_COLS:
                    if c not in df_new.columns:
                        df_new[c] = None
                df_new = df_new[EXPECTED_COLS]
                ok = save_transactions_to_sheet(df_new)
                if ok:
                    st.success(f"{type_tx} enregistré : {transaction['Ticker']}")
                    st.session_state.df_transactions = df_new
                else:
                    st.error("Erreur lors de l'enregistrement. Transaction non sauvegardée.")

    # ---- Historique ----
    st.subheader("Historique des transactions (dernières 200)")
    if st.session_state.df_transactions is not None and not st.session_state.df_transactions.empty:
        df_tx = st.session_state.df_transactions.copy()
        # afficher trié
        if "Date" in df_tx.columns:
            # transformer date pour tri
            df_tx["Date_sort"] = pd.to_datetime(df_tx["Date"], errors="coerce")
            st.dataframe(df_tx.sort_values(by="Date_sort", ascending=False).drop(columns=["Date_sort"]).reset_index(drop=True).head(200), width='stretch')
        else:
            st.dataframe(df_tx.reset_index(drop=True).head(200), width='stretch')
    else:
        st.info("Aucune transaction enregistrée.")

# -----------------------
# Onglet 2 : Portefeuille consolidé
# -----------------------
with tab2:
    st.header("Portefeuille consolidé")

    df_tx = st.session_state.df_transactions.copy() if st.session_state.df_transactions is not None else pd.DataFrame(columns=EXPECTED_COLS)

    if df_tx.empty:
        st.info("Aucune transaction pour calculer le portefeuille.")
    else:
        # Normaliser types / colonnes
        if "Date" in df_tx.columns:
            df_tx["Date"] = pd.to_datetime(df_tx["Date"], errors="coerce")
        # Calculs cash / achats / ventes
        depots_mask = df_tx["Type"].str.lower() == "dépot €"
        total_depots = df_tx.loc[depots_mask, "Quantité"].sum() if not df_tx.loc[depots_mask].empty else 0.0

        achats_mask = df_tx["Type"].str.lower() == "achat"
        total_achats = (df_tx.loc[achats_mask, "Quantité"] * df_tx.loc[achats_mask, "Prix_unitaire"]).sum() if not df_tx.loc[achats_mask].empty else 0.0

        ventes_mask = df_tx["Type"].str.lower() == "vente"
        ventes = df_tx.loc[ventes_mask]
        total_ventes = ((-ventes["Quantité"]) * ventes["Prix_unitaire"]).sum() if not ventes.empty else 0.0

        total_frais = df_tx["Frais (€/$)"].sum() if "Frais (€/$)" in df_tx.columns else 0.0

        cash = total_depots + total_ventes - total_achats - total_frais

        df_actifs = df_tx[df_tx["Ticker"].str.upper() != "CASH"]
        portefeuille = pd.DataFrame()

        if not df_actifs.empty:
            grp = df_actifs.groupby("Ticker", as_index=False).agg({"Quantité": "sum"})
            prix_moy = {}
            for tk in grp["Ticker"]:
                df_tk = df_actifs[df_actifs["Ticker"] == tk]
                achats = df_tk[df_tk["Quantité"] > 0]
                if not achats.empty:
                    denom = achats["Quantité"].sum()
                    prix_moy[tk] = ((achats["Quantité"] * achats["Prix_unitaire"]).sum() / denom) if denom != 0 else 0.0
                else:
                    prix_moy[tk] = 0.0

            tickers = [t for t in grp["Ticker"] if grp.loc[grp["Ticker"]==t, "Quantité"].values[0] != 0]
            closes = fetch_last_close_batch(tickers)

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
                    "Prix moyen pondéré": round(avg_cost, 6) if avg_cost is not None else None,
                    "Prix actuel": round(current, 6) if current is not None else None,
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

        df_ventes = df_tx[df_tx["Type"].str.lower() == "vente"].copy()
        if not df_ventes.empty:
            df_ventes = df_ventes.sort_values(by="Date")
            df_ventes["Cumul PnL réalisé"] = df_ventes["PnL réalisé (€/$)"].cumsum()
            fig3 = px.line(df_ventes, x="Date", y="Cumul PnL réalisé", title="PnL Réalisé Cumulatif")
            st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Voir transactions détaillées"):
            st.dataframe(df_tx.reset_index(drop=True), width='stretch')

# -----------------------
# Onglet 3 : Répartition par Profil
# -----------------------
with tab3:
    st.header("Comparatif portefeuilles individuels")
    profils = sorted(st.session_state.df_transactions["Profil"].unique().tolist()) if st.session_state.df_transactions is not None and "Profil" in st.session_state.df_transactions.columns else ["Gas", "Marc"]
    cols = st.columns(max(1, len(profils)))

    for i, p in enumerate(profils):
        df_p = st.session_state.df_transactions.copy()
        if "Profil" not in df_p.columns:
            df_p["Profil"] = "Gas"
        df_p = df_p[df_p["Profil"]==p]
        if df_p.empty:
            cols[i].info(f"Aucune transaction pour {p}")
            continue

        df_actifs = df_p[df_p["Ticker"].str.upper() != "CASH"]
        portefeuille = pd.DataFrame()
        if not df_actifs.empty:
            grp = df_actifs.groupby("Ticker", as_index=False).agg({"Quantité":"sum"})
            prix_moy = {}
            for tk in grp["Ticker"]:
                df_tk = df_actifs[df_actifs["Ticker"]==tk]
                achats = df_tk[df_tk["Quantité"] > 0]
                prix_moy[tk] = (achats["Quantité"]*achats["Prix_unitaire"]).sum()/achats["Quantité"].sum() if not achats.empty else 0.0

            tickers = [t for t in grp["Ticker"] if grp.loc[grp["Ticker"]==t,"Quantité"].values[0] != 0]
            closes = fetch_last_close_batch(tickers)

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
                    "Prix moyen pondéré": round(avg_cost,6),
                    "Prix actuel": round(current,6) if current is not None else None,
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

# -----------------------
# Onglet 4 : Calendrier (placeholder)
# -----------------------
with tab4:
    st.header("Calendrier économique / Événements (à venir)")
    st.info("Fonctionnalité non encore implémentée dans cette version — backlog: intégration calendrier économique & reporting résultats.")
