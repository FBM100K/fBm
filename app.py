"""
Dashboard Portefeuille V2.1 - Multi-devises
Fusion optimisée V1 + V2 avec PRU figé, taux de change figés, validation devise unique
"""

import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date
from google.oauth2.service_account import Credentials
import requests

# Import des moteurs
from portfolio_engine import PortfolioEngine
from currency_manager import CurrencyManager

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Dashboard Portefeuille V2.1", layout="wide")
st.markdown("<h1 style='text-align: left; font-size: 32px;'>📊 Dashboard Portefeuille - FBM V2.1</h1>", unsafe_allow_html=True)

SHEET_NAME = "transactions_dashboard"

# ---- Initialisation clé devise ----
if "devise_affichage" not in st.session_state:
    st.session_state.devise_affichage = "EUR"

# ---- Cache local pour les tickers (nom complet / région)
if "ticker_cache" not in st.session_state:
    st.session_state.ticker_cache = {}

# -----------------------
# Initialisation des états Streamlit
# -----------------------
if "currency_manager" not in st.session_state:
    from currency_manager import CurrencyManager
    st.session_state.currency_manager = CurrencyManager()

currency_manager = st.session_state.currency_manager

if "df_transactions" not in st.session_state:
    st.session_state.df_transactions = None

# ⭐ NOUVEAU : Toggle devise en header
col_title, col_currency = st.columns([3, 1])
with col_title:
    st.divider()

    # Indicateur taux de change
    cache_info = currency_manager.get_cache_info()
    if cache_info["status"] != "Non initialisé":
        if cache_info["using_fallback"]:
            st.warning(f"⚠️ {cache_info['status']}")
        else:
            st.success(f"✅ {cache_info['status']}")
        st.caption(f"Mise à jour: {cache_info['last_update']}")
    
    # Indicateur PRU_vente
    if st.session_state.df_transactions is not None:
        ventes = st.session_state.df_transactions[st.session_state.df_transactions["Type"] == "Vente"]
        if not ventes.empty:
            ventes_avec_pru = ventes[ventes["PRU_vente"].notna() & (ventes["PRU_vente"] > 0)]
            pct_migre = len(ventes_avec_pru) / len(ventes) * 100
            
            if pct_migre < 100:
                st.warning(f"⚠️ Migration V2 incomplète: {pct_migre:.0f}%")
            else:
                st.success("✅ Toutes les ventes ont PRU_vente")

# COLONNES V2.1 MULTI-DEVISES
EXPECTED_COLS = [
    "Date",
    "Profil",
    "Type",
    "Ticker",
    "Nom complet",
    "Quantité",
    "Prix_unitaire",
    "PRU_vente",
    "Devise",
    "Taux_change",
    "Devise_reference",
    "Frais (€/$)",
    "PnL réalisé (€/$)",
    "PnL réalisé (%)",
    "Note",
    "History_Log"
]

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# -----------------------
# Google Sheets Auth
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
    st.error("Erreur d'authentification Google Sheets")
    st.exception(e)
    sheet = None
    sh = None
    gc_client = None

# -----------------------
# Helpers
# -----------------------
def parse_float(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", ".")
    if s == "":
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

# -----------------------
# Lecture / écriture transactions
# -----------------------
@st.cache_data(ttl=0)
def load_transactions_from_sheet():
    if sheet is None:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    try:
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None
        
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        
        numeric_cols = ["Quantité", "Prix_unitaire", "Frais (€/$)", 
                        "PnL réalisé (€/$)", "PnL réalisé (%)", "PRU_vente", "Taux_change"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(["", "None", "nan"], "0").str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        df["Devise"] = df["Devise"].fillna("EUR")
        df["Devise_reference"] = df["Devise_reference"].fillna("EUR")
        df["Profil"] = df["Profil"].fillna("Gas")
        df["Type"] = df["Type"].fillna("Achat")
        
        df = df.reindex(columns=EXPECTED_COLS)
        
        return df
    except Exception as e:
        st.error(f"Erreur lecture Google Sheet: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def save_transactions_to_sheet(df):
    if sheet is None or sh is None:
        st.error("Pas de connexion à Google Sheets")
        return False
    
    df_out = df.copy()
    if df.empty:
        st.error("Tentative de sauvegarde d'un DataFrame vide — opération annulée.")
        return False
    
    # ---- Formatage des dates ----
    if "Date" in df_out.columns:
        df_out["Date"] = df_out["Date"].apply(
            lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) and isinstance(d, (date, pd.Timestamp)) else (d if d else "")
        )
    
    # ---- Vérifie que toutes les colonnes attendues existent ----
    for c in EXPECTED_COLS:
        if c not in df_out.columns:
            df_out[c] = ""
    
    values = [EXPECTED_COLS] + df_out[EXPECTED_COLS].fillna("").astype(str).values.tolist()
    
    try:
        # ---- 🔄 Création du backup avant écrasement ----
        try:
            backup_name = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            old_data = sheet.get_all_values()
            
            backup_ws = sh.add_worksheet(
                title=backup_name,
                rows=str(len(old_data) + 5),
                cols=str(len(EXPECTED_COLS))
            )
            
            if old_data:
                backup_ws.update("A1", old_data, value_input_option="USER_ENTERED")
            
            st.info(f"📦 Backup créé : {backup_name}")
        
        except Exception as e:
            st.warning(f"⚠️ Backup non créé : {e}")
        
        # ---- ✏️ Mise à jour du sheet principal ----
        sheet.clear()
        sheet.update("A1", values, value_input_option="USER_ENTERED")
        
        # ---- ♻️ Rotation automatique des backups (max 5) ----
        try:
            backups = [w for w in sh.worksheets() if w.title.startswith("backup_")]
            if len(backups) > 5:
                backups_sorted = sorted(backups, key=lambda w: w.title, reverse=True)
                for old in backups_sorted[5:]:
                    sh.del_worksheet(old)
                    st.info(f"🗑️ Ancien backup supprimé : {old.title}")
        except Exception as e:
            st.warning(f"⚠️ Rotation backup non appliquée : {e}")
        
        return True
    
    except Exception as e:
        st.error(f"Erreur écriture : {e}")
        return False

# -----------------------
# Fetch prices
# -----------------------
@st.cache_data(ttl=60)
def fetch_last_close_batch(tickers):
    result = {}
    if not tickers:
        return result
    
    tickers = sorted({t.strip().upper() for t in tickers if t and str(t).strip().upper() != "CASH"})
    if not tickers:
        return result
    
    try:
        data = yf.download(tickers, period="7d", progress=False, threads=True, group_by='ticker', auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    ser = data[t]['Close'].dropna()
                    result[t] = float(ser.iloc[-1]) if not ser.empty else None
                except:
                    result[t] = None
        else:
            try:
                ser = data['Close'].dropna()
                result[tickers[0]] = float(ser.iloc[-1]) if not ser.empty else None
            except:
                result[tickers[0]] = None
        return result
    except:
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="7d")
                result[t] = float(hist['Close'].dropna().iloc[-1]) if not hist.empty else None
            except:
                result[t] = None
        return result

# -----------------------
# State init
# -----------------------
if sheet is not None:
    if "df_transactions" not in st.session_state or st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        df_loaded = load_transactions_from_sheet()
        if df_loaded is not None and not df_loaded.empty:
            st.session_state.df_transactions = df_loaded
        else:
            st.warning("⚠️ Aucune donnée chargée depuis Google Sheets (peut-être vide ou non accessible)")
else:
    st.error("❌ Impossible de se connecter à Google Sheets - vérifiez st.secrets")


if "currency_manager" not in st.session_state:
    st.session_state.currency_manager = CurrencyManager()

currency_manager = st.session_state.currency_manager

# -----------------------
# Changement devise
# -----------------------
with col_currency:
    if "devise_affichage" not in st.session_state:
        st.session_state.devise_affichage = "EUR"
    
    devise_affichage = st.radio(
        "💱 Devise",
        options=["EUR", "USD"],
        index=0 if st.session_state.devise_affichage == "EUR" else 1,
        horizontal=True,
        key="currency_toggle"
    )
    st.session_state.devise_affichage = devise_affichage

# -----------------------
# Onglets
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["💰 Transactions", "📂 Portefeuille", "📊 Répartition", "Calendrier"])

# -----------------------
# ONGLET 1 : Transactions
# -----------------------
with tab1:
    st.header("Ajouter une transaction")
    
    profil = st.selectbox("Portefeuille / Profil", ["Gas", "Marc"], index=0)
    type_tx = st.selectbox("Type", ["Achat", "Vente", "Dépôt", "Retrait", "Dividende"], index=0)
    
    st.markdown("### Recherche de titre")
    
    if "ticker_query" not in st.session_state:
        st.session_state.ticker_query = ""
    if "ticker_suggestions" not in st.session_state:
        st.session_state.ticker_suggestions = []
    if "ticker_selected" not in st.session_state:
        st.session_state.ticker_selected = ""
    
    ALPHA_VANTAGE_API_KEY = None
    try:
        ALPHA_VANTAGE_API_KEY = st.secrets["alpha_vantage"]["api_key"]
    except:
        ALPHA_VANTAGE_API_KEY = None
    
    @st.cache_data(ttl=1600)
    def get_alpha_vantage_suggestions(query: str):
        """Recherche des tickers sur Alpha Vantage, avec cache local + debug."""
        if not query or len(query.strip()) < 2:
            return []

        if not ALPHA_VANTAGE_API_KEY:
            st.warning("⚠️ Clé API Alpha Vantage manquante.")
            return []

    # Cache local pour éviter de rappeler Alpha Vantage inutilement
        if "suggestion_cache" not in st.session_state:
            st.session_state.suggestion_cache = {}

        query = query.strip().lower()
        if query in st.session_state.suggestion_cache:
            return st.session_state.suggestion_cache[query]

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": query,
            "apikey": ALPHA_VANTAGE_API_KEY
        }

        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            matches = data.get("bestMatches", [])

            if not matches:
                # 🔍 Ajoute un log temporaire pour diagnostiquer le problème
                st.info(f"Aucun résultat pour '{query}'. Réponse brute : {data}")
                return []

            suggestions = []
            for m in matches:
                symbol = m.get("1. symbol", "")
                name = m.get("2. name", "")
                region = m.get("4. region", "")
                if symbol and name:
                    suggestions.append(f"{symbol} — {name} ({region})")

            st.session_state.suggestion_cache[query] = suggestions[:15]
            return suggestions[:15]

        except Exception as e:
            st.error(f"Erreur Alpha Vantage : {e}")
            return []

    # --- Barre de recherche alignée horizontalement ---
    col_rech1, col_rech2 = st.columns([5, 1])
    with col_rech1:
        query = st.text_input(
            "Entrez un nom ou ticker :",
            value=st.session_state.ticker_query,
            label_visibility="collapsed",  # cache le label pour un meilleur alignement
            placeholder="Ex: AAPL, Tesla..."
        )
    with col_rech2:
        st.write("")  # espace vertical pour aligner le bouton
        if st.button("🔎 Rechercher", use_container_width=True):
            st.session_state.ticker_query = query
            if query:
                st.session_state.ticker_suggestions = get_alpha_vantage_suggestions(query)
                if not st.session_state.ticker_suggestions:
                    st.warning("Aucun résultat")

    if st.session_state.ticker_suggestions:
        sel = st.selectbox("Résultats :", st.session_state.ticker_suggestions, key="ticker_selectbox")
        if sel:
            st.session_state.ticker_selected = sel.split(" — ")[0]
    
    if st.session_state.ticker_selected:
        st.success(f"✅ Ticker : {st.session_state.ticker_selected}")
    
    ticker_selected = st.session_state.ticker_selected or None
    
    st.markdown("### Détails de la transaction")
    
    col1, col2 = st.columns(2)
    with col1:
        quantite_input = st.text_input("Quantité", "0")
        prix_input = st.text_input("Prix unitaire (€/$)", "1.0" if type_tx in ["Dépôt", "Retrait"] else "0")
    with col2:
        frais_input = st.text_input("Frais (€/$)", "0")
        date_input = st.date_input("Date", value=datetime.today())
    
    devise = st.selectbox("Devise", ["EUR", "USD"], index=0)
    note = st.text_area("Note (optionnel)", "", max_chars=500)

    @st.cache_data(ttl=1600)
    def get_ticker_full_name_from_api(ticker: str):
        """Requête Alpha Vantage pour obtenir le nom complet et la région."""
        if not ALPHA_VANTAGE_API_KEY or not ticker:
            return ticker
        try:
            url = "https://www.alphavantage.co/query"
            params = {"function": "SYMBOL_SEARCH", "keywords": ticker, "apikey": ALPHA_VANTAGE_API_KEY}
            res = requests.get(url, params=params, timeout=10)
            data = res.json().get("bestMatches", [])
            if not data:
                return ticker
            m = data[0]
            name = m.get("2. name", "")
            region = m.get("4. region", "")
            return f"{name} ({region})" if name else ticker
        except Exception:
            return ticker

    def get_ticker_full_name(ticker: str):
        """Retourne le nom complet depuis le cache local, ou via API si inconnu."""
        ticker = ticker.upper().strip()
        cache = st.session_state.ticker_cache

        if ticker in cache:
            return cache[ticker]

        # Appel API seulement si pas en cache
        full_name = get_ticker_full_name_from_api(ticker)
        cache[ticker] = full_name  # On enregistre la valeur dans le cache local
        st.session_state.ticker_cache = cache
        return full_name

    if st.button("➕ Ajouter Transaction", type="primary"):
        quantite = parse_float(quantite_input)
        prix = parse_float(prix_input)
        frais = parse_float(frais_input)
            
        if type_tx in ("Achat", "Vente") and not ticker_selected:
            st.error("Ticker requis pour Achat/Vente")
        elif quantite <= 0.0001:
            st.error("Quantité doit être > 0.0001")
        elif prix <= 0.0001 and type_tx not in ["Dépôt", "Retrait"]:
            st.error("Prix doit être > 0.0001")
        else:
            # 🔧 Correction ici
            if isinstance(st.session_state.df_transactions, pd.DataFrame) and not st.session_state.df_transactions.empty:
                df_hist = st.session_state.df_transactions.copy()
            else:
                df_hist = load_transactions_from_sheet()
            
            if df_hist.empty:
                df_hist = pd.DataFrame(columns=EXPECTED_COLS)
            
            engine = PortfolioEngine(df_hist)
            ticker = ticker_selected if ticker_selected else "CASH"
            date_tx = pd.to_datetime(date_input)
            transaction = None
            
            if type_tx == "Achat" and ticker != "CASH":
                is_valid_currency, currency_error = engine.validate_currency_consistency(ticker, profil, devise)
                if not is_valid_currency:
                    st.error(currency_error)
                else:
                    transaction = engine.prepare_achat_transaction(
                        ticker=ticker, profil=profil, quantite=quantite,
                        prix_achat=prix, frais=frais, date_achat=date_tx,
                        devise=devise, note=note, currency_manager=currency_manager
                    )
            
            elif type_tx == "Vente":
                transaction = engine.prepare_sale_transaction(
                    ticker=ticker, profil=profil, quantite=quantite,
                    prix_vente=prix, frais=frais, date_vente=date_tx,
                    devise=devise, note=note, currency_manager=currency_manager
                )
                if transaction is None:
                    st.error("Impossible de créer la vente (quantité insuffisante)")
            
            elif type_tx == "Dépôt":
                transaction = engine.prepare_depot_transaction(
                    profil=profil, montant=quantite if quantite > 0 else prix,
                    date_depot=date_tx, devise=devise, note=note, currency_manager=currency_manager
                )
            
            elif type_tx == "Retrait":
                transaction = engine.prepare_retrait_transaction(
                    profil=profil, montant=quantite if quantite > 0 else prix,
                    date_retrait=date_tx, devise=devise, note=note, currency_manager=currency_manager
                )
            
            elif type_tx == "Dividende":
                transaction = engine.prepare_dividende_transaction(
                    ticker=ticker, profil=profil, montant_brut=quantite,
                    retenue_source=frais, date_dividende=date_tx,
                    devise=devise, note=note, currency_manager=currency_manager
                )
            
            if transaction:
                if transaction["Ticker"] != "CASH":
                    transaction["Nom complet"] = get_ticker_full_name(transaction["Ticker"])
                else:
                    transaction["Nom complet"] = "CASH"
                df_new = pd.concat([df_hist, pd.DataFrame([transaction])], ignore_index=True)
                ok = save_transactions_to_sheet(df_new)
                if ok:
                    st.success(f"✅ {type_tx} enregistré : {transaction['Ticker']}")
                    if type_tx == "Vente":
                        st.info(f"📊 PRU_vente figé : {transaction['PRU_vente']:.2f} {devise}")
                        st.info(f"💰 PnL réalisé : {transaction['PnL réalisé (€/$)']:.2f} {devise}")
                    if transaction.get("Taux_change") and transaction["Taux_change"] != 1.0:
                        st.info(f"💱 Taux de change figé : {transaction['Taux_change']:.4f}")
                    st.session_state.df_transactions = load_transactions_from_sheet()  # <== recharge propre
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Erreur enregistrement")

    st.divider()
    st.subheader("📜 Historique des transactions")
    if st.session_state.df_transactions is not None and not st.session_state.df_transactions.empty:
        df_display = st.session_state.df_transactions.copy()
        df_display["Date_sort"] = pd.to_datetime(df_display["Date"], errors="coerce")
        df_display = df_display.sort_values(by="Date_sort", ascending=False).drop(columns=["Date_sort", "History_Log", "Taux_change", "Devise_reference"], errors='ignore')
        st.dataframe(df_display.head(100), use_container_width=True)
    else:
        st.info("Aucune transaction")

# -----------------------
# ONGLET 2 : Portefeuille avec conversion
# -----------------------
with tab2:
    st.header("📂 Portefeuille consolidé")
    
    if st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        st.info("Aucune transaction")
    else:
        devise_affichage = st.session_state.devise_affichage
        symbole = "€" if devise_affichage == "EUR" else "$"
        
        engine = PortfolioEngine(st.session_state.df_transactions)
        summary = engine.get_portfolio_summary_converted(
            target_currency=devise_affichage,
            currency_manager=currency_manager
        )
        positions = engine.get_positions_consolide()
        
        cache_info = currency_manager.get_cache_info()
        if cache_info["status"] != "Non initialisé":
            status_color = "🟢" if not cache_info["using_fallback"] else "🟠"
            st.caption(f"{status_color} {currency_manager.get_rate_display('EUR', 'USD')} | {cache_info['status']} (màj: {cache_info['age_minutes']}min)")
        
        st.subheader(f"Indicateurs clés ({devise_affichage})")
        k1, k2, k3, k4, k5 = st.columns(5)
        
        k1.metric("💵 Dépôts totaux", f"{summary['total_depots']:,.2f} {symbole}")
        k2.metric("💰 Liquidités", f"{summary['cash']:,.2f} {symbole}")
        
        if not positions.empty:
            tickers = positions["Ticker"].tolist()
            prices = fetch_last_close_batch(tickers)
            positions["Prix_actuel"] = positions["Ticker"].map(prices)
            
            positions["Valeur_origine"] = positions["Quantité"] * positions["Prix_actuel"]
            positions["Valeur_convertie"] = positions.apply(
                lambda row: currency_manager.convert(row["Valeur_origine"], row["Devise"], devise_affichage) 
                if row["Devise"] != devise_affichage and row["Prix_actuel"] else row["Valeur_origine"],
                axis=1
            )
            
            positions["PnL_latent"] = (positions["Prix_actuel"] - positions["PRU"]) * positions["Quantité"]
            positions["PnL_latent_%"] = ((positions["Prix_actuel"] - positions["PRU"]) / positions["PRU"] * 100).round(2)
            
            positions["Valeur_display"] = positions.apply(
                lambda row: f"{row['Valeur_origine']:,.2f} {row['Devise']}" +
                           (f" ({row['Valeur_convertie']:,.2f} {symbole})" if row['Devise'] != devise_affichage else ""),
                axis=1
            )
            
            total_valeur = positions["Valeur_convertie"].sum()
            total_pnl_latent = positions["PnL_latent"].sum()
        else:
            total_valeur = 0.0
            total_pnl_latent = 0.0
        
        k3.metric("📊 Valeur actifs", f"{total_valeur:,.2f} {symbole}")
        k4.metric("📈 PnL Latent", f"{total_pnl_latent:,.2f} {symbole}",
                 delta=f"{(total_pnl_latent/total_valeur*100):.2f}%" if total_valeur > 0 else "0%")
        k5.metric("✅ PnL Réalisé", f"{summary['pnl_realise_total']:,.2f} {symbole}")
        
        st.divider()
        
        if not positions.empty:
            st.subheader("Positions ouvertes")
            display_positions = positions[["Ticker","Nom complet", "Quantité", "PRU", "Devise", "Prix_actuel", "Valeur_display", "PnL_latent", "PnL_latent_%"]].copy()
            display_positions.columns = ["Ticker", "Qté", "PRU", "Dev", "Prix actuel", "Valeur", "PnL €/$", "PnL %"]
            display_positions = display_positions.sort_values("PnL €/$", ascending=False)
            st.dataframe(display_positions, use_container_width=True, hide_index=True)
            
            fig = px.pie(positions.dropna(subset=["Valeur_convertie"]), values="Valeur_convertie", names="Ticker",
                        title=f"Répartition du portefeuille ({devise_affichage})")
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.bar(positions.dropna(subset=["PnL_latent"]), x="Ticker", y="PnL_latent",
                         title="PnL Latent par position", color="PnL_latent",
                         color_continuous_scale=["red", "gray", "green"])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune position ouverte")
        
        df_ventes = st.session_state.df_transactions[st.session_state.df_transactions["Type"] == "Vente"].copy()
        
        if not df_ventes.empty:
            df_ventes["Date_sort"] = pd.to_datetime(df_ventes["Date"])
            df_ventes = df_ventes.sort_values("Date_sort")
            df_ventes["PnL_cumule"] = df_ventes["PnL réalisé (€/$)"].cumsum()
            
            fig3 = px.line(df_ventes, x="Date_sort", y="PnL_cumule",
                          title="PnL Réalisé Cumulatif",
                          labels={"Date_sort": "Date", "PnL_cumule": "PnL Cumulé"})
            st.plotly_chart(fig3, use_container_width=True)
        
        with st.expander("🔍 Détails des calculs"):
            st.write(f"**Résumé financier ({devise_affichage}):**")
            st.json(summary)

# -----------------------
# ONGLET 3 : Répartition par Profil
# -----------------------
with tab3:
    st.header("📊 Répartition portefeuilles individuels")
    
    if st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        st.info("Aucune transaction")
    else:
        devise_affichage = st.session_state.devise_affichage
        symbole = "€" if devise_affichage == "EUR" else "$"
        
        profils = sorted(st.session_state.df_transactions["Profil"].unique())
        cols = st.columns(len(profils))
        
        for i, profil in enumerate(profils):
            with cols[i]:
                st.subheader(f"{profil}")
                
                df_profil = st.session_state.df_transactions[st.session_state.df_transactions["Profil"] == profil]
                
                engine_profil = PortfolioEngine(df_profil)
                summary_profil = engine_profil.get_portfolio_summary_converted(
                    profil=profil, target_currency=devise_affichage, currency_manager=currency_manager
                )
                positions_profil = engine_profil.get_positions(profil=profil)
                
                if not positions_profil.empty:
                    tickers_profil = positions_profil["Ticker"].tolist()
                    prices_profil = fetch_last_close_batch(tickers_profil)
                    positions_profil["Prix_actuel"] = positions_profil["Ticker"].map(prices_profil)
                    positions_profil["Valeur_origine"] = positions_profil["Quantité"] * positions_profil["Prix_actuel"]
                    positions_profil["Valeur_convertie"] = positions_profil.apply(
                        lambda row: currency_manager.convert(row["Valeur_origine"], row["Devise"], devise_affichage)
                        if row["Devise"] != devise_affichage and row["Prix_actuel"] else row["Valeur_origine"],
                        axis=1
                    )
                    positions_profil["PnL_latent"] = (positions_profil["Prix_actuel"] - positions_profil["PRU"]) * positions_profil["Quantité"]
                    
                    total_valeur_profil = positions_profil["Valeur_convertie"].sum()
                    total_pnl_latent_profil = positions_profil["PnL_latent"].sum()
                else:
                    total_valeur_profil = 0.0
                    total_pnl_latent_profil = 0.0
                
                row1_col1, row1_col2 = st.columns(2)
                row2_col1, row2_col2 = st.columns(2)
                row3_col1, row3_col2 = st.columns(2)
                
                row1_col1.metric("💵 Dépôts", f"{summary_profil['total_depots']:,.0f} {symbole}")
                row1_col2.metric("💰 Liquidités", f"{summary_profil['cash']:,.0f} {symbole}")
                row2_col1.metric("📊 Valeur actifs", f"{total_valeur_profil:,.0f} {symbole}")
                row2_col2.metric("📈 PnL Latent", f"{total_pnl_latent_profil:,.0f} {symbole}")
                row3_col1.metric("✅ PnL Réalisé", f"{summary_profil['pnl_realise_total']:,.0f} {symbole}")
                row3_col2.metric("💎 Total", f"{summary_profil['cash'] + total_valeur_profil:,.0f} {symbole}")
                
                if not positions_profil.empty:
                    fig_profil = px.pie(
                        positions_profil.dropna(subset=["Valeur_convertie"]),
                        values="Valeur_convertie", 
                        names="Nom complet",
                        title=f"Répartition {profil}"
                    )
                    st.plotly_chart(fig_profil, use_container_width=True)
                    
                    st.markdown("**Top 5 positions:**")
                    top5 = positions_profil.nlargest(5, "Valeur_convertie")[["Ticker", "Nom complet", "Quantité", "Valeur_convertie", "PnL_latent"]]
                    top5.columns = ["Ticker", "Nom complet", "Qté", f"Valeur {symbole}", f"PnL {symbole}"]
                    st.dataframe(top5, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune position")
# ---- Séparateur visuel entre profils ----
            if i < len(profils) - 1:
                st.markdown(
                    "<div style='height:3px; background:linear-gradient(to right, #ccc, #888, #ccc); margin:20px 0; border-radius:3px;'></div>",
                    unsafe_allow_html=True
                )


# -----------------------
# ONGLET 4 : Calendrier
# -----------------------
with tab4:
    st.header("📅 Calendrier économique")
    st.info("Fonctionnalité à venir - Phase 2")
    
    st.subheader("💰 Dividendes reçus")
    
    if st.session_state.df_transactions is not None:
        df_div = st.session_state.df_transactions[st.session_state.df_transactions["Type"] == "Dividende"].copy()
        
        if not df_div.empty:
            df_div["Date_sort"] = pd.to_datetime(df_div["Date"])
            df_div = df_div.sort_values("Date_sort", ascending=False)
            
            display_div = df_div[["Date", "Profil", "Ticker", "PnL réalisé (€/$)", "Devise", "Note"]].head(20)
            st.dataframe(display_div, use_container_width=True, hide_index=True)
            
            div_by_ticker = df_div.groupby("Ticker")["PnL réalisé (€/$)"].sum().sort_values(ascending=False)
            
            fig_div = px.bar(x=div_by_ticker.index, y=div_by_ticker.values,
                            title="Total dividendes par ticker",
                            labels={"x": "Ticker", "y": "Dividendes nets"})
            st.plotly_chart(fig_div, use_container_width=True)
        else:
            st.info("Aucun dividende enregistré")

# -----------------------
# SIDEBAR : Infos & Actions
# -----------------------
with st.sidebar:
    st.title("⚙️ Paramètres")
    
    st.divider()
    
    st.subheader("📊 Statistiques")
    if st.session_state.df_transactions is not None:
        nb_tx = len(st.session_state.df_transactions)
        nb_profils = st.session_state.df_transactions["Profil"].nunique()
        nb_tickers = st.session_state.df_transactions[st.session_state.df_transactions["Ticker"] != "CASH"]["Ticker"].nunique()
        
        st.metric("Transactions", nb_tx)
        st.metric("Profils", nb_profils)
        st.metric("Titres uniques", nb_tickers)

    st.divider()
    
    st.subheader("🔄 Actions")
    
    if st.button("♻️ Rafraîchir données", use_container_width=True):
        st.cache_data.clear()
        st.session_state.df_transactions = load_transactions_from_sheet()
        st.session_state.currency_manager.clear_cache()
        st.success("✅ Données rechargées")
        st.rerun()
    
    if st.button("📥 Exporter CSV", use_container_width=True):
        if st.session_state.df_transactions is not None:
            csv = st.session_state.df_transactions.to_csv(index=False)
            st.download_button(
                label="💾 Télécharger",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    st.subheader("ℹ️ Informations")
    st.caption("Dashboard Portefeuille V2.1")
    st.caption("Multi-devises EUR/USD")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# -----------------------
# FOOTER
# -----------------------
st.divider()
st.caption("© 2025 FBM Fintech - Dashboard Portefeuille V2.1 | Multi-devises EUR/USD | Données temps réel via yfinance")

