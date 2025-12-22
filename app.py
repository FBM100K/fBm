"""
Dashboard Portefeuille V3.0 - Multi-devises
"""

import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date
from google.oauth2.service_account import Credentials
import requests
import time

# Import des moteurs
from portfolio_engine import PortfolioEngine
from currency_manager import CurrencyManager
from utils import (
    format_positions_display,
    format_currency_value,
    get_color_pnl,
    validate_dataframe_columns,
    safe_divide
)

# -----------------------
# Configuration
# -----------------------
st.set_page_config(
    page_title="Dashboard Portefeuille V3.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='text-align: left; font-size: 32px;'>📊 Dashboard Portefeuille - FBM V3.0</h1>",
    unsafe_allow_html=True
)

# Constantes
SHEET_NAME = "transactions_dashboard"
EXPECTED_COLS = [
    "Date", "Profil", "Type", "Ticker", "Nom complet",
    "Quantité", "Prix_unitaire", "PRU_vente", "Devise",
    "Taux_change", "Devise_reference", "Frais (€/$)",
    "PnL réalisé (€/$)", "PnL réalisé (%)", "Note", "History_Log"
]
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

# -----------------------
# Initialisation Session State
# -----------------------
if "devise_affichage" not in st.session_state:
    st.session_state.devise_affichage = "EUR"

if "ticker_cache" not in st.session_state:
    st.session_state.ticker_cache = {}

if "suggestion_cache" not in st.session_state:
    st.session_state.suggestion_cache = {}

if "currency_manager" not in st.session_state:
    st.session_state.currency_manager = CurrencyManager()

if "df_transactions" not in st.session_state:
    st.session_state.df_transactions = None

if "last_devise_affichage" not in st.session_state:
    st.session_state.last_devise_affichage = st.session_state.devise_affichage

# Références
currency_manager = st.session_state.currency_manager

# -----------------------
# Google Sheets Authentication
# -----------------------
def init_google_sheets():
    """Initialise la connexion Google Sheets."""
    try:
        creds_info = st.secrets["google_service_account"]
        credentials = Credentials.from_service_account_info(creds_info, scopes=SCOPE)
        gc_client = gspread.authorize(credentials)
        sh = gc_client.open(SHEET_NAME)
        sheet = sh.sheet1
        return sheet, sh, gc_client
    except Exception as e:
        st.error("❌ Erreur d'authentification Google Sheets")
        st.exception(e)
        return None, None, None

sheet, sh, gc_client = init_google_sheets()

# -----------------------
# Helper Functions
# -----------------------
def parse_float(val):
    """Parse une valeur en float de manière sécurisée."""
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

@st.cache_data(ttl=60, show_spinner=False) 
def load_transactions_from_sheet():
    if sheet is None:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    try:
        values = sheet.get_all_values()
        
        if len(values) <= 1:
            return pd.DataFrame(columns=EXPECTED_COLS)
        
        # Conversion en DataFrame avec header
        df = pd.DataFrame(values[1:], columns=values[0])
        
        # Ajout colonnes manquantes
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None
        
        # Normalisation dates vectorisée
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="%Y-%m-%d").dt.date
        
        # Normalisation numériques en bloc
        numeric_cols = [
            "Quantité", "Prix_unitaire", "Frais (€/$)",
            "PnL réalisé (€/$)", "PnL réalisé (%)", "PRU_vente", "Taux_change"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # Nettoyage et conversion en une seule passe
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .replace(["", "None", "nan", "NaN"], "0")
                )
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Valeurs par défaut
        df["Devise"] = df["Devise"].fillna("EUR")
        df["Devise_reference"] = df["Devise_reference"].fillna("EUR")
        df["Profil"] = df["Profil"].fillna("Gas")
        df["Type"] = df["Type"].fillna("Achat")
        
        # Réorganisation colonnes
        df = df.reindex(columns=EXPECTED_COLS)
        return df
    
    except Exception as e:
        st.error(f"❌ Erreur lecture Google Sheet: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

def save_transactions_to_sheet(df: pd.DataFrame) -> bool:
    if sheet is None or sh is None:
        st.error("❌ Pas de connexion à Google Sheets")
        return False
    
    if df.empty:
        st.error("❌ Tentative de sauvegarde d'un DataFrame vide")
        return False
    
    df_out = df.copy()
    
    # Formatage dates
    if "Date" in df_out.columns:
        df_out["Date"] = df_out["Date"].apply(
            lambda d: d.strftime("%Y-%m-%d")
            if pd.notna(d) and isinstance(d, (date, pd.Timestamp))
            else (d if d else "")
        )
    
    # Vérification colonnes
    for c in EXPECTED_COLS:
        if c not in df_out.columns:
            df_out[c] = ""
    
    values = [EXPECTED_COLS] + df_out[EXPECTED_COLS].fillna("").astype(str).values.tolist()
    
    try:
        # Création backup
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
        
        except Exception as e:
            st.warning(f"⚠️ Backup non créé : {e}")
        sheet.clear()
        sheet.update("A1", values, value_input_option="USER_ENTERED")
        
        # Rotation backups (max 5)
        try:
            backups = [w for w in sh.worksheets() if w.title.startswith("backup_")]
            if len(backups) > 5:
                backups_sorted = sorted(backups, key=lambda w: w.title, reverse=True)
                for old in backups_sorted[5:]:
                    sh.del_worksheet(old)
        except Exception as e:
            st.warning(f"⚠️ Rotation backup non appliquée : {e}")
        
        return True
    
    except Exception as e:
        st.error(f"❌ Erreur écriture : {e}")
        return False

@st.cache_data(ttl=60, show_spinner=False)
def fetch_last_close_batch(tickers: list) -> dict:
    if not tickers:
        return {}

    # Nettoyage, dédoublonnage + normalisation Euronext: .PAR -> .PA
    tickers_clean = sorted({
        str(t).strip().upper().replace(".PAR", ".PA")
        for t in tickers
        if t and str(t).strip().upper() != "CASH"
    })
    if not tickers_clean:
        return {}

    prices = {}

    # 1) Batch rapide 1d
    try:
        data = yf.download(
            tickers_clean,
            period="1d",
            progress=False,
            threads=True,
            group_by="ticker",
            auto_adjust=False,
            timeout=8,
        )

        if isinstance(data.columns, pd.MultiIndex):
            # colonnes: (TICKER, OHLCV)
            lvl0 = set(data.columns.get_level_values(0))
            for t in tickers_clean:
                if t in lvl0:
                    ser = data[t]["Close"].dropna()
                    if not ser.empty:
                        prices[t] = float(ser.iloc[-1])
        else:
            # cas 1 seul ticker
            if "Close" in data and len(tickers_clean) == 1:
                ser = data["Close"].dropna()
                if not ser.empty:
                    prices[tickers_clean[0]] = float(ser.iloc[-1])

    except Exception:
        pass

    # 2) Fallback 5d pour ceux manquants
    for t in tickers_clean:
        if prices.get(t, 0.0) > 0:
            continue
        try:
            fb = yf.download(
                t,
                period="5d",
                progress=False,
                auto_adjust=False,
                timeout=8,
            )
            if "Close" in fb:
                ser_fb = fb["Close"].dropna()
                prices[t] = float(ser_fb.iloc[-1]) if not ser_fb.empty else 0.0
            else:
                prices[t] = 0.0
        except Exception:
            prices[t] = 0.0

    return prices

# -----------------------
# Chargement initial données avec indicateurs visuels
# -----------------------
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = False

if sheet is not None:
    if (
        "df_transactions" not in st.session_state
        or st.session_state.df_transactions is None
        or st.session_state.df_transactions.empty
    ):
        # Affichage barre de progression
        if not st.session_state.app_initialized:
            # Créer conteneur pour progression
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### Initialisation du Dashboard")
                progress_bar = st.progress(0, text="Connexion en cours...")
                status_text = st.empty()
                
                # Étape 1 : Connexion établie
                status_text.info("Connexion à Google Sheets établie")
                progress_bar.progress(25, text="Téléchargement des données...")
                
                # Étape 2 : Chargement données
                with st.spinner("Chargement des transactions..."):
                    df_loaded = load_transactions_from_sheet()
                
                progress_bar.progress(60, text="Traitement des données...")
                
                if df_loaded is not None and not df_loaded.empty:
                    st.session_state.df_transactions = df_loaded
                    nb_transactions = len(df_loaded)
                    
                    # Étape 3 : Initialisation currency manager
                    status_text.info("Initialisation des taux de change...")
                    progress_bar.progress(80, text="Finalisation...")
                    
                    if "currency_manager" not in st.session_state:
                        st.session_state.currency_manager = CurrencyManager()
                    
                    # Étape 4 : Terminé
                    progress_bar.progress(100, text="Chargement terminé !")
                    status_text.success(f"✅ {nb_transactions} transactions chargées avec succès")
                    
                    # Marquer comme initialisé
                    st.session_state.app_initialized = True
                    
                    # Nettoyer les indicateurs et recharger
                    st.rerun()
                else:
                    progress_bar.progress(100, text="Aucune donnée")
                    status_text.warning("⚠️ Aucune donnée chargée (sheet non accessible)")
                    st.session_state.app_initialized = True
        else:
            # Chargement silencieux (déjà initialisé)
            with st.spinner("🔄 Rechargement des données..."):
                df_loaded = load_transactions_from_sheet()
                if df_loaded is not None and not df_loaded.empty:
                    st.session_state.df_transactions = df_loaded
else:
    st.error("❌ Impossible de se connecter à Google Sheets - vérifiez st.secrets")
    st.info("💡 Vérifiez que le fichier `.streamlit/secrets.toml` contient les bonnes credentials")

# -----------------------
# Header avec indicateurs et toggle devise
# -----------------------
col_title, col_currency = st.columns([3, 1])

with col_title:
    st.divider()
    # Indicateur PRU_vente migration
    if st.session_state.df_transactions is not None:
        ventes = st.session_state.df_transactions[
            st.session_state.df_transactions["Type"] == "Vente"
        ]
        if not ventes.empty:
            ventes_avec_pru = ventes[
                ventes["PRU_vente"].notna() & (ventes["PRU_vente"] > 0)
            ]
            pct_migre = len(ventes_avec_pru) / len(ventes) * 100

with col_currency:
    # Récupération de la devise actuelle
    current_devise = st.session_state.devise_affichage
    # Calcul de l'index correct (0=EUR, 1=USD)
    current_index = 0 if current_devise == "EUR" else 1
    # Widget radio
    selected_devise = st.radio(
        "💱 Devise d'affichage",
        options=["EUR", "USD"],
        index=current_index,
        horizontal=True,
        key="currency_toggle",
        help="Basculez entre Euro et Dollar pour l'affichage des montants"
    )
    st.session_state.devise_affichage = selected_devise
    # Toast auto-disparaissant lors d'un changement de devise
    if selected_devise != st.session_state.last_devise_affichage:
        cache_info = currency_manager.get_cache_info()
        if cache_info["status"] != "Non initialisé":
            icon = "⚠️" if cache_info["using_fallback"] else "✅"
            msg = f"{icon} {cache_info['status']} — Mise à jour: {cache_info['last_update']}"
            st.toast(msg, icon="💱")
        st.session_state.last_devise_affichage = selected_devise


# -----------------------
# Recherche Ticker - Fonctions
# -----------------------
ALPHA_VANTAGE_API_KEY = None
try:
    ALPHA_VANTAGE_API_KEY = st.secrets["alpha_vantage"]["api_key"]
except:
    ALPHA_VANTAGE_API_KEY = None


@st.cache_data(ttl=1600)
def get_alpha_vantage_suggestions(query: str) -> list:
    """
    Recherche des tickers sur Alpha Vantage avec cache.
    Args:
        query: Terme de recherche (min 2 caractères)
    Returns:
        Liste de suggestions formatées ["TICKER — Nom (Région)"]
    """
    if not query or len(query.strip()) < 2:
        return []
    
    if not ALPHA_VANTAGE_API_KEY:
        st.warning("⚠️ Clé API Alpha Vantage manquante")
        return []
    
    # Cache local session
    if "suggestion_cache" not in st.session_state:
        st.session_state.suggestion_cache = {}
    
    query_lower = query.strip().lower()
    if query_lower in st.session_state.suggestion_cache:
        return st.session_state.suggestion_cache[query_lower]
    
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
            return []
        
        suggestions = []
        for m in matches:
            symbol = m.get("1. symbol", "")
            name = m.get("2. name", "")
            region = m.get("4. region", "")
            if symbol and name:
                suggestions.append(f"{symbol} — {name} ({region})")
        
        # Cache local
        st.session_state.suggestion_cache[query_lower] = suggestions[:15]
        return suggestions[:15]
    
    except Exception as e:
        st.error(f"❌ Erreur Alpha Vantage : {e}")
        return []


@st.cache_data(ttl=1600)
def get_ticker_full_name_from_api(ticker: str) -> str:
    """
    Requête Alpha Vantage pour obtenir le nom complet.
    
    Args:
        ticker: Code ticker
    
    Returns:
        Nom formaté "Nom (Région)" ou ticker si échec
    """
    if not ALPHA_VANTAGE_API_KEY or not ticker:
        return ticker
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
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


def get_ticker_full_name(ticker: str) -> str:
    ticker = ticker.upper().strip()
    cache = st.session_state.ticker_cache
    
    if ticker in cache:
        return cache[ticker]
    
    # Appel API si pas en cache
    full_name = get_ticker_full_name_from_api(ticker)
    cache[ticker] = full_name
    st.session_state.ticker_cache = cache
    
    return full_name

# -----------------------
# ONGLET 1 : Transactions - VERSION FINALE V3.1
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Transactions",
    "📂 Portefeuille",
    "📊 Répartition",
    "📅 Calendrier"
])

with tab1:
    st.markdown(
        """
        <style>
        /* ✅ SUPPRESSION de l'effet de flou blanchâtre lors du chargement */
        [data-testid="stAppViewContainer"] > section {
            background: transparent !important;
        }
        
        /* Désactiver l'overlay de rechargement qui cause le flou */
        .stSpinner > div {
            background: transparent !important;
        }
        
        /* Pas de backdrop flou */
        [data-testid="stStatusWidget"] {
            background: transparent !important;
        }
        
        /* Réduire espacement entre champs pour layout compact */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stDateInput > div > div > input,
        .stTextArea > div > div > textarea {
            padding: 8px 12px !important;
            font-size: 14px !important;
        }
        
        /* Labels plus compacts */
        .stTextInput label, .stSelectbox label, .stDateInput label, .stTextArea label {
            font-size: 13px !important;
            margin-bottom: 4px !important;
        }
        
        /* Formulaire dans expander avec bordure discrète */
        [data-testid="stExpander"] {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #fafafa;
            margin-bottom: 20px;
        }
        
        /* Titre expander plus clair */
        [data-testid="stExpander"] summary {
            font-size: 16px;
            font-weight: 600;
            color: #1f1f1f;
        }
        
        /* Bouton principal plus visible */
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
            font-weight: 600;
            font-size: 16px;
            padding: 12px 24px;
            border-radius: 6px;
            transition: all 0.3s ease;
        }
        
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }
        
        /* Bouton secondaire discret */
        .stButton > button:not([kind="primary"]) {
            background: #f5f5f5;
            color: #666;
            border: 1px solid #ddd;
        }
        
        /* Messages d'erreur plus visibles */
        .stAlert[data-baseweb="notification"] {
            border-left: 5px solid #ff4b4b;
            background-color: #fff5f5;
            padding: 16px;
            border-radius: 8px;
        }
        
        /* Masquer footer Streamlit qui gêne */
        footer {visibility: hidden;}
        
        /* Scroll fluide */
        html {
            scroll-behavior: smooth;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.header("Transactions")
    
    # ============================================
    # FORMULAIRE DANS EXPANDER
    # ============================================
    with st.expander("➕ Ajouter une transaction", expanded=True):
        
        # --- Paramètres généraux (layout compact sur 1 ligne) ---
        col_profil, col_type = st.columns(2)
        with col_profil:
            profil = st.selectbox(
                "Portefeuille / Profil",
                ["Gas", "Marc"],
                index=0,
                help="Sélectionnez le propriétaire de la transaction"
            )
        with col_type:
            type_tx = st.selectbox(
                "Type",
                ["Achat", "Vente", "Dépôt", "Retrait", "Dividende"],
                index=0,
                help="Type d'opération à enregistrer"
            )
        
        # --- Initialisation états recherche ---
        if "ticker_query" not in st.session_state:
            st.session_state.ticker_query = ""
        if "ticker_suggestions" not in st.session_state:
            st.session_state.ticker_suggestions = []
        if "ticker_selected" not in st.session_state:
            st.session_state.ticker_selected = ""
        
        # --- Recherche de titre (si Achat/Vente/Dividende) ---
        if type_tx in ["Achat", "Vente", "Dividende"]:
            st.markdown("#### 🔍 Recherche de titre")
            
            col_rech1, col_rech2 = st.columns([5, 1])
            with col_rech1:
                query = st.text_input(
                    "Entrez un nom ou ticker :",
                    value=st.session_state.ticker_query,
                    label_visibility="collapsed",
                    placeholder="Ex: AAPL, Tesla, LVMH...",
                    help="Saisissez au moins 2 caractères"
                )
            with col_rech2:
                if st.button("🔎", use_container_width=True, help="Lancer la recherche"):
                    st.session_state.ticker_query = query
                    if query:
                        with st.spinner("🔍 Recherche en cours..."):
                            suggestions = get_alpha_vantage_suggestions(query)
                            st.session_state.ticker_suggestions = suggestions
                            if not suggestions:
                                st.warning("⚠️ Aucun résultat")
            
            # Affichage résultats
            if st.session_state.ticker_suggestions:
                sel = st.selectbox(
                    "Choisissez l'action :",
                    st.session_state.ticker_suggestions,
                    key="ticker_selectbox"
                )
                if sel:
                    ticker_extracted = sel.split(" — ")[0]
                    st.session_state.ticker_selected = ticker_extracted
                    ticker_extracted = sel.split(" — ")[0].strip().upper()
                # Normalisation simple Euronext Paris : .PAR -> .PA (yfinance)
                if ticker_extracted.endswith(".PAR"):
                    ticker_extracted = ticker_extracted[:-4] + ".PA"

                st.session_state.ticker_selected = ticker_extracted

            
            # Confirmation ticker sélectionné
            if st.session_state.ticker_selected:
                st.success(f"✅ Titre sélectionné : **{st.session_state.ticker_selected}**")
        
        ticker_selected = st.session_state.ticker_selected or None
        
        # --- Détails de la transaction (layout compact 2x2) ---
        st.markdown("#### 📝 Détails de la transaction")
        
        col1, col2 = st.columns(2)
        with col1:
            quantite_input = st.text_input(
                "Quantité",
                "0",
                help="Nombre d'actions (ou montant pour Dépôt/Retrait)"
            )
            prix_default = "1.0" if type_tx in ["Dépôt", "Retrait"] else "0"
            prix_input = st.text_input(
                "Prix unitaire (€/$)",
                prix_default,
                help="Prix par action (ou 1.0 pour Dépôt/Retrait)"
            )
        with col2:
            frais_input = st.text_input(
                "Frais (€/$)",
                "0",
                help="Frais de transaction ou courtage"
            )
            date_input = st.date_input(
                "Date",
                value=datetime.today(),
                max_value=datetime.today(),
                help="Date de la transaction (ne peut pas être future)"
            )
        
        devise = st.selectbox(
            "Devise",
            ["EUR", "USD"],
            index=0,
            help="Devise dans laquelle la transaction est effectuée"
        )
        
        note = st.text_area(
            "Note (optionnel)",
            "",
            max_chars=250,
            height=70,
            placeholder="Ajoutez un commentaire sur cette transaction...",
            help="Commentaire libre (max 250 caractères)"
        )
        
        # ============================================
        # INDICATEUR DE COMPLÉTION
        # ============================================
        quantite = parse_float(quantite_input)
        prix = parse_float(prix_input)
        
        # Calcul complétion
        total_fields = 4
        filled_fields = 2
        
        if type_tx in ["Achat", "Vente", "Dividende"]:
            total_fields += 1
            if ticker_selected:
                filled_fields += 1
        
        if quantite > 0:
            filled_fields += 1
        
        if prix > 0:
            filled_fields += 1
        
        pct_complete = int((filled_fields / total_fields) * 100)
        
        if pct_complete < 50:
            emoji_status = "🔴"
        elif pct_complete < 80:
            emoji_status = "🟡"
        else:
            emoji_status = "🟢"
        
        # Affichage discret de la complétion
        st.markdown("---")
        col_progress, col_spacer = st.columns([3, 1])
        with col_progress:
            st.progress(pct_complete / 100)
            st.caption(
                f"{emoji_status} Formulaire complété à **{pct_complete}%**",
                help="Remplissez tous les champs obligatoires"
            )
        
        # ============================================
        # BOUTONS (Ajouter + Effacer)
        # ============================================
        st.markdown("---")
        
        col_submit, col_clear = st.columns([4, 1])
        
        with col_submit:
            submit_btn = st.button(
                "➕ Ajouter Transaction",
                type="primary",
                use_container_width=True,
                disabled=(pct_complete < 80),
                help="Enregistrer la transaction dans le portefeuille"
            )
        
        with col_clear:
            if st.button(
                "🗑️ Effacer",
                use_container_width=True,
                help="Réinitialiser tous les champs"
            ):
                st.session_state.ticker_selected = ""
                st.session_state.ticker_suggestions = []
                st.session_state.ticker_query = ""
                st.success("✅ Formulaire réinitialisé")
                st.rerun()
        
        # TRAITEMENT DU FORMULAIRE
        # ============================================
        if submit_btn:
            quantite = parse_float(quantite_input)
            prix = parse_float(prix_input)
            frais = parse_float(frais_input)
            errors = []
            
            # Validation 1 : Ticker requis pour Achat/Vente/Dividende
            if type_tx in ("Achat", "Vente", "Dividende") and not ticker_selected:
                errors.append("❌ **Ticker requis** : Veuillez rechercher et sélectionner une action")
            
            # Validation 2 : Quantité strictement positive (sauf Retrait)
            if type_tx not in ["Retrait"]:
                if quantite <= 0.0001:
                    errors.append(f"❌ **Quantité invalide** : `{quantite:.4f}` - Doit être > 0.0001")
            else:
                if quantite <= 0.0001 and prix <= 0.0001:
                    errors.append("❌ **Montant requis** : Indiquez le montant du retrait")
            
            # Validation 3 : Prix unitaire strictement positif
            if type_tx == "Achat":
                if prix <= 0.0001:
                    errors.append(f"❌ **Prix d'achat invalide** : `{prix:.4f}` - Doit être > 0.0001")
            
            elif type_tx == "Vente":
                if prix <= 0.0001:
                    errors.append(f"❌ **Prix de vente invalide** : `{prix:.4f}` - Doit être > 0.0001")
            
            elif type_tx == "Dépôt":
                if quantite <= 0.0001 and prix <= 1.0:
                    errors.append("❌ **Montant du dépôt invalide** : Indiquez le montant")
            
            elif type_tx == "Dividende":
                if quantite <= 0.0001:
                    errors.append(f"❌ **Montant brut dividende invalide** : `{quantite:.4f}` - Doit être > 0")
            
            # Validation 4 : Frais ne peuvent pas être négatifs
            if frais < 0:
                errors.append(f"❌ **Frais invalides** : `{frais:.2f}` - Ne peuvent pas être négatifs")
            
            # Validation 5 : Date ne peut pas être dans le futur
            date_limite = datetime.today().date()
            if date_input > date_limite:
                errors.append(f"❌ **Date invalide** : `{date_input}` - Ne peut pas être dans le futur")
            
            if errors:
                # Ancre pour scroll
                st.markdown('<div id="error-anchor"></div>', unsafe_allow_html=True)
                
                st.error("### ⚠️ Erreurs de validation\n\n" + "\n\n".join(errors))
                
                # Animation du bouton
                st.markdown(
                    """
                    <style>
                    .stButton button[kind="primary"] {
                        border: 2px solid #ff4b4b !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                # Auto-scroll vers erreur
                st.markdown(
                    """
                    <script>
                    setTimeout(function() {
                        document.getElementById('error-anchor').scrollIntoView({
                            behavior: 'smooth',
                            block: 'center'
                        });
                    }, 100);
                    </script>
                    """,
                    unsafe_allow_html=True
                )
            
            else:
                # ✅ ENREGISTREMENT
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
                
                # --- Préparation transaction selon type ---
                if type_tx == "Achat" and ticker != "CASH":
                    is_valid_currency, currency_error = engine.validate_currency_consistency(
                        ticker, profil, devise
                    )
                    if not is_valid_currency:
                        st.error(currency_error)
                    else:
                        transaction = engine.prepare_achat_transaction(
                            ticker=ticker,
                            profil=profil,
                            quantite=quantite,
                            prix_achat=prix,
                            frais=frais,
                            date_achat=date_tx,
                            devise=devise,
                            note=note,
                            currency_manager=currency_manager
                        )
                
                elif type_tx == "Vente":
                    transaction = engine.prepare_sale_transaction(
                        ticker=ticker,
                        profil=profil,
                        quantite=quantite,
                        prix_vente=prix,
                        frais=frais,
                        date_vente=date_tx,
                        devise=devise,
                        note=note,
                        currency_manager=currency_manager
                    )
                    if transaction is None:
                        st.error("❌ Impossible de créer la vente (quantité insuffisante)")
                
                elif type_tx == "Dépôt":
                    transaction = engine.prepare_depot_transaction(
                        profil=profil,
                        montant=quantite if quantite > 0 else prix,
                        date_depot=date_tx,
                        devise=devise,
                        note=note,
                        currency_manager=currency_manager
                    )
                
                elif type_tx == "Retrait":
                    transaction = engine.prepare_retrait_transaction(
                        profil=profil,
                        montant=quantite if quantite > 0 else prix,
                        date_retrait=date_tx,
                        devise=devise,
                        note=note,
                        currency_manager=currency_manager
                    )
                
                elif type_tx == "Dividende":
                    transaction = engine.prepare_dividende_transaction(
                        ticker=ticker,
                        profil=profil,
                        montant_brut=quantite,
                        retenue_source=frais,
                        date_dividende=date_tx,
                        devise=devise,
                        note=note,
                        currency_manager=currency_manager
                    )
                
                # --- Enregistrement ---
                if transaction:
                    if transaction["Ticker"] != "CASH":
                        transaction["Nom complet"] = get_ticker_full_name(transaction["Ticker"])
                    else:
                        transaction["Nom complet"] = "CASH"
                    
                    df_new = pd.concat([df_hist, pd.DataFrame([transaction])], ignore_index=True)
                    
                    ok = save_transactions_to_sheet(df_new)
                    if ok:
                        st.success(f"✅ **{type_tx} enregistré** : {transaction['Ticker']}")
                        
                        if type_tx == "Vente":
                            st.info(f"📊 PRU_vente figé : {transaction['PRU_vente']:.2f} {devise}")
                            st.info(f"💰 PnL réalisé : {transaction['PnL réalisé (€/$)']:.2f} {devise}")
                        
                        if transaction.get("Taux_change") and transaction["Taux_change"] != 1.0:
                            st.info(f"💱 Taux de change figé : {transaction['Taux_change']:.4f}")
                        
                        st.session_state.df_transactions = load_transactions_from_sheet()
                        st.cache_data.clear()
                        
                        # Scroll vers le haut après succès
                        st.markdown(
                            """
                            <script>
                            window.scrollTo({top: 0, behavior: 'smooth'});
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.rerun()
                    else:
                        st.error("❌ Erreur enregistrement")

    # HISTORIQUE 
    # ============================================
    st.divider()
    st.subheader("📜 Historique des transactions")
    
    if st.session_state.df_transactions is not None and not st.session_state.df_transactions.empty:
        df_display = st.session_state.df_transactions.copy()
        df_display["Date_sort"] = pd.to_datetime(df_display["Date"], errors="coerce")
        df_display = df_display.sort_values(by="Date_sort", ascending=False)
        
        # Colonnes à afficher
        cols_to_show = [
            "Date", "Type", "Ticker", "Nom complet", "Profil",
            "Quantité", "Prix_unitaire", "Devise", "Frais (€/$)",
            "PnL réalisé (€/$)", "Note"
        ]
        df_display = df_display[[c for c in cols_to_show if c in df_display.columns]]
        
        st.dataframe(df_display.head(100), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Aucune transaction enregistrée")

# -----------------------
# ONGLET 2 : Portefeuille Consolidé - BLOC CORRIGÉ
# -----------------------
with tab2:
    st.header("Portefeuille consolidé")
    
    if st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        st.info("ℹ️ Aucune transaction")
    else:
        devise_affichage = st.session_state.devise_affichage
        symbole = "€" if devise_affichage == "EUR" else "$"
        
        # Calculs résumé et positions
        engine = PortfolioEngine(st.session_state.df_transactions)
        summary = engine.get_portfolio_summary_converted(
            target_currency=devise_affichage,
            currency_manager=currency_manager
        )
        positions = engine.get_positions_consolide()  # ✅ V3 corrigé
        
        # --- Indicateur taux de change ---
        cache_info = currency_manager.get_cache_info()
        if cache_info["status"] != "Non initialisé":
            status_color = "🟢" if not cache_info["using_fallback"] else "🟠"
            st.caption(
                f"{status_color} {currency_manager.get_rate_display('EUR', 'USD')} | "
                f"{cache_info['status']} (mà j: {cache_info['age_minutes']}min)"
            )
        
        # --- Indicateurs clés ---
        st.subheader(f"Indicateurs clés ({devise_affichage})")
        k1, k2, k3, k4, k5 = st.columns(5)
        
        k1.metric("💵 Dépôts totaux", f"{summary['total_depots']:,.2f} {symbole}")
        k2.metric("💰 Liquidités", f"{summary['cash']:,.2f} {symbole}")
        
        # ============================================
        # ✅ BLOC CORRIGÉ : Calculs dans le BON ORDRE
        # ============================================
        if not positions.empty:
            # ÉTAPE 1 : Récupération des prix
            tickers = positions["Ticker"].tolist()
            prices = fetch_last_close_batch(tickers)
            
            # ÉTAPE 2 : Ajout prix actuels avec sécurité None
            positions["Prix_actuel"] = positions["Ticker"].map(prices)
            positions["Prix_actuel"] = positions["Prix_actuel"].fillna(0.0)
            
            # ÉTAPE 3 : Calcul Valeur origine
            positions["Valeur_origine"] = positions["Quantité"] * positions["Prix_actuel"]
            
            # ÉTAPE 4 : Calcul PnL latent (AVANT conversion)
            positions["PnL_latent"] = (positions["Prix_actuel"] - positions["PRU"]) * positions["Quantité"]
            positions["PnL_latent_%"] = ((positions["Prix_actuel"] - positions["PRU"]) / positions["PRU"] * 100).round(2)
            positions["PnL_latent_%"] = positions["PnL_latent_%"].fillna(0.0)
            
            # ÉTAPE 5 : Conversion Valeur (APRÈS avoir créé Valeur_origine)
            positions["Valeur_convertie"] = positions.apply(
                lambda row: currency_manager.convert(
                    row["Valeur_origine"], row["Devise"], devise_affichage
                ) if row["Devise"] != devise_affichage and row["Prix_actuel"] is not None and row["Prix_actuel"] > 0
                else row["Valeur_origine"],
                axis=1
            )
            
            # ÉTAPE 6 : Conversion PnL latent (APRÈS avoir créé PnL_latent)
            positions["PnL_latent_converti"] = positions.apply(
                lambda row: currency_manager.convert(
                    row["PnL_latent"], row["Devise"], devise_affichage
                ) if row["Devise"] != devise_affichage and row["Prix_actuel"] is not None
                else row["PnL_latent"],
                axis=1
            )
            
            # ÉTAPE 7 : Agrégation totaux
            total_valeur = positions["Valeur_convertie"].sum()
            total_pnl_latent = positions["PnL_latent_converti"].sum()
        else:
            total_valeur = 0.0
            total_pnl_latent = 0.0
        
        # ============================================
        # FIN BLOC CORRIGÉ
        # ============================================
        
        k3.metric("📊 Valeur actifs", f"{total_valeur:,.2f} {symbole}")
        k4.metric(
            "📈 PnL Latent",
            f"{total_pnl_latent:,.2f} {symbole}",
            delta=f"{(total_pnl_latent/total_valeur*100):.2f}%" if total_valeur > 0 else "0%"
        )
        k5.metric("✅ PnL Réalisé", f"{summary['pnl_realise_total']:,.2f} {symbole}")
        
        st.divider()
        
        # --- Tableau positions avec format_positions_display ---
        if not positions.empty:
            st.subheader("📋 Positions ouvertes")
            
            # ✅ Utilisation de la fonction utilitaire (si vous l'avez)
            try:
                from utils import format_positions_display
                
                positions_display = format_positions_display(
                    positions=positions,
                    prices=prices,
                    currency_manager=currency_manager,
                    target_currency=devise_affichage,
                    sort_by="PnL_latent_converti",
                    ascending=False
                )
                st.dataframe(positions_display, use_container_width=True, hide_index=True)
            
            except ImportError:
                # Fallback si utils.py n'existe pas encore
                st.warning("⚠️ Module utils.py non trouvé - Affichage basique")
                display_cols = ["Ticker", "Nom complet", "Quantité", "PRU", "Devise", "Prix_actuel"]
                st.dataframe(positions[display_cols], use_container_width=True, hide_index=True)
            
            # --- Graphique répartition ---
            fig_pie = px.pie(
                positions.dropna(subset=["Valeur_convertie"]),
                values="Valeur_convertie",
                names="Nom complet",
                title=f"Répartition du portefeuille ({devise_affichage})"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # --- Graphique PnL ---
            fig_bar = px.bar(
                positions.dropna(subset=["PnL_latent_converti"]),
                x="Ticker",
                y="PnL_latent_converti",
                title="PnL Latent par position",
                color="PnL_latent_converti",
                color_continuous_scale=["red", "gray", "green"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("ℹ️ Aucune position ouverte")
        
        # --- Graphique PnL réalisé cumulatif ---
        df_ventes = st.session_state.df_transactions[
            st.session_state.df_transactions["Type"] == "Vente"
        ].copy()
        
        if not df_ventes.empty:
            df_ventes["Date_sort"] = pd.to_datetime(df_ventes["Date"])
            df_ventes = df_ventes.sort_values("Date_sort")
            df_ventes["PnL_cumule"] = df_ventes["PnL réalisé (€/$)"].cumsum()
            
            fig_line = px.line(
                df_ventes,
                x="Date_sort",
                y="PnL_cumule",
                title="PnL Réalisé Cumulatif",
                labels={"Date_sort": "Date", "PnL_cumule": "PnL Cumulé"}
            )
            st.plotly_chart(fig_line, use_container_width=True)

# -----------------------
# ONGLET 3 : Répartition par Profil
# -----------------------
with tab3:
    st.header("Répartition portefeuilles individuels")
    
    if st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        st.info("ℹ️ Aucune transaction")
    else:
        devise_affichage = st.session_state.devise_affichage
        symbole = "€" if devise_affichage == "EUR" else "$"
        
        profils = sorted(st.session_state.df_transactions["Profil"].unique())
        cols = st.columns(len(profils))
        
        for i, profil in enumerate(profils):
            with cols[i]:
                st.subheader(f"👤 {profil}")
                
                # Filtrage transactions profil
                df_profil = st.session_state.df_transactions[
                    st.session_state.df_transactions["Profil"] == profil
                ]
                
                engine_profil = PortfolioEngine(df_profil)
                summary_profil = engine_profil.get_portfolio_summary_converted(
                    profil=profil,
                    target_currency=devise_affichage,
                    currency_manager=currency_manager
                )
                positions_profil = engine_profil.get_positions(profil=profil)
                
                if not positions_profil.empty:
                    # ÉTAPE 1 : Récupération des prix
                    tickers_profil = positions_profil["Ticker"].tolist()
                    prices_profil = fetch_last_close_batch(tickers_profil)
                    
                    # ÉTAPE 2 : Ajout prix actuels avec sécurité None
                    positions_profil["Prix_actuel"] = positions_profil["Ticker"].map(prices_profil)
                    positions_profil["Prix_actuel"] = positions_profil["Prix_actuel"].fillna(0.0)
                    
                    # ÉTAPE 3 : Calcul Valeur origine
                    positions_profil["Valeur_origine"] = (
                        positions_profil["Quantité"] * positions_profil["Prix_actuel"]
                    )
                    
                    # ÉTAPE 4 : Calcul PnL latent (AVANT conversion)
                    positions_profil["PnL_latent"] = (
                        (positions_profil["Prix_actuel"] - positions_profil["PRU"])
                        * positions_profil["Quantité"]
                    )
                    positions_profil["PnL_latent_%"] = (
                        (positions_profil["Prix_actuel"] - positions_profil["PRU"]) 
                        / positions_profil["PRU"] * 100
                    ).round(2)
                    positions_profil["PnL_latent_%"] = positions_profil["PnL_latent_%"].fillna(0.0)
                    
                    # ÉTAPE 5 : Conversion Valeur (APRÈS avoir créé Valeur_origine)
                    positions_profil["Valeur_convertie"] = positions_profil.apply(
                        lambda row: currency_manager.convert(
                            row["Valeur_origine"], row["Devise"], devise_affichage
                        ) if row["Devise"] != devise_affichage and row["Prix_actuel"] is not None and row["Prix_actuel"] > 0
                        else row["Valeur_origine"],
                        axis=1
                    )
                    
                    # ÉTAPE 6 : Conversion PnL latent (APRÈS avoir créé PnL_latent)
                    positions_profil["PnL_latent_converti"] = positions_profil.apply(
                        lambda row: currency_manager.convert(
                            row["PnL_latent"], row["Devise"], devise_affichage
                        ) if row["Devise"] != devise_affichage and row["Prix_actuel"] is not None
                        else row["PnL_latent"],
                        axis=1
                    )
                    
                    # ÉTAPE 7 : Agrégation totaux
                    total_valeur_profil = positions_profil["Valeur_convertie"].sum()
                    total_pnl_latent_profil = positions_profil["PnL_latent_converti"].sum()
                else:
                    total_valeur_profil = 0.0
                    total_pnl_latent_profil = 0.0
                
                # --- KPI Bloc compact ---
                row1_col1, row1_col2 = st.columns(2)
                row2_col1, row2_col2 = st.columns(2)
                row3_col1, row3_col2 = st.columns(2)
                
                row1_col1.metric("💵 Dépôts", f"{summary_profil['total_depots']:,.0f} {symbole}")
                row1_col2.metric("💰 Liquidités", f"{summary_profil['cash']:,.0f} {symbole}")
                row2_col1.metric("📊 Valeur actifs", f"{total_valeur_profil:,.0f} {symbole}")
                row2_col2.metric("📈 PnL Latent", f"{total_pnl_latent_profil:,.0f} {symbole}")
                row3_col1.metric("✅ PnL Réalisé", f"{summary_profil['pnl_realise_total']:,.0f} {symbole}")
                row3_col2.metric("💎 Total", f"{summary_profil['cash'] + total_valeur_profil:,.0f} {symbole}")
                
                st.divider()
                
                # --- Tableau positions ---
                if not positions_profil.empty:
                    st.caption("**Top 5 Positions**")
                    
                    # ✅ Utilisation de la fonction utilitaire (si disponible)
                    try:
                        from utils import format_positions_display
                        
                        positions_display_profil = format_positions_display(
                            positions=positions_profil,
                            prices=prices_profil,
                            currency_manager=currency_manager,
                            target_currency=devise_affichage,
                            sort_by="PnL_latent_converti",
                            ascending=False
                        )
                        st.dataframe(
                            positions_display_profil.head(5),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    except ImportError:
                        # Fallback si utils.py n'existe pas
                        display_cols = ["Ticker", "Nom complet", "Quantité", "PRU", "Devise"]
                        st.dataframe(
                            positions_profil[display_cols].head(5),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # --- Graphique camembert ---
                    fig_profil = px.pie(
                        positions_profil.dropna(subset=["Valeur_convertie"]),
                        values="Valeur_convertie",
                        names="Nom complet",
                        title=f"Répartition {profil}"
                    )
                    st.plotly_chart(fig_profil, use_container_width=True)
                else:
                    st.info("ℹ️ Aucune position ouverte")
            
            # Séparateur visuel entre profils
            if i < len(profils) - 1:
                st.markdown(
                    "<div style='height:3px; background:linear-gradient(to right, #ccc, #888, #ccc); "
                    "margin:20px 0; border-radius:3px;'></div>",
                    unsafe_allow_html=True
                )
# -----------------------
# ONGLET 4 : Calendrier
# -----------------------
with tab4:
    st.header("📅 Calendrier économique")
    st.info("ℹ️ Fonctionnalité à venir - Phase 2")
    
    st.subheader("💰 Dividendes reçus")
    
    if st.session_state.df_transactions is not None:
        df_div = st.session_state.df_transactions[
            st.session_state.df_transactions["Type"] == "Dividende"
        ].copy()
        
        if not df_div.empty:
            df_div["Date_sort"] = pd.to_datetime(df_div["Date"])
            df_div = df_div.sort_values("Date_sort", ascending=False)
            
            # Tableau dividendes
            display_div = df_div[[
                "Date", "Profil", "Ticker", "Nom complet",
                "PnL réalisé (€/$)", "Devise", "Note"
            ]].head(20)
            st.dataframe(display_div, use_container_width=True, hide_index=True)
            
            # Graphique total dividendes par ticker
            div_by_ticker = df_div.groupby("Ticker")["PnL réalisé (€/$)"].sum().sort_values(ascending=False)
            
            fig_div = px.bar(
                x=div_by_ticker.index,
                y=div_by_ticker.values,
                title="Total dividendes par ticker",
                labels={"x": "Ticker", "y": "Dividendes nets"},
                color=div_by_ticker.values,
                color_continuous_scale=["lightblue", "darkblue"]
            )
            st.plotly_chart(fig_div, use_container_width=True)
        else:
            st.info("ℹ️ Aucun dividende enregistré")

# -----------------------
# SIDEBAR : Statistiques & Actions
# -----------------------
with st.sidebar:
    st.title("Paramètres")
    st.divider()
    
    # --- Statistiques ---
    st.subheader("📊 Statistiques")
    if st.session_state.df_transactions is not None:
        nb_tx = len(st.session_state.df_transactions)
        nb_profils = st.session_state.df_transactions["Profil"].nunique()
        nb_tickers = st.session_state.df_transactions[
            st.session_state.df_transactions["Ticker"] != "CASH"
        ]["Ticker"].nunique()
        
        st.metric("Transactions", nb_tx)
        st.metric("Profils", nb_profils)
        st.metric("Titres uniques", nb_tickers)
    
    st.divider()
    
    # --- Actions ---
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
    
    # --- Informations ---
    st.subheader("ℹ️ Informations")
    st.caption("Dashboard Portefeuille V3.0")
    st.caption("Multi-devises EUR/USD")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Version badge
    st.markdown(
        "<div style='text-align:center; margin-top:20px;'>"
        "<span style='background:#4CAF50; color:white; padding:4px 8px; border-radius:4px; font-size:12px;'>"
        "V3.0 STABLE"
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

# -----------------------
# FOOTER
# -----------------------
st.divider()
st.caption(
    "© 2025 FBM Fintech - Dashboard Portefeuille V3.0 | "
    "Multi-devises EUR/USD | Données temps réel via yfinance"
)

# -----------------------
# FIN APP V3.0
# -----------------------
