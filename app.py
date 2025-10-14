"""
Dashboard Portefeuille V2 - Intégration PortfolioEngine
Remplace les sections critiques du code V1
"""

import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date
from google.oauth2.service_account import Credentials
import requests

# Import du moteur de calculs
from portfolio_engine import PortfolioEngine

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Dashboard Portefeuille V2", layout="wide")
st.markdown("<h1 style='text-align: left; font-size: 30px;'>📊 Dashboard Portefeuille - FBM V2</h1>", unsafe_allow_html=True)

SHEET_NAME = "transactions_dashboard"

# ⭐ COLONNES V2 MISES À JOUR
EXPECTED_COLS = [
    "Date",
    "Profil",
    "Type",
    "Ticker",
    "Quantité",
    "Prix_unitaire",
    "PRU_vente",  # ⭐ NOUVEAU
    "Devise",
    "Frais (€/$)",
    "PnL réalisé (€/$)",
    "PnL réalisé (%)",
    "Note",
    "History_Log"  # ⭐ NOUVEAU
]

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# -----------------------
# Google Sheets Auth (identique V1)
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
# Helpers utilitaires (identiques V1)
# -----------------------
def parse_float(val):
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

# -----------------------
# Lecture / écriture transactions
# -----------------------
@st.cache_data(ttl=60)
def load_transactions_from_sheet():
    if sheet is None:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    try:
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
        
        # Ajouter colonnes manquantes
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None
        
        # Normaliser Date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        
        # Colonnes numériques
        numeric_cols = ["Quantité", "Prix_unitaire", "Frais (€/$)", 
                       "PnL réalisé (€/$)", "PnL réalisé (%)", "PRU_vente"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(["", "None", "nan"], "0").str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Defaults
        df["Devise"] = df["Devise"].fillna("EUR")
        df["Profil"] = df["Profil"].fillna("Gas")
        df["Type"] = df["Type"].fillna("Achat")
        
        # Réordonner
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
    
    # Convertir Date
    if "Date" in df_out.columns:
        df_out["Date"] = df_out["Date"].apply(
            lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) and isinstance(d, (date, pd.Timestamp)) else (d if d else "")
        )
    
    # Assurer colonnes
    for c in EXPECTED_COLS:
        if c not in df_out.columns:
            df_out[c] = ""
    
    values = [EXPECTED_COLS] + df_out[EXPECTED_COLS].fillna("").astype(str).values.tolist()
    
    try:
        # Backup (simplifié)
        try:
            backup_name = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            sh.add_worksheet(title=backup_name, rows="1000", cols="20")
        except:
            pass  # Best effort
        
        sheet.clear()
        sheet.update("A1", values, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        st.error(f"Erreur écriture: {e}")
        return False

# -----------------------
# Fetch prices (identique V1)
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
if "df_transactions" not in st.session_state:
    df_loaded = load_transactions_from_sheet()
    st.session_state.df_transactions = df_loaded

# -----------------------
# Onglets
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["💰 Transactions", "📂 Portefeuille", "📊 Répartition", "Calendrier"])

# -----------------------
# ONGLET 1 : Transactions avec PortfolioEngine
# -----------------------
with tab1:
    st.header("Ajouter une transaction")
    
    profil = st.selectbox("Portefeuille / Profil", ["Gas", "Marc"], index=0)
    type_tx = st.selectbox("Type", ["Achat", "Vente", "Dépôt", "Retrait", "Dividende"], index=0)
    
    # Recherche ticker (identique V1)
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
    
    def get_alpha_vantage_suggestions(query: str):
        if not ALPHA_VANTAGE_API_KEY or not query or len(query) < 2:
            return []
        url = "https://www.alphavantage.co/query"
        params = {"function": "SYMBOL_SEARCH", "keywords": query, "apikey": ALPHA_VANTAGE_API_KEY}
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
        except:
            return []
    
    query = st.text_input("Entrez un nom ou ticker :", value=st.session_state.ticker_query)
    if st.button("🔎 Rechercher"):
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
    
    # Formulaire
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
    
    # ⭐ NOUVEAU : Utilisation PortfolioEngine
    if st.button("➕ Ajouter Transaction", type="primary"):
        quantite = parse_float(quantite_input)
        prix = parse_float(prix_input)
        frais = parse_float(frais_input)
        
        # Validation basique
        if type_tx in ("Achat", "Vente") and not ticker_selected:
            st.error("Ticker requis pour Achat/Vente")
        elif quantite <= 0.0001:
            st.error("Quantité doit être > 0.0001")
        elif prix <= 0.0001 and type_tx not in ["Dépôt", "Retrait"]:
            st.error("Prix doit être > 0.0001")
        else:
            df_hist = st.session_state.df_transactions.copy()
            engine = PortfolioEngine(df_hist)
            
            ticker = ticker_selected if ticker_selected else "CASH"
            date_tx = pd.to_datetime(date_input)
            
            transaction = None
            
            # ⭐ Préparation selon type avec PortfolioEngine
            if type_tx == "Achat":
                transaction = engine.prepare_achat_transaction(
                    ticker=ticker, profil=profil, quantite=quantite,
                    prix_achat=prix, frais=frais, date_achat=date_tx,
                    devise=devise, note=note
                )
            
            elif type_tx == "Vente":
                transaction = engine.prepare_sale_transaction(
                    ticker=ticker, profil=profil, quantite=quantite,
                    prix_vente=prix, frais=frais, date_vente=date_tx,
                    devise=devise, note=note
                )
                if transaction is None:
                    st.error("Impossible de créer la vente (quantité insuffisante)")
            
            elif type_tx == "Dépôt":
                transaction = engine.prepare_depot_transaction(
                    profil=profil, montant=quantite if quantite > 0 else prix,
                    date_depot=date_tx, devise=devise, note=note
                )
            
            elif type_tx == "Retrait":
                transaction = engine.prepare_retrait_transaction(
                    profil=profil, montant=quantite if quantite > 0 else prix,
                    date_retrait=date_tx, devise=devise, note=note
                )
            
            elif type_tx == "Dividende":
                # Pour dividendes, quantite = montant brut, frais = retenue
                transaction = engine.prepare_dividende_transaction(
                    ticker=ticker, profil=profil, montant_brut=quantite,
                    retenue_source=frais, date_dividende=date_tx,
                    devise=devise, note=note
                )
            
            # Enregistrement
            if transaction:
                df_new = pd.concat([df_hist, pd.DataFrame([transaction])], ignore_index=True)
                ok = save_transactions_to_sheet(df_new)
                if ok:
                    st.success(f"✅ {type_tx} enregistré : {transaction['Ticker']}")
                    if type_tx == "Vente":
                        st.info(f"📊 PRU_vente figé : {transaction['PRU_vente']:.2f} {devise}")
                        st.info(f"💰 PnL réalisé : {transaction['PnL réalisé (€/$)']:.2f} {devise}")
                    st.session_state.df_transactions = df_new
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Erreur enregistrement")
    
    # Historique
    st.divider()
    st.subheader("📜 Historique des transactions")
    if st.session_state.df_transactions is not None and not st.session_state.df_transactions.empty:
        df_display = st.session_state.df_transactions.copy()
        df_display["Date_sort"] = pd.to_datetime(df_display["Date"], errors="coerce")
        df_display = df_display.sort_values(by="Date_sort", ascending=False).drop(columns=["Date_sort", "History_Log"])
        st.dataframe(df_display.head(100), use_container_width=True)
    else:
        st.info("Aucune transaction")

# -----------------------
# ONGLET 2 : Portefeuille avec PortfolioEngine
# -----------------------
with tab2:
    st.header("📂 Portefeuille consolidé")
    
    if st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        st.info("Aucune transaction")
    else:
        # ⭐ Utilisation PortfolioEngine pour calculs
        engine = PortfolioEngine(st.session_state.df_transactions)
        summary = engine.get_portfolio_summary()
        positions = engine.get_positions()
        
        # KPIs
        st.subheader("📊 Indicateurs clés")
        k1, k2, k3, k4, k5 = st.columns(5)
        
        k1.metric("💵 Dépôts totaux", f"{summary['total_depots']:,.2f} €")
        k2.metric("💰 Liquidités", f"{summary['cash']:,.2f} €")
        
        # Calculer valeur actifs
        if not positions.empty:
            tickers = positions["Ticker"].tolist()
            prices = fetch_last_close_batch(tickers)
            positions["Prix_actuel"] = positions["Ticker"].map(prices)
            positions["Valeur"] = positions["Quantité"] * positions["Prix_actuel"]
            positions["PnL_latent"] = (positions["Prix_actuel"] - positions["PRU"]) * positions["Quantité"]
            positions["PnL_latent_%"] = ((positions["Prix_actuel"] - positions["PRU"]) / positions["PRU"] * 100).round(2)
            
            total_valeur = positions["Valeur"].sum()
            total_pnl_latent = positions["PnL_latent"].sum()
        else:
            total_valeur = 0.0
            total_pnl_latent = 0.0
        
        k3.metric("📊 Valeur actifs", f"{total_valeur:,.2f} €")
        k4.metric("📈 PnL Latent", f"{total_pnl_latent:,.2f} €", 
                 delta=f"{(total_pnl_latent/total_valeur*100):.2f}%" if total_valeur > 0 else "0%")
        k5.metric("✅ PnL Réalisé", f"{summary['pnl_realise_total']:,.2f} €")
        
        st.divider()
        
        # Tableau positions
        if not positions.empty:
            st.subheader("📋 Positions ouvertes")
            display_positions = positions[["Ticker", "Profil", "Quantité", "PRU", "Prix_actuel", "Valeur", "PnL_latent", "PnL_latent_%"]].copy()
            display_positions = display_positions.sort_values("Valeur", ascending=False)
            st.dataframe(display_positions, use_container_width=True)
            
            # Graphique
            fig = px.pie(display_positions, values="Valeur", names="Ticker",
                        title="Répartition du portefeuille")
            st.plotly_chart(fig, use_container_width=True)
            
            # PnL par position
            fig2 = px.bar(display_positions, x="Ticker", y="PnL_latent",
                         title="PnL Latent par position",
                         color="PnL_latent",
                         color_continuous_scale=["red", "gray", "green"])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune position ouverte")
        
        # PnL réalisé cumulé
        df_ventes = st.session_state.df_transactions[
            st.session_state.df_transactions["Type"] == "Vente"
        ].copy()
        
        if not df_ventes.empty:
            df_ventes["Date_sort"] = pd.to_datetime(df_ventes["Date"])
            df_ventes = df_ventes.sort_values("Date_sort")
            df_ventes["PnL_cumule"] = df_ventes["PnL réalisé (€/$)"].cumsum()
            
            fig3 = px.line(df_ventes, x="Date_sort", y="PnL_cumule",
                          title="PnL Réalisé Cumulatif",
                          labels={"Date_sort": "Date", "PnL_cumule": "PnL Cumulé (€)"})
            st.plotly_chart(fig3, use_container_width=True)
        
        # Détails calculs
        with st.expander("🔍 Détails des calculs"):
            st.write("**Résumé financier:**")
            st.json(summary)

# -----------------------
# ONGLET 3 : Répartition par Profil avec PortfolioEngine
# -----------------------
with tab3:
    st.header("📊 Comparatif portefeuilles individuels")
    
    if st.session_state.df_transactions is None or st.session_state.df_transactions.empty:
        st.info("Aucune transaction")
    else:
        profils = sorted(st.session_state.df_transactions["Profil"].unique())
        cols = st.columns(len(profils))
        
        for i, profil in enumerate(profils):
            with cols[i]:
                st.subheader(f"📊 {profil}")
                
                # Filtrer sur profil
                df_profil = st.session_state.df_transactions[
                    st.session_state.df_transactions["Profil"] == profil
                ]
                
                # ⭐ Utilisation PortfolioEngine
                engine_profil = PortfolioEngine(df_profil)
                summary_profil = engine_profil.get_portfolio_summary(profil=profil)
                positions_profil = engine_profil.get_positions(profil=profil)
                
                # Calculer valeurs avec prix actuels
                if not positions_profil.empty:
                    tickers_profil = positions_profil["Ticker"].tolist()
                    prices_profil = fetch_last_close_batch(tickers_profil)
                    positions_profil["Prix_actuel"] = positions_profil["Ticker"].map(prices_profil)
                    positions_profil["Valeur"] = positions_profil["Quantité"] * positions_profil["Prix_actuel"]
                    positions_profil["PnL_latent"] = (positions_profil["Prix_actuel"] - positions_profil["PRU"]) * positions_profil["Quantité"]
                    
                    total_valeur_profil = positions_profil["Valeur"].sum()
                    total_pnl_latent_profil = positions_profil["PnL_latent"].sum()
                else:
                    total_valeur_profil = 0.0
                    total_pnl_latent_profil = 0.0
                
                # KPIs
                row1_col1, row1_col2 = st.columns(2)
                row2_col1, row2_col2 = st.columns(2)
                row3_col1, row3_col2 = st.columns(2)
                
                row1_col1.metric("💵 Dépôts", f"{summary_profil['total_depots']:,.0f} €")
                row1_col2.metric("💰 Liquidités", f"{summary_profil['cash']:,.0f} €")
                row2_col1.metric("📊 Valeur actifs", f"{total_valeur_profil:,.0f} €")
                row2_col2.metric("📈 PnL Latent", f"{total_pnl_latent_profil:,.0f} €")
                row3_col1.metric("✅ PnL Réalisé", f"{summary_profil['pnl_realise_total']:,.0f} €")
                row3_col2.metric("💎 Total", f"{summary_profil['cash'] + total_valeur_profil:,.0f} €")
                
                # Graphique répartition
                if not positions_profil.empty:
                    fig_profil = px.pie(
                        positions_profil.dropna(subset=["Valeur"]),
                        values="Valeur",
                        names="Ticker",
                        title=f"Répartition {profil}"
                    )
                    st.plotly_chart(fig_profil, use_container_width=True)
                    
                    # Top 5 positions
                    st.markdown("**Top 5 positions:**")
                    top5 = positions_profil.nlargest(5, "Valeur")[["Ticker", "Quantité", "Valeur", "PnL_latent"]]
                    st.dataframe(top5, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune position")

# -----------------------
# ONGLET 4 : Calendrier
# -----------------------
with tab4:
    st.header("📅 Calendrier économique")
    st.info("Fonctionnalité à venir - Phase 2")
    
    # Afficher prochains dividendes attendus (si données)
    st.subheader("💰 Dividendes reçus")
    
    if st.session_state.df_transactions is not None:
        df_div = st.session_state.df_transactions[
            st.session_state.df_transactions["Type"] == "Dividende"
        ].copy()
        
        if not df_div.empty:
            df_div["Date_sort"] = pd.to_datetime(df_div["Date"])
            df_div = df_div.sort_values("Date_sort", ascending=False)
            
            display_div = df_div[["Date", "Profil", "Ticker", "PnL réalisé (€/$)", "Devise", "Note"]].head(20)
            st.dataframe(display_div, use_container_width=True, hide_index=True)
            
            # Total dividendes par ticker
            div_by_ticker = df_div.groupby("Ticker")["PnL réalisé (€/$)"].sum().sort_values(ascending=False)
            
            fig_div = px.bar(
                x=div_by_ticker.index,
                y=div_by_ticker.values,
                title="Total dividendes par ticker",
                labels={"x": "Ticker", "y": "Dividendes nets (€)"}
            )
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
        nb_tickers = st.session_state.df_transactions[
            st.session_state.df_transactions["Ticker"] != "CASH"
        ]["Ticker"].nunique()
        
        st.metric("Transactions", nb_tx)
        st.metric("Profils", nb_profils)
        st.metric("Titres uniques", nb_tickers)
    
    st.divider()
    
    st.subheader("🔄 Actions")
    
    if st.button("♻️ Rafraîchir données", use_container_width=True):
        st.cache_data.clear()
        st.session_state.df_transactions = load_transactions_from_sheet()
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
    st.caption("Dashboard Portefeuille V2")
    st.caption("Powered by PortfolioEngine")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Indicateur PRU_vente
    if st.session_state.df_transactions is not None:
        ventes = st.session_state.df_transactions[
            st.session_state.df_transactions["Type"] == "Vente"
        ]
        if not ventes.empty:
            ventes_avec_pru = ventes[ventes["PRU_vente"].notna() & (ventes["PRU_vente"] > 0)]
            pct_migre = len(ventes_avec_pru) / len(ventes) * 100
            
            if pct_migre < 100:
                st.warning(f"⚠️ Migration V2 incomplète: {pct_migre:.0f}% des ventes ont PRU_vente")
                if st.button("🔧 Lancer migration", use_container_width=True):
                    st.info("Utilisez le script migrate_to_v2.py")
            else:
                st.success("✅ Toutes les ventes ont PRU_vente")


# -----------------------
# FOOTER
# -----------------------
st.divider()
st.caption("© 2025 FBM Fintech - Dashboard Portefeuille V2 | Données temps réel via yfinance | Stockage Google Sheets")
