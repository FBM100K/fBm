"""
Portfolio Engine V3.0 - Module de calculs financiers avec multi-devises
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, Optional, Tuple


class PortfolioEngine:
    """Moteur de calculs pour le portefeuille."""
    
    def __init__(self, df_transactions: pd.DataFrame):
        self.df = df_transactions.copy()
        self._normalize_dataframe()
    
    def _normalize_dataframe(self):
        """Normalise les types de données du DataFrame"""
        if self.df.empty:
            return
        
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        
        numeric_cols = ["Quantité", "Prix_unitaire", "Frais (€/$)", 
                       "PnL réalisé (€/$)", "PnL réalisé (%)", "PRU_vente", "Taux_change"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)
        
        for col in ["Type", "Ticker", "Profil", "Devise", "Devise_reference"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)
    
    def calculate_pru(self, ticker: str, profil: str = None, date_limite: Optional[datetime] = None) -> float:
        """
        Calcule le PRU avec frais pour un ticker donné.
        Si profil=None, calcul global (tous profils confondus).
        """
        mask = (
            (self.df["Ticker"] == ticker) &
            (self.df["Type"] == "Achat") &
            (self.df["Quantité"] > 0)
        )
        
        if profil is not None:
            mask &= (self.df["Profil"] == profil)
        
        if date_limite is not None:
            mask &= (self.df["Date"] < date_limite)

        achats = self.df[mask]

        if achats.empty:
            return 0.0

        total_cout = (achats["Quantité"] * achats["Prix_unitaire"]).sum()
        total_frais_achat = achats["Frais (€/$)"].sum()
        total_quantite = achats["Quantité"].sum()

        if total_quantite == 0:
            return 0.0

        pru = (total_cout + total_frais_achat) / total_quantite
        return round(pru, 6)
    
    def get_position_quantity(self, ticker: str, profil: str = None, date_limite: Optional[datetime] = None) -> float:
        """
        Retourne la quantité nette détenue pour un ticker.
        Si profil=None, calcul global (tous profils confondus).
        """
        mask = (
            (self.df["Ticker"] == ticker) &
            (self.df["Type"].isin(["Achat", "Vente"]))
        )
        
        if profil is not None:
            mask &= (self.df["Profil"] == profil)
        
        if date_limite is not None:
            mask &= (self.df["Date"] < date_limite)

        transactions = self.df[mask]
        if transactions.empty:
            return 0.0

        return transactions["Quantité"].sum()
    
    def validate_currency_consistency(self, ticker: str, profil: str, devise: str) -> Tuple[bool, str]:
        """Vérifie qu'un ticker n'est pas déjà détenu dans une autre devise."""
        existing = self.df[
            (self.df["Ticker"] == ticker) & 
            (self.df["Profil"] == profil) &
            (self.df["Type"].isin(["Achat", "Vente"]))
        ]
        
        if existing.empty:
            return True, ""
        
        existing_devises = existing["Devise"].unique()
        
        if len(existing_devises) > 0 and devise not in existing_devises:
            existing_devise = existing_devises[0]
            return False, (
                f"❌ Impossible : {ticker} est déjà détenu en {existing_devise}\n\n"
                f"Solutions :\n"
                f"1. Acheter en {existing_devise}\n"
                f"2. Vendre votre position {existing_devise} avant d'acheter en {devise}"
            )
        
        return True, 
    def validate_sale(self, ticker: str, profil: str, quantite_vente: float, date_vente: datetime) -> Tuple[bool, str]:
        # Vérifier quantité pour LE PROFIL spécifique
        qty_disponible = self.get_position_quantity(ticker, profil=profil, date_limite=date_vente)
        
        # Vérification quantité positive
        if quantite_vente <= 0:
            return False, "❌ La quantité de vente doit être supérieure à 0"
        
        # Vérification stock suffisant
        if qty_disponible < quantite_vente:
            return False, (
                f"❌ Quantité insuffisante pour {profil}\n"
                f"📊 Disponible : {qty_disponible:.4f} actions\n"
                f"🔴 Demandé : {quantite_vente:.4f} actions\n"
                f"💡 Vous pouvez vendre maximum {qty_disponible:.4f} actions"
            )
        return True,""
    
    def prepare_achat_transaction(self, ticker: str, profil: str, quantite: float, prix_achat: float,
                                 frais: float, date_achat: datetime, devise: str, note: str = "",
                                 currency_manager=None) -> Dict:
        """Prépare une transaction d'achat avec taux de change figé."""
        devise_reference = "EUR"
        taux_change = 1.0
        
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        return {
            "Date": date_achat.date() if isinstance(date_achat, datetime) else date_achat,
            "Profil": profil,
            "Type": "Achat",
            "Ticker": ticker,
            "Quantité": round(quantite, 6),
            "Prix_unitaire": round(prix_achat, 6),
            "PRU_vente": None,
            "Devise": devise,
            "Taux_change": round(taux_change, 6),
            "Devise_reference": devise_reference,
            "Frais (€/$)": round(frais, 2),
            "PnL réalisé (€/$)": 0.0,
            "PnL réalisé (%)": 0.0,
            "Note": note,
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
    
    def prepare_sale_transaction(self, ticker: str, profil: str, quantite: float, prix_vente: float,
                                frais: float, date_vente: datetime, devise: str, note: str = "",
                                currency_manager=None) -> Optional[Dict]:
        """Prépare une transaction de vente avec PRU_vente figé."""
        is_valid, error_msg = self.validate_sale(ticker, profil, quantite, date_vente)
        if not is_valid:
            print(error_msg)
            return None
        
        pru_vente = self.calculate_pru(ticker, profil, date_vente)
        
        if pru_vente == 0:
            print(f"⚠️ Warning: PRU = 0 pour {ticker}")
        
        pnl_reel = (prix_vente - pru_vente) * quantite - frais
        pnl_pct = ((prix_vente - pru_vente) / pru_vente * 100) if pru_vente != 0 else 0.0
        
        devise_reference = "EUR"
        taux_change = 1.0
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        return {
            "Date": date_vente.date() if isinstance(date_vente, datetime) else date_vente,
            "Profil": profil,
            "Type": "Vente",
            "Ticker": ticker,
            "Quantité": -abs(round(quantite, 6)),
            "Prix_unitaire": round(prix_vente, 6),
            "PRU_vente": round(pru_vente, 6),
            "Devise": devise,
            "Taux_change": round(taux_change, 6),
            "Devise_reference": devise_reference,
            "Frais (€/$)": round(frais, 2),
            "PnL réalisé (€/$)": round(pnl_reel, 2),
            "PnL réalisé (%)": round(pnl_pct, 2),
            "Note": note,
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
    
    def prepare_depot_transaction(self, profil: str, montant: float, date_depot: datetime, 
                                  devise: str, note: str = "", currency_manager=None) -> Dict:
        """Prépare une transaction de dépôt."""
        devise_reference = "EUR"
        taux_change = 1.0
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        return {
            "Date": date_depot.date() if isinstance(date_depot, datetime) else date_depot,
            "Profil": profil,
            "Type": "Dépôt",
            "Ticker": "CASH",
            "Quantité": round(montant, 2),
            "Prix_unitaire": 1.0,
            "PRU_vente": None,
            "Devise": devise,
            "Taux_change": round(taux_change, 6),
            "Devise_reference": devise_reference,
            "Frais (€/$)": 0.0,
            "PnL réalisé (€/$)": 0.0,
            "PnL réalisé (%)": 0.0,
            "Note": note,
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
    
    def prepare_retrait_transaction(self, profil: str, montant: float, date_retrait: datetime,
                                   devise: str, note: str = "", currency_manager=None) -> Dict:
        """Prépare une transaction de retrait."""
        devise_reference = "EUR"
        taux_change = 1.0
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        return {
            "Date": date_retrait.date() if isinstance(date_retrait, datetime) else date_retrait,
            "Profil": profil,
            "Type": "Retrait",
            "Ticker": "CASH",
            "Quantité": -round(montant, 2),
            "Prix_unitaire": 1.0,
            "PRU_vente": None,
            "Devise": devise,
            "Taux_change": round(taux_change, 6),
            "Devise_reference": devise_reference,
            "Frais (€/$)": 0.0,
            "PnL réalisé (€/$)": 0.0,
            "PnL réalisé (%)": 0.0,
            "Note": note,
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
    
    def prepare_dividende_transaction(self, ticker: str, profil: str, montant_brut: float, 
                                     retenue_source: float, date_dividende: datetime, devise: str,
                                     note: str = "", currency_manager=None) -> Dict:
        """Prépare une transaction de dividende."""
        montant_net = montant_brut - retenue_source
        
        devise_reference = "EUR"
        taux_change = 1.0
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        return {
            "Date": date_dividende.date() if isinstance(date_dividende, datetime) else date_dividende,
            "Profil": profil,
            "Type": "Dividende",
            "Ticker": ticker,
            "Quantité": 0.0,
            "Prix_unitaire": 0.0,
            "PRU_vente": None,
            "Devise": devise,
            "Taux_change": round(taux_change, 6),
            "Devise_reference": devise_reference,
            "Frais (€/$)": round(retenue_source, 2),
            "PnL réalisé (€/$)": round(montant_net, 2),
            "PnL réalisé (%)": 0.0,
            "Note": note + f" | Brut: {montant_brut:.2f}, Retenue: {retenue_source:.2f}",
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
    
    def get_portfolio_summary(self, profil: Optional[str] = None) -> Dict:
        """Calcule un résumé du portefeuille (sans conversion)."""
        df = self.df.copy()
        if profil:
            df = df[df["Profil"] == profil]
        
        if df.empty:
            return {
                "total_depots": 0.0,
                "total_retraits": 0.0,
                "cash": 0.0,
                "total_achats": 0.0,
                "total_ventes": 0.0,
                "total_frais": 0.0,
                "pnl_realise_total": 0.0,
                "pnl_dividendes": 0.0
            }
        
        depots = df[df["Type"] == "Dépôt"]
        total_depots = depots["Quantité"].sum()
        
        retraits = df[df["Type"] == "Retrait"]
        total_retraits = abs(retraits["Quantité"].sum())
        
        achats = df[df["Type"] == "Achat"]
        total_achats = (achats["Quantité"] * achats["Prix_unitaire"]).sum()
        
        ventes = df[df["Type"] == "Vente"]
        total_ventes = (abs(ventes["Quantité"]) * ventes["Prix_unitaire"]).sum()
        
        total_frais = df["Frais (€/$)"].sum()
        
        pnl_ventes = ventes["PnL réalisé (€/$)"].sum()
        dividendes = df[df["Type"] == "Dividende"]
        pnl_dividendes = dividendes["PnL réalisé (€/$)"].sum()
        pnl_realise_total = pnl_ventes + pnl_dividendes
        
        cash = total_depots - total_retraits + total_ventes - total_achats - total_frais + pnl_dividendes
        
        return {
            "total_depots": round(total_depots, 2),
            "total_retraits": round(total_retraits, 2),
            "cash": round(cash, 2),
            "total_achats": round(total_achats, 2),
            "total_ventes": round(total_ventes, 2),
            "total_frais": round(total_frais, 2),
            "pnl_realise_total": round(pnl_realise_total, 2),
            "pnl_dividendes": round(pnl_dividendes, 2)
        }
    
    def get_portfolio_summary_converted(self, profil: Optional[str] = None, 
                                       target_currency: str = "EUR",
                                       currency_manager=None) -> Dict:
        """Calcule un résumé du portefeuille avec conversion."""
        df = self.df.copy()
        if profil:
            df = df[df["Profil"] == profil]
        
        if df.empty:
            return {
                "total_depots": 0.0,
                "total_retraits": 0.0,
                "cash": 0.0,
                "total_achats": 0.0,
                "total_ventes": 0.0,
                "total_frais": 0.0,
                "pnl_realise_total": 0.0,
                "pnl_dividendes": 0.0,
                "target_currency": target_currency
            }
        
        def convert_amount(row, target_curr, curr_mgr):
            """Convertit avec taux figé"""
            montant = row.get("montant", 0.0)
            devise_orig = row.get("Devise", target_curr)
            taux_fige = row.get("Taux_change", 1.0)
            devise_ref = row.get("Devise_reference", "EUR")
            
            if devise_orig == target_curr:
                return montant
            
            if devise_orig != devise_ref and taux_fige != 1.0:
                montant_ref = montant / taux_fige
            else:
                montant_ref = montant
            
            if curr_mgr and devise_ref != target_curr:
                rate = curr_mgr.get_rate(devise_ref, target_curr)
                return montant_ref * rate
            
            return montant_ref
        
        depots = df[df["Type"] == "Dépôt"].copy()
        depots["montant"] = depots["Quantité"]
        total_depots = sum(convert_amount(row, target_currency, currency_manager) 
                          for _, row in depots.iterrows()) if not depots.empty else 0.0
        
        retraits = df[df["Type"] == "Retrait"].copy()
        retraits["montant"] = abs(retraits["Quantité"])
        total_retraits = sum(convert_amount(row, target_currency, currency_manager)
                            for _, row in retraits.iterrows()) if not retraits.empty else 0.0
        
        achats = df[df["Type"] == "Achat"].copy()
        achats["montant"] = achats["Quantité"] * achats["Prix_unitaire"]
        total_achats = sum(convert_amount(row, target_currency, currency_manager)
                          for _, row in achats.iterrows()) if not achats.empty else 0.0
        
        ventes = df[df["Type"] == "Vente"].copy()
        ventes["montant"] = abs(ventes["Quantité"]) * ventes["Prix_unitaire"]
        total_ventes = sum(convert_amount(row, target_currency, currency_manager)
                          for _, row in ventes.iterrows()) if not ventes.empty else 0.0
        
        df_frais = df[df["Frais (€/$)"] > 0].copy()
        df_frais["montant"] = df_frais["Frais (€/$)"]
        total_frais = sum(convert_amount(row, target_currency, currency_manager)
                         for _, row in df_frais.iterrows()) if not df_frais.empty else 0.0
        
        ventes_pnl = df[df["Type"] == "Vente"].copy()
        ventes_pnl["montant"] = ventes_pnl["PnL réalisé (€/$)"]
        pnl_ventes = sum(convert_amount(row, target_currency, currency_manager)
                        for _, row in ventes_pnl.iterrows()) if not ventes_pnl.empty else 0.0
        
        dividendes = df[df["Type"] == "Dividende"].copy()
        dividendes["montant"] = dividendes["PnL réalisé (€/$)"]
        pnl_dividendes = sum(convert_amount(row, target_currency, currency_manager)
                            for _, row in dividendes.iterrows()) if not dividendes.empty else 0.0
        
        pnl_realise_total = pnl_ventes + pnl_dividendes
        cash = total_depots - total_retraits + total_ventes - total_achats - total_frais + pnl_dividendes
        
        return {
            "total_depots": round(total_depots, 2),
            "total_retraits": round(total_retraits, 2),
            "cash": round(cash, 2),
            "total_achats": round(total_achats, 2),
            "total_ventes": round(total_ventes, 2),
            "total_frais": round(total_frais, 2),
            "pnl_realise_total": round(pnl_realise_total, 2),
            "pnl_dividendes": round(pnl_dividendes, 2),
            "target_currency": target_currency
        }
    
    def get_positions(self, profil: Optional[str] = None) -> pd.DataFrame:
        """
        Retourne les positions ouvertes avec PRU pour un profil spécifique.
        Format: Ticker, Nom complet, Profil, Quantité, PRU, Devise
        """
        df = self.df.copy()
        if profil:
            df = df[df["Profil"] == profil]
        
        df_actifs = df[df["Ticker"].str.upper() != "CASH"]
        
        if df_actifs.empty:
            return pd.DataFrame(columns=["Ticker", "Nom complet", "Profil", "Quantité", "PRU", "Devise"])
        
        positions = []
        for (ticker, prof), group in df_actifs.groupby(["Ticker", "Profil"]):
            qty = group["Quantité"].sum()
            if qty > 0:  # Uniquement positions ouvertes
                pru = self.calculate_pru(ticker, prof)
                devise_position = group.iloc[0]["Devise"]
                nom_complet = group.iloc[0].get("Nom complet", ticker)
                positions.append({
                    "Ticker": ticker,
                    "Nom complet": nom_complet,
                    "Profil": prof,
                    "Quantité": round(qty, 6),
                    "PRU": round(pru, 6),
                    "Devise": devise_position
                })
        
        return pd.DataFrame(positions)
    
    def get_positions_consolide(self) -> pd.DataFrame:
        """
        ✅ V3.0 - Retourne les positions consolidées (tous profils confondus).
        CORRECTIONS:
        - Ajout colonne "Nom complet"
        - Calcul PRU correct avec frais
        - Gestion devise unique par ticker
        
        Format: Ticker, Nom complet, Quantité, PRU, Devise
        """
        df = self.df.copy()
        df_actifs = df[df["Ticker"].str.upper() != "CASH"]

        if df_actifs.empty:
            return pd.DataFrame(columns=["Ticker", "Nom complet", "Quantité", "PRU", "Devise"])

        # Groupement par Ticker (on suppose devise unique par ticker)
        positions = []
        
        for ticker, group in df_actifs.groupby("Ticker"):
            # Filtrer achats/ventes uniquement
            trades = group[group["Type"].isin(["Achat", "Vente"])]
            
            if trades.empty:
                continue
            
            # Quantité nette
            qty_nette = trades["Quantité"].sum()
            
            if qty_nette <= 0:
                continue  # Position fermée
            
            # Calcul PRU consolidé (sans filtre profil)
            pru = self.calculate_pru(ticker, profil=None)
            
            # Récupération devise et nom complet
            devise = trades.iloc[0]["Devise"]
            nom_complet = trades.iloc[0].get("Nom complet", ticker)
            
            positions.append({
                "Ticker": ticker,
                "Nom complet": nom_complet,
                "Quantité": round(qty_nette, 6),
                "PRU": round(pru, 6),
                "Devise": devise
            })
        
        return pd.DataFrame(positions)
