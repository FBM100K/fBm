"""
Portfolio Engine V2.1 - Module de calculs financiers avec multi-devises
Gère PRU avec frais, PRU_vente figé, taux de change figés, et validations
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, Optional, Tuple


class PortfolioEngine:
    """
    Moteur de calculs pour le portefeuille.
    Gère PRU, PnL réalisé/latent, validations, multi-devises.
    """
    
    def __init__(self, df_transactions: pd.DataFrame):
        """
        Args:
            df_transactions: DataFrame avec colonnes attendues
        """
        self.df = df_transactions.copy()
        self._normalize_dataframe()
    
    def _normalize_dataframe(self):
        """Normalise les types de données du DataFrame"""
        if self.df.empty:
            return
        
        # Date
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        
        # Colonnes numériques
        numeric_cols = ["Quantité", "Prix_unitaire", "Frais (€/$)", 
                       "PnL réalisé (€/$)", "PnL réalisé (%)", "PRU_vente", "Taux_change"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)
        
        # String columns
        for col in ["Type", "Ticker", "Profil", "Devise", "Devise_reference"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)
    
    def calculate_pru(self, ticker: str, profil: str, 
                     date_limite: Optional[datetime] = None) -> float:
        """
        Calcule le PRU (Prix de Revient Unitaire) avec prise en compte des frais.
        
        Formule: PRU = [Σ(Qté achat × Prix achat) + Σ(Frais achat)] / Σ(Qté achat)
        
        Args:
            ticker: Symbole du titre
            profil: Nom du portefeuille
            date_limite: Si fourni, calcule PRU jusqu'à cette date uniquement
        
        Returns:
            PRU moyen pondéré avec frais
        """
        mask = (
            (self.df["Ticker"] == ticker) & 
            (self.df["Profil"] == profil) &
            (self.df["Type"] == "Achat") &
            (self.df["Quantité"] > 0)
        )
        
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
    
    def get_position_quantity(self, ticker: str, profil: str, 
                             date_limite: Optional[datetime] = None) -> float:
        """
        Calcule la quantité nette détenue (achats - ventes).
        
        Args:
            ticker: Symbole du titre
            profil: Nom du portefeuille
            date_limite: Si fourni, calcule quantité jusqu'à cette date
        
        Returns:
            Quantité nette
        """
        mask = (
            (self.df["Ticker"] == ticker) & 
            (self.df["Profil"] == profil) &
            (self.df["Type"].isin(["Achat", "Vente"]))
        )
        
        if date_limite is not None:
            mask &= (self.df["Date"] < date_limite)
        
        transactions = self.df[mask]
        
        if transactions.empty:
            return 0.0
        
        return transactions["Quantité"].sum()
    
    def validate_currency_consistency(self, ticker: str, profil: str, 
                                     devise: str) -> Tuple[bool, str]:
        """
        Vérifie qu'un ticker n'est pas déjà détenu dans une autre devise.
        
        RÈGLE: Un ticker ne peut être détenu que dans UNE SEULE devise par profil.
        
        Args:
            ticker: Symbole du titre
            profil: Nom du portefeuille
            devise: Devise de la nouvelle transaction (EUR, USD)
        
        Returns:
            (is_valid, error_message)
        """
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
                f"❌ Impossible : {ticker} est déjà détenu en {existing_devise} "
                f"dans le portefeuille {profil}.\n\n"
                f"Vous ne pouvez pas l'acheter en {devise}.\n\n"
                f"💡 Solutions :\n"
                f"  1. Acheter en {existing_devise} (devise existante)\n"
                f"  2. Vendre votre position {existing_devise} avant d'acheter en {devise}"
            )
        
        return True, ""
    
    def validate_sale(self, ticker: str, profil: str, 
                     quantite_vente: float, date_vente: datetime) -> Tuple[bool, str]:
        """
        Valide qu'une vente est possible (quantité disponible suffisante).
        
        Args:
            ticker: Symbole du titre
            profil: Nom du portefeuille
            quantite_vente: Quantité à vendre (valeur positive)
            date_vente: Date de la vente
        
        Returns:
            (is_valid, message_erreur)
        """
        qty_disponible = self.get_position_quantity(ticker, profil, date_vente)
        
        if qty_disponible < quantite_vente:
            return False, f"❌ Quantité insuffisante. Disponible: {qty_disponible:.6f}, Demandé: {quantite_vente:.6f}"
        
        if quantite_vente <= 0:
            return False, "❌ La quantité de vente doit être > 0"
        
        return True, ""
    
    def prepare_achat_transaction(self, ticker: str, profil: str,
                                 quantite: float, prix_achat: float,
                                 frais: float, date_achat: datetime,
                                 devise: str, note: str = "",
                                 currency_manager=None) -> Dict:
        """
        Prépare une transaction d'achat avec taux de change figé.
        
        Args:
            currency_manager: Instance de CurrencyManager (optionnel)
        
        Returns:
            Dict représentant la transaction
        """
        devise_reference = "EUR"
        taux_change = 1.0
        
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        transaction = {
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
        
        return transaction
    
    def prepare_sale_transaction(self, ticker: str, profil: str, 
                                quantite: float, prix_vente: float,
                                frais: float, date_vente: datetime,
                                devise: str, note: str = "",
                                currency_manager=None) -> Optional[Dict]:
        """
        Prépare une transaction de vente avec PRU_vente figé et taux de change.
        """
        is_valid, error_msg = self.validate_sale(ticker, profil, quantite, date_vente)
        if not is_valid:
            print(error_msg)
            return None
        
        pru_vente = self.calculate_pru(ticker, profil, date_vente)
        
        if pru_vente == 0:
            print(f"⚠️ Warning: PRU = 0 pour {ticker}. Calcul PnL impossible.")
        
        pnl_reel = (prix_vente - pru_vente) * quantite - frais
        pnl_pct = ((prix_vente - pru_vente) / pru_vente * 100) if pru_vente != 0 else 0.0
        
        devise_reference = "EUR"
        taux_change = 1.0
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        transaction = {
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
        
        return transaction
    
    def prepare_depot_transaction(self, profil: str, montant: float,
                                  date_depot: datetime, devise: str,
                                  note: str = "", currency_manager=None) -> Dict:
        """Prépare une transaction de dépôt avec taux de change figé."""
        devise_reference = "EUR"
        taux_change = 1.0
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        transaction = {
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
        
        return transaction
    
    def prepare_retrait_transaction(self, profil: str, montant: float,
                                   date_retrait: datetime, devise: str,
                                   note: str = "", currency_manager=None) -> Dict:
        """Prépare une transaction de retrait avec taux de change figé."""
        devise_reference = "EUR"
        taux_change = 1.0
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        transaction = {
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
        
        return transaction
    
    def prepare_dividende_transaction(self, ticker: str, profil: str,
                                     montant_brut: float, retenue_source: float,
                                     date_dividende: datetime, devise: str,
                                     note: str = "", currency_manager=None) -> Dict:
        """Prépare une transaction de dividende avec taux de change figé."""
        montant_net = montant_brut - retenue_source
        
        devise_reference = "EUR"
        taux_change = 1.0
        if devise != devise_reference and currency_manager:
            taux_change = currency_manager.get_rate(devise_reference, devise)
        
        transaction = {
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
        
        return transaction
    
    def get_portfolio_summary(self, profil: Optional[str] = None) -> Dict:
        """
        Calcule un résumé du portefeuille (SANS conversion).
        
        Args:
            profil: Si fourni, filtre sur ce profil uniquement
        
        Returns:
            Dict avec métriques clés
        """
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
            "pnl_dividendes": round(pnl_dividendes, 2),
            "target_currency": target_currency
        }
    
    def get_positions(self, profil: Optional[str] = None) -> pd.DataFrame:
        """
        Retourne les positions ouvertes avec PRU calculé.
        
        Args:
            profil: Si fourni, filtre sur ce profil
        
        Returns:
            DataFrame avec colonnes: Ticker, Profil, Quantité, PRU, Devise
        """
        df = self.df.copy()
        if profil:
            df = df[df["Profil"] == profil]
        
        df_actifs = df[df["Ticker"].str.upper() != "CASH"]
        
        if df_actifs.empty:
            return pd.DataFrame(columns=["Ticker", "Profil", "Quantité", "PRU", "Devise"])
        
        positions = []
        for (ticker, prof), group in df_actifs.groupby(["Ticker", "Profil"]):
            qty = group["Quantité"].sum()
            if qty != 0:
                pru = self.calculate_pru(ticker, prof)
                devise_position = group.iloc[0]["Devise"]
                positions.append({
                    "Ticker": ticker,
                    "Profil": prof,
                    "Quantité": round(qty, 6),
                    "PRU": round(pru, 6),
                    "Devise": devise_position
                })
        
        return pd.DataFrame(positions)


if __name__ == "__main__":
    print("✅ Portfolio Engine V2.1 chargé avec succès")
its, 2),
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
        """
        Calcule un résumé du portefeuille AVEC conversion dans la devise cible.
        Utilise les taux figés historiques pour chaque transaction.
        
        Args:
            profil: Si fourni, filtre sur ce profil
            target_currency: Devise cible (EUR, USD)
            currency_manager: Instance de CurrencyManager
        
        Returns:
            Dict avec toutes les valeurs converties dans target_currency
        """
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
        
        def convert_with_historic_rate(row, target_curr, curr_manager):
            """Convertit en utilisant le taux figé de la transaction"""
            montant_devise_origine = row.get("montant", 0.0)
            devise_origine = row.get("Devise", target_curr)
            taux_fige = row.get("Taux_change", 1.0)
            devise_ref = row.get("Devise_reference", "EUR")
            
            if devise_origine == target_curr:
                return montant_devise_origine
            
            if devise_origine != devise_ref and taux_fige != 1.0:
                montant_ref = montant_devise_origine / taux_fige
            else:
                montant_ref = montant_devise_origine
            
            if curr_manager and devise_ref != target_curr:
                rate_to_target = curr_manager.get_rate(devise_ref, target_curr)
                return montant_ref * rate_to_target
            
            return montant_ref
        
        depots = df[df["Type"] == "Dépôt"].copy()
        depots["montant"] = depots["Quantité"]
        total_depots = sum(convert_with_historic_rate(row, target_currency, currency_manager) 
                          for _, row in depots.iterrows()) if not depots.empty else 0.0
        
        retraits = df[df["Type"] == "Retrait"].copy()
        retraits["montant"] = abs(retraits["Quantité"])
        total_retraits = sum(convert_with_historic_rate(row, target_currency, currency_manager)
                            for _, row in retraits.iterrows()) if not retraits.empty else 0.0
        
        achats = df[df["Type"] == "Achat"].copy()
        achats["montant"] = achats["Quantité"] * achats["Prix_unitaire"]
        total_achats = sum(convert_with_historic_rate(row, target_currency, currency_manager)
                          for _, row in achats.iterrows()) if not achats.empty else 0.0
        
        ventes = df[df["Type"] == "Vente"].copy()
        ventes["montant"] = abs(ventes["Quantité"]) * ventes["Prix_unitaire"]
        total_ventes = sum(convert_with_historic_rate(row, target_currency, currency_manager)
                          for _, row in ventes.iterrows()) if not ventes.empty else 0.0
        
        df_avec_frais = df[df["Frais (€/$)"] > 0].copy()
        df_avec_frais["montant"] = df_avec_frais["Frais (€/$)"]
        total_frais = sum(convert_with_historic_rate(row, target_currency, currency_manager)
                         for _, row in df_avec_frais.iterrows()) if not df_avec_frais.empty else 0.0
        
        ventes_pnl = df[df["Type"] == "Vente"].copy()
        ventes_pnl["montant"] = ventes_pnl["PnL réalisé (€/$)"]
        pnl_ventes = sum(convert_with_historic_rate(row, target_currency, currency_manager)
                        for _, row in ventes_pnl.iterrows()) if not ventes_pnl.empty else 0.0
        
        dividendes = df[df["Type"] == "Dividende"].copy()
        dividendes["montant"] = dividendes["PnL réalisé (€/$)"]
        pnl_dividendes = sum(convert_with_historic_rate(row, target_currency, currency_manager)
                            for _, row in dividendes.iterrows()) if not dividendes.empty else 0.0
        
        pnl_realise_total = pnl_ventes + pnl_dividendes
        cash = total_depots - total_retraits + total_ventes - total_achats - total_frais + pnl_dividendes
        
        return {
            "total_depots": round(total_depots, 2),
            "total_retraits": round(total_retra
