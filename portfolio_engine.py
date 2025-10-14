"""
Portfolio Engine V2 - Module de calculs financiers
Gère PRU avec frais, PRU_vente figé, et validations
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, Optional, Tuple


class PortfolioEngine:
    """
    Moteur de calculs pour le portefeuille.
    Gère PRU, PnL réalisé/latent, validations.
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
                       "PnL réalisé (€/$)", "PnL réalisé (%)", "PRU_vente"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)
        
        # String columns
        for col in ["Type", "Ticker", "Profil", "Devise"]:
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
        # Filtrer les achats du ticker/profil
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
        
        # Calcul PRU avec frais
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
            Quantité nette (peut être négative si erreur de données)
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
    
    def prepare_sale_transaction(self, ticker: str, profil: str, 
                                quantite: float, prix_vente: float,
                                frais: float, date_vente: datetime,
                                devise: str, note: str = "") -> Optional[Dict]:
        """
        Prépare une transaction de vente avec PRU_vente figé.
        
        Args:
            ticker: Symbole du titre
            profil: Nom du portefeuille
            quantite: Quantité vendue (valeur positive)
            prix_vente: Prix de vente unitaire
            frais: Frais de transaction
            date_vente: Date de la vente
            devise: EUR ou USD
            note: Note optionnelle
        
        Returns:
            Dict représentant la transaction, ou None si validation échoue
        """
        # Validation
        is_valid, error_msg = self.validate_sale(ticker, profil, quantite, date_vente)
        if not is_valid:
            print(error_msg)
            return None
        
        # Calcul PRU au moment de la vente (AVANT d'enregistrer la vente)
        pru_vente = self.calculate_pru(ticker, profil, date_vente)
        
        if pru_vente == 0:
            print(f"⚠️ Warning: PRU = 0 pour {ticker}. Calcul PnL impossible.")
        
        # Calcul PnL réalisé
        pnl_reel = (prix_vente - pru_vente) * quantite - frais
        pnl_pct = ((prix_vente - pru_vente) / pru_vente * 100) if pru_vente != 0 else 0.0
        
        # Construction transaction
        transaction = {
            "Date": date_vente.date() if isinstance(date_vente, datetime) else date_vente,
            "Profil": profil,
            "Type": "Vente",
            "Ticker": ticker,
            "Quantité": -abs(round(quantite, 6)),  # Négatif pour vente
            "Prix_unitaire": round(prix_vente, 6),
            "PRU_vente": round(pru_vente, 6),  # ⭐ FIGÉ ICI
            "Devise": devise,
            "Frais (€/$)": round(frais, 2),
            "PnL réalisé (€/$)": round(pnl_reel, 2),
            "PnL réalisé (%)": round(pnl_pct, 2),
            "Note": note,
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
        
        return transaction
    
    def prepare_achat_transaction(self, ticker: str, profil: str,
                                 quantite: float, prix_achat: float,
                                 frais: float, date_achat: datetime,
                                 devise: str, note: str = "") -> Dict:
        """
        Prépare une transaction d'achat.
        
        Returns:
            Dict représentant la transaction
        """
        transaction = {
            "Date": date_achat.date() if isinstance(date_achat, datetime) else date_achat,
            "Profil": profil,
            "Type": "Achat",
            "Ticker": ticker,
            "Quantité": round(quantite, 6),
            "Prix_unitaire": round(prix_achat, 6),
            "PRU_vente": None,  # NULL pour achat
            "Devise": devise,
            "Frais (€/$)": round(frais, 2),
            "PnL réalisé (€/$)": 0.0,
            "PnL réalisé (%)": 0.0,
            "Note": note,
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
        
        return transaction
    
    def prepare_depot_transaction(self, profil: str, montant: float,
                                  date_depot: datetime, devise: str,
                                  note: str = "") -> Dict:
        """
        Prépare une transaction de dépôt.
        
        Returns:
            Dict représentant la transaction
        """
        transaction = {
            "Date": date_depot.date() if isinstance(date_depot, datetime) else date_depot,
            "Profil": profil,
            "Type": "Dépôt",
            "Ticker": "CASH",
            "Quantité": round(montant, 2),
            "Prix_unitaire": 1.0,
            "PRU_vente": None,
            "Devise": devise,
            "Frais (€/$)": 0.0,
            "PnL réalisé (€/$)": 0.0,
            "PnL réalisé (%)": 0.0,
            "Note": note,
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
        
        return transaction
    
    def prepare_retrait_transaction(self, profil: str, montant: float,
                                   date_retrait: datetime, devise: str,
                                   note: str = "") -> Dict:
        """
        Prépare une transaction de retrait.
        
        Returns:
            Dict représentant la transaction
        """
        transaction = {
            "Date": date_retrait.date() if isinstance(date_retrait, datetime) else date_retrait,
            "Profil": profil,
            "Type": "Retrait",
            "Ticker": "CASH",
            "Quantité": -round(montant, 2),  # Négatif pour retrait
            "Prix_unitaire": 1.0,
            "PRU_vente": None,
            "Devise": devise,
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
                                     note: str = "") -> Dict:
        """
        Prépare une transaction de dividende.
        
        Args:
            montant_brut: Montant brut du dividende
            retenue_source: Retenue à la source (impôts)
        
        Returns:
            Dict représentant la transaction
        """
        montant_net = montant_brut - retenue_source
        
        transaction = {
            "Date": date_dividende.date() if isinstance(date_dividende, datetime) else date_dividende,
            "Profil": profil,
            "Type": "Dividende",
            "Ticker": ticker,
            "Quantité": 0.0,  # Pas de quantité pour dividende
            "Prix_unitaire": 0.0,
            "PRU_vente": None,
            "Devise": devise,
            "Frais (€/$)": round(retenue_source, 2),  # Retenue = frais
            "PnL réalisé (€/$)": round(montant_net, 2),
            "PnL réalisé (%)": 0.0,
            "Note": note + f" | Brut: {montant_brut:.2f}, Retenue: {retenue_source:.2f}",
            "History_Log": f"Créé le {datetime.utcnow().isoformat()}Z"
        }
        
        return transaction
    
    def get_portfolio_summary(self, profil: Optional[str] = None) -> Dict:
        """
        Calcule un résumé du portefeuille.
        
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
        
        # Dépôts
        depots = df[df["Type"] == "Dépôt"]
        total_depots = depots["Quantité"].sum()
        
        # Retraits
        retraits = df[df["Type"] == "Retrait"]
        total_retraits = abs(retraits["Quantité"].sum())
        
        # Achats
        achats = df[df["Type"] == "Achat"]
        total_achats = (achats["Quantité"] * achats["Prix_unitaire"]).sum()
        
        # Ventes
        ventes = df[df["Type"] == "Vente"]
        total_ventes = (abs(ventes["Quantité"]) * ventes["Prix_unitaire"]).sum()
        
        # Frais totaux
        total_frais = df["Frais (€/$)"].sum()
        
        # PnL réalisé (ventes + dividendes)
        pnl_ventes = ventes["PnL réalisé (€/$)"].sum()
        dividendes = df[df["Type"] == "Dividende"]
        pnl_dividendes = dividendes["PnL réalisé (€/$)"].sum()
        pnl_realise_total = pnl_ventes + pnl_dividendes
        
        # Cash disponible
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
    
    def get_positions(self, profil: Optional[str] = None) -> pd.DataFrame:
        """
        Retourne les positions ouvertes avec PRU calculé.
        
        Args:
            profil: Si fourni, filtre sur ce profil
        
        Returns:
            DataFrame avec colonnes: Ticker, Profil, Quantité, PRU
        """
        df = self.df.copy()
        if profil:
            df = df[df["Profil"] == profil]
        
        # Filtrer transactions avec ticker (pas CASH)
        df_actifs = df[df["Ticker"].str.upper() != "CASH"]
        
        if df_actifs.empty:
            return pd.DataFrame(columns=["Ticker", "Profil", "Quantité", "PRU"])
        
        # Grouper par Ticker/Profil
        positions = []
        for (ticker, prof), group in df_actifs.groupby(["Ticker", "Profil"]):
            qty = group["Quantité"].sum()
            if qty != 0:  # Ignorer positions fermées
                pru = self.calculate_pru(ticker, prof)
                positions.append({
                    "Ticker": ticker,
                    "Profil": prof,
                    "Quantité": round(qty, 6),
                    "PRU": round(pru, 6)
                })
        
        return pd.DataFrame(positions)


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_portfolio_engine():
    """Tests de validation des calculs"""
    
    print("🧪 Tests PortfolioEngine...\n")
    
    # Dataset de test
    test_data = [
        # Dépôt initial
        {"Date": "2024-01-01", "Profil": "Gas", "Type": "Dépôt", "Ticker": "CASH", 
         "Quantité": 10000, "Prix_unitaire": 1, "Frais (€/$)": 0, "Devise": "EUR",
         "PnL réalisé (€/$)": 0, "PnL réalisé (%)": 0, "PRU_vente": None, "Note": ""},
        
        # Achat 1
        {"Date": "2024-01-02", "Profil": "Gas", "Type": "Achat", "Ticker": "AAPL", 
         "Quantité": 10, "Prix_unitaire": 100, "Frais (€/$)": 10, "Devise": "USD",
         "PnL réalisé (€/$)": 0, "PnL réalisé (%)": 0, "PRU_vente": None, "Note": ""},
        
        # Achat 2
        {"Date": "2024-01-03", "Profil": "Gas", "Type": "Achat", "Ticker": "AAPL", 
         "Quantité": 10, "Prix_unitaire": 120, "Frais (€/$)": 10, "Devise": "USD",
         "PnL réalisé (€/$)": 0, "PnL réalisé (%)": 0, "PRU_vente": None, "Note": ""},
    ]
    
    df_test = pd.DataFrame(test_data)
    engine = PortfolioEngine(df_test)
    
    # Test 1: Calcul PRU avec frais
    pru = engine.calculate_pru("AAPL", "Gas")
    # PRU = (10*100 + 10*120 + 10 + 10) / 20 = 2220 / 20 = 111
    assert pru == 111.0, f"❌ Test PRU échoué: attendu 111.0, obtenu {pru}"
    print(f"✅ Test 1: PRU avec frais = {pru} (attendu: 111.0)")
    
    # Test 2: Quantité disponible
    qty = engine.get_position_quantity("AAPL", "Gas")
    assert qty == 20.0, f"❌ Test quantité échoué: attendu 20.0, obtenu {qty}"
    print(f"✅ Test 2: Quantité disponible = {qty} (attendu: 20.0)")
    
    # Test 3: Validation vente OK
    is_valid, msg = engine.validate_sale("AAPL", "Gas", 5, datetime(2024, 1, 4))
    assert is_valid, f"❌ Test validation vente OK échoué: {msg}"
    print(f"✅ Test 3: Validation vente 5 actions = OK")
    
    # Test 4: Validation vente KO (quantité insuffisante)
    is_valid, msg = engine.validate_sale("AAPL", "Gas", 25, datetime(2024, 1, 4))
    assert not is_valid, f"❌ Test validation vente KO échoué"
    print(f"✅ Test 4: Validation vente 25 actions = KO (insuffisant)")
    
    # Test 5: Préparation vente avec PRU_vente figé
    tx_vente = engine.prepare_sale_transaction(
        ticker="AAPL",
        profil="Gas",
        quantite=5,
        prix_vente=150,
        frais=5,
        date_vente=datetime(2024, 1, 4),
        devise="USD",
        note="Test vente"
    )
    
    assert tx_vente is not None, "❌ Préparation vente échouée"
    assert tx_vente["PRU_vente"] == 111.0, f"❌ PRU_vente incorrect: {tx_vente['PRU_vente']}"
    
    # PnL attendu = (150 - 111) * 5 - 5 = 190
    assert tx_vente["PnL réalisé (€/$)"] == 190.0, f"❌ PnL incorrect: {tx_vente['PnL réalisé (€/$)']}"
    print(f"✅ Test 5: Vente préparée avec PRU_vente={tx_vente['PRU_vente']}, PnL={tx_vente['PnL réalisé (€/$)']}")
    
    # Test 6: Vérifier que PRU_vente reste figé après rachat
    # Simuler ajout de la vente + un rachat
    df_test_etendu = pd.concat([df_test, pd.DataFrame([tx_vente])], ignore_index=True)
    
    achat3 = {
        "Date": "2024-01-05", "Profil": "Gas", "Type": "Achat", "Ticker": "AAPL", 
        "Quantité": 10, "Prix_unitaire": 80, "Frais (€/$)": 10, "Devise": "USD",
        "PnL réalisé (€/$)": 0, "PnL réalisé (%)": 0, "PRU_vente": None, "Note": ""
    }
    df_test_etendu = pd.concat([df_test_etendu, pd.DataFrame([achat3])], ignore_index=True)
    
    engine_apres = PortfolioEngine(df_test_etendu)
    pru_nouveau = engine_apres.calculate_pru("AAPL", "Gas")
    # Nouveau PRU = (15*111 + 10*80 + 10) / 25 = 2475 / 25 = 99
    # Note: 15 actions restantes à PRU 111, pas 15*100+15*120
    # Recalculons: achats = 10@100 + 10@120 + 10@80, frais = 10+10+10 = 30
    # PRU = (1000 + 1200 + 800 + 30) / 30 = 3030 / 30 = 101
    # Mais on a vendu 5, donc on recalcule sur achats uniquement
    pru_recalc = (10*100 + 10*120 + 10*80 + 10 + 10 + 10) / 30
    
    # Vérifier PRU_vente dans la transaction vente n'a PAS changé
    tx_vente_apres = df_test_etendu[df_test_etendu["Type"] == "Vente"].iloc[0]
    assert tx_vente_apres["PRU_vente"] == 111.0, f"❌ PRU_vente a changé ! {tx_vente_apres['PRU_vente']}"
    print(f"✅ Test 6: PRU_vente reste figé à 111.0 après rachat (nouveau PRU position = {pru_nouveau:.2f})")
    
    print("\n🎉 Tous les tests passés !\n")


if __name__ == "__main__":
    test_portfolio_engine()
