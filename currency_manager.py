"""
Currency Manager - Gestion des taux de change avec cache
API: exchangerate-api.com (gratuit 1500 req/mois)
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, Optional


class CurrencyManager:
    """
    Gestionnaire de taux de change avec cache et fallback.
    """
    
    # Devises supportées Phase 1
    SUPPORTED_CURRENCIES = ["EUR", "USD"]
    
    # Cache TTL : 4 heures
    CACHE_TTL_SECONDS = 14400
    
    # Taux de fallback si API down
    FALLBACK_RATES = {
        "EUR": {"USD": 1.10, "EUR": 1.0},
        "USD": {"EUR": 0.91, "USD": 1.0}
    }
    
    # API endpoint (gratuit, pas de clé requise)
    API_URL = "https://api.exchangerate-api.com/v4/latest/{base}"
    
    def __init__(self):
        """Initialise le gestionnaire avec cache vide"""
        self.cache = {}
        self.last_update = None
        self.using_fallback = False
    
    def get_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Retourne le taux de change from_currency → to_currency.
        
        Args:
            from_currency: Devise source (EUR, USD)
            to_currency: Devise cible (EUR, USD)
        
        Returns:
            Taux de change (float)
        """
        if from_currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Devise non supportée: {from_currency}")
        if to_currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Devise non supportée: {to_currency}")
        
        if from_currency == to_currency:
            return 1.0
        
        rates = self._get_rates(from_currency)
        return rates.get(to_currency, 1.0)
    
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Convertit un montant d'une devise à une autre.
        
        Args:
            amount: Montant à convertir
            from_currency: Devise source
            to_currency: Devise cible
        
        Returns:
            Montant converti
        """
        rate = self.get_rate(from_currency, to_currency)
        return round(amount * rate, 2)
    
    def _get_rates(self, base_currency: str) -> Dict[str, float]:
        """Récupère les taux depuis le cache ou l'API."""
        if base_currency in self.cache:
            cached_data = self.cache[base_currency]
            timestamp = cached_data["timestamp"]
            
            if datetime.now() - timestamp < timedelta(seconds=self.CACHE_TTL_SECONDS):
                self.using_fallback = False
                return cached_data["rates"]
        
        rates = self._fetch_rates_from_api(base_currency)
        
        if rates:
            self.cache[base_currency] = {
                "rates": rates,
                "timestamp": datetime.now()
            }
            self.last_update = datetime.now()
            self.using_fallback = False
            return rates
        else:
            self.using_fallback = True
            return self.FALLBACK_RATES.get(base_currency, {})
    
    def _fetch_rates_from_api(self, base_currency: str) -> Optional[Dict[str, float]]:
        """Fetch taux depuis exchangerate-api.com."""
        try:
            url = self.API_URL.format(base=base_currency)
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            all_rates = data.get("rates", {})
            
            rates = {
                currency: all_rates[currency] 
                for currency in self.SUPPORTED_CURRENCIES 
                if currency in all_rates
            }
            
            return rates
            
        except Exception as e:
            print(f"⚠️ Erreur fetch API taux de change: {e}")
            return None
    
    def get_cache_info(self) -> Dict:
        """Retourne des infos sur le cache (pour affichage UI)."""
        if not self.last_update:
            return {
                "status": "Non initialisé",
                "last_update": None,
                "using_fallback": False,
                "age_minutes": None
            }
        
        age = datetime.now() - self.last_update
        age_minutes = int(age.total_seconds() / 60)
        
        return {
            "status": "Taux fallback" if self.using_fallback else "Taux API",
            "last_update": self.last_update.strftime("%d/%m/%Y %H:%M"),
            "using_fallback": self.using_fallback,
            "age_minutes": age_minutes
        }
    
    def clear_cache(self):
        """Force le rafraîchissement en vidant le cache."""
        self.cache = {}
        self.last_update = None
    
    def get_rate_display(self, from_currency: str, to_currency: str) -> str:
        """Retourne le taux formaté pour affichage UI."""
        rate = self.get_rate(from_currency, to_currency)
        return f"1 {from_currency} = {rate:.4f} {to_currency}"


if __name__ == "__main__":
    # Tests rapides
    manager = CurrencyManager()
    print(f"Taux EUR→USD: {manager.get_rate('EUR', 'USD')}")
    print(f"100 EUR = {manager.convert(100, 'EUR', 'USD')} USD")
    print(f"Cache: {manager.get_cache_info()}")
