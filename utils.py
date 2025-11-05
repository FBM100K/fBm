"""
Utils V3.0 - Fonctions utilitaires pour le dashboard
✅ Standardisation affichage tableaux positions
✅ Formatage valeurs avec conversions devise
"""

import pandas as pd
from typing import Optional


def format_positions_display(
    positions: pd.DataFrame,
    prices: dict,
    currency_manager,
    target_currency: str = "EUR",
    sort_by: str = "PnL_latent_converti",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Formate un DataFrame de positions pour affichage unifié.
    Args:
        positions: DataFrame avec colonnes [Ticker, Nom complet, Quantité, PRU, Devise]
        prices: Dict {ticker: prix_actuel} depuis yfinance
        currency_manager: Instance CurrencyManager pour conversions
        target_currency: Devise d'affichage (EUR/USD)
        sort_by: Colonne de tri (défaut: PnL_latent_converti)
        ascending: Ordre tri (défaut: descendant)
    
    Returns:
        DataFrame formaté prêt à afficher avec colonnes:
        [Ticker, Nom complet, Qté, PRU, Dev, Prix actuel, Valeur, PnL €/$, PnL %]
    """
    if positions.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Nom complet", "Qté", "PRU", "Dev",
            "Prix actuel", "Valeur", "PnL €/$", "PnL %"
        ])
    
    # Copie de sécurité
    df = positions.copy()
    
    # Symbole devise cible
    symbole = "€" if target_currency == "EUR" else "$"
    
    # --- Ajout prix actuels ---
    df["Prix_actuel"] = df["Ticker"].map(prices)
    df["Prix_actuel"] = df["Prix_actuel"].fillna(0.0)
    
    # --- Calculs valorisation ---
    df["Valeur_origine"] = df["Quantité"] * df["Prix_actuel"]
    
    # Conversion valeur si devise différente
    df["Valeur_convertie"] = df.apply(
        lambda row: currency_manager.convert(
            row["Valeur_origine"], row["Devise"], target_currency
        ) if row["Devise"] != target_currency and row["Prix_actuel"] > 0
        else row["Valeur_origine"],
        axis=1
    )
    
    # --- Calculs PnL ---
    df["PnL_latent"] = (df["Prix_actuel"] - df["PRU"]) * df["Quantité"]
    df["PnL_latent_%"] = ((df["Prix_actuel"] - df["PRU"]) / df["PRU"] * 100).round(2)
    df["PnL_latent_%"] = df["PnL_latent_%"].fillna(0.0)
    
    # Conversion PnL si devise différente
    df["PnL_latent_converti"] = df.apply(
        lambda row: currency_manager.convert(
            row["PnL_latent"], row["Devise"], target_currency
        ) if row["Devise"] != target_currency
        else row["PnL_latent"],
        axis=1
    )
    
    # --- Formatage affichage avec conversion ---
    df["Valeur_display"] = df.apply(
        lambda row: f"{row['Valeur_origine']:,.2f} {row['Devise']}" +
                   (f" ({row['Valeur_convertie']:,.2f} {symbole})" 
                    if row['Devise'] != target_currency else ""),
        axis=1
    )
    
    df["PnL_display"] = df.apply(
        lambda row: f"{row['PnL_latent']:,.2f} {row['Devise']}" +
                   (f" ({row['PnL_latent_converti']:,.2f} {symbole})"
                    if row['Devise'] != target_currency else ""),
        axis=1
    )
    
    df["PnL_%_display"] = df["PnL_latent_%"].apply(lambda x: f"{x:+.2f}%")
    
    # --- Formatage prix actuel ---
    df["Prix_actuel_display"] = df.apply(
        lambda row: f"{row['Prix_actuel']:,.2f}" if row['Prix_actuel'] > 0 else "N/A",
        axis=1
    )
    
    # --- Sélection et renommage colonnes finales ---
    display_df = df[[
        "Ticker", "Nom complet", "Quantité", "PRU", "Devise",
        "Prix_actuel_display", "Valeur_display", "PnL_display", "PnL_%_display"
    ]].copy()
    
    display_df.columns = [
        "Ticker", "Nom complet", "Qté", "PRU", "Dev",
        "Prix actuel", "Valeur", "PnL €/$", "PnL %"
    ]
    
    # --- Tri si colonne disponible ---
    if sort_by in df.columns:
        # On trie sur la colonne non formatée pour ordre numérique correct
        sort_values = df[sort_by].fillna(0)
        display_df = display_df.iloc[sort_values.sort_values(ascending=ascending).index]
    
    return display_df.reset_index(drop=True)


def format_currency_value(
    value: float,
    currency: str,
    target_currency: str,
    currency_manager,
    show_conversion: bool = True
) -> str:
    """
    Formate une valeur avec conversion optionnelle.
    
    Args:
        value: Montant à formater
        currency: Devise d'origine
        target_currency: Devise cible
        currency_manager: Instance CurrencyManager
        show_conversion: Si True, affiche conversion entre parenthèses
    
    Returns:
        String formaté ex: "1,500 USD (1,382.49 €)"
    """
    symbole_origine = "€" if currency == "EUR" else "$"
    symbole_cible = "€" if target_currency == "EUR" else "$"
    
    formatted = f"{value:,.2f} {symbole_origine}"
    
    if show_conversion and currency != target_currency:
        converted = currency_manager.convert(value, currency, target_currency)
        formatted += f" ({converted:,.2f} {symbole_cible})"
    
    return formatted


def get_color_pnl(pnl_percent: float) -> str:
    """
    Retourne la couleur appropriée selon performance PnL.
    
    Args:
        pnl_percent: Pourcentage de PnL
    
    Returns:
        Code couleur: 'green', 'red', ou 'gray'
    """
    if pnl_percent > 0.5:
        return "green"
    elif pnl_percent < -0.5:
        return "red"
    else:
        return "gray"


def validate_dataframe_columns(df: pd.DataFrame, required_cols: list) -> tuple[bool, str]:
    """
    Valide qu'un DataFrame possède toutes les colonnes requises.
    
    Args:
        df: DataFrame à valider
        required_cols: Liste des colonnes obligatoires
    
    Returns:
        Tuple (is_valid, error_message)
    """
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        return False, f"Colonnes manquantes: {', '.join(missing)}"
    
    return True, ""


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division sécurisée évitant ZeroDivisionError."""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def format_number_compact(value: float, decimals: int = 2) -> str:
    """
    Formate un nombre de manière compacte (K, M, B).
    
    Args:
        value: Nombre à formater
        decimals: Nombre de décimales
    
    Returns:
        String formaté ex: "1.5K", "2.3M"
    """
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1_000_000_000:
        return f"{sign}{abs_value/1_000_000_000:.{decimals}f}B"
    elif abs_value >= 1_000_000:
        return f"{sign}{abs_value/1_000_000:.{decimals}f}M"
    elif abs_value >= 1_000:
        return f"{sign}{abs_value/1_000:.{decimals}f}K"
    else:
        return f"{sign}{abs_value:.{decimals}f}"
