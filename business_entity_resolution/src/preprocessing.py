"""
Preprocessing module for Business Entity Resolution.
Handles data preparation, tokenization, and structured component extraction
(e.g., postal codes, street numbers, token signatures).
Preserves all country records (including France, US, India, and any unseen regions).
"""

import re
from typing import List
import pandas as pd
from src.normalization import add_normalized_columns

# Regular expressions for address component parsing
RE_INDIA_PIN = re.compile(r"\b[1-9][0-9]{5}\b")
RE_GENERIC_POSTAL = re.compile(r"\b[0-9]{5}\b")
RE_STREET_NUM = re.compile(r"\b0*([0-9]+)\b")


def extract_postal_code(addr_norm: str, country: str) -> str:
    """
    Extract 6-digit Indian PIN code or 5-digit US/French postal code.
    Returns empty string if not found.
    """
    if not addr_norm:
        return ""

    if country.lower() == "india":
        matches = RE_INDIA_PIN.findall(addr_norm)
    else:  # US, France, or any other country
        matches = RE_GENERIC_POSTAL.findall(addr_norm)

    return matches[-1] if matches else ""


def extract_street_number(addr_norm: str) -> str:
    """
    Extract house/building street number without leading zeros.
    Example: '0337 oakland avenue' -> '337'
    """
    if not addr_norm:
        return ""
    matches = RE_STREET_NUM.findall(addr_norm)
    return matches[0] if matches else ""


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply complete normalization and preprocessing pipeline to a DataFrame.
    Adds token sets, prefixes, postal codes, and street numbers.
    Never drops or filters any country (including France).
    """
    # 1. Base Normalization
    df = add_normalized_columns(df)

    # 2. Token-based attributes for business names
    norm_names = df["name_norm"].tolist()
    core_names = df["name_core"].tolist()

    name_tokens = [n.split() for n in norm_names]
    core_tokens = [n.split() for n in core_names]

    df["name_tokens"] = name_tokens
    df["first_token"] = [t[0] if t else "" for t in core_tokens]
    df["first_2_tokens"] = [" ".join(t[:2]) if len(t) >= 2 else (t[0] if t else "") for t in core_tokens]
    df["token_sorted_name"] = [" ".join(sorted(set(t))) for t in core_tokens]

    # 3. Address Tokenization and Components
    addr_norm_list = df["address_norm"].tolist()
    countries = df["country_norm"].tolist()

    df["address_tokens"] = [a.split() for a in addr_norm_list]
    df["postal_code"] = [
        extract_postal_code(addr, c) for addr, c in zip(addr_norm_list, countries)
    ]
    df["street_number"] = [extract_street_number(addr) for addr in addr_norm_list]

    return df
