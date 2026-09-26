"""
Normalization module for Business Entity Resolution.
Creates separate normalized representations for business names, addresses, and countries.
"""

import re
import unicodedata
from typing import List, Set
import pandas as pd
from src.config import LEGAL_SUFFIXES, ADDRESS_ABBREVIATIONS

# Precompiled regex patterns for maximum performance
RE_DOMAIN = re.compile(r"\b(www\.|https?://|\.com|\.org|\.net|\.in|\.co\.in|\.io|\.fr|\.ai)\b", re.IGNORECASE)
RE_NON_ALPHANUM = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")

# Sorted multi-word legal suffixes by length descending so longer phrases match first
SORTED_LEGAL_SUFFIXES = sorted(LEGAL_SUFFIXES, key=lambda x: len(x), reverse=True)


def normalize_text(value: str) -> str:
    """
    Standardize raw text:
    - Handle None/empty
    - NFKD Unicode normalization (decomposing accents, umlauts, etc.)
    - Lowercasing
    - Strip common domain extensions
    - Replace non-alphanumeric characters with space
    - Collapse redundant whitespace and trim
    """
    if value is None:
        return ""

    value = str(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = RE_DOMAIN.sub(" ", value)
    value = RE_NON_ALPHANUM.sub(" ", value)
    value = RE_WHITESPACE.sub(" ", value).strip()
    return value


def extract_name_core(name_norm: str) -> str:
    """
    Extract the core business name by stripping legal entity suffixes and generic corporate noise.
    Example: 'moore bitwise inc' -> 'moore bitwise'
    """
    if not name_norm:
        return ""

    tokens = name_norm.split()
    if not tokens:
        return ""

    # Remove single-token legal suffixes from the tail
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    curr_name = " ".join(tokens)

    # Remove multi-word legal suffixes from the end or start
    for suffix in SORTED_LEGAL_SUFFIXES:
        if " " in suffix:
            if curr_name.endswith(" " + suffix):
                curr_name = curr_name[: -len(suffix) - 1].strip()
            elif curr_name.startswith(suffix + " "):
                curr_name = curr_name[len(suffix) + 1 :].strip()

    return curr_name if curr_name else name_norm


def normalize_address(addr: str) -> str:
    """
    Standardize address string and expand common street/building abbreviations.
    Example: '337 Oakland Ave' -> '337 oakland avenue'
    """
    cleaned = normalize_text(addr)
    if not cleaned:
        return ""

    tokens = cleaned.split()
    normalized_tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(normalized_tokens)


def normalize_country(country: str) -> str:
    """
    Standardize country string, preserving unseen country labels.
    """
    if country is None:
        return ""
    return str(country).strip()


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create normalized representations for business name, address, and country.
    """
    df = df.copy()

    # Ensure required base columns exist
    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""

    # 1. Business Name Normalization
    raw_names = df["business_name"].fillna("").tolist()
    norm_names = [normalize_text(n) for n in raw_names]
    core_names = [extract_name_core(n) for n in norm_names]
    compact_names = [n.replace(" ", "") for n in norm_names]

    df["name_norm"] = norm_names
    df["name_core"] = core_names
    df["name_compact"] = compact_names

    # 2. Address Normalization
    raw_addrs = df["business_address"].fillna("").tolist()
    norm_addrs = [normalize_address(a) for a in raw_addrs]
    compact_addrs = [a.replace(" ", "") for a in norm_addrs]

    df["address_norm"] = norm_addrs
    df["address_compact"] = compact_addrs

    # 3. Country Normalization
    df["country_norm"] = [normalize_country(c) for c in df["country"]]

    return df
