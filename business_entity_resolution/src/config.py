"""
Configuration module for Business Entity Resolution pipeline.
Defines file paths, hyperparameters, preprocessing constants, and model configuration.
"""

from pathlib import Path

# Base Directory Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Default dataset search paths
POSSIBLE_DATA_DIRS = [
    WORKSPACE_ROOT / "dataset",
    PROJECT_ROOT / "dataset",
    Path("dataset"),
]

DATASET_DIR = next((p for p in POSSIBLE_DATA_DIRS if p.exists()), POSSIBLE_DATA_DIRS[0])
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Output and Model Paths
OUTPUT_DIR = WORKSPACE_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

MATCHING_RESULTS_FILENAME = "matching_results.tsv"
CANDIDATE_PAIRS_FILENAME = "candidate_pairs.tsv"
MODEL_FILENAME = "business_matcher.pkl"

# Pipeline Hyperparameters
RANDOM_STATE = 42
MAX_CANDIDATES_PER_S1 = 30
TFIDF_TOP_K = 15
DEFAULT_DECISION_THRESHOLD = 0.65
BATCH_SIZE_FEATURES = 100000
INFERENCE_CHUNK_SIZE = 50000

# Legal Entity Suffixes to normalize and strip for core business name extraction
LEGAL_SUFFIXES = {
    "inc", "inc.", "incorporated",
    "corp", "corp.", "corporation",
    "llc", "l.l.c.", "limited liability company",
    "ltd", "ltd.", "limited",
    "pvt", "pvt.", "private",
    "co", "co.", "company",
    "gmbh", "sa", "sarl", "sas", "spa", "bv", "nv",
    "llp", "l.l.p.", "pllc",
    "enterprise", "enterprises",
    "services", "solutions", "group", "holdings",
    "technologies", "tech", "international", "intl",
    "industries", "associates", "consulting",
    "and sons", "& sons", "and co", "& co",
    "center", "centre", "ventures", "partners"
}

# Street and Address Abbreviations for standardization
ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "st.": "street",
    "str": "street",
    "rd": "road",
    "rd.": "road",
    "ave": "avenue",
    "ave.": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "ct": "court",
    "ct.": "court",
    "pl": "place",
    "pl.": "place",
    "hwy": "highway",
    "hwy.": "highway",
    "pkwy": "parkway",
    "pkwy.": "parkway",
    "cir": "circle",
    "cir.": "circle",
    "apt": "apartment",
    "apt.": "apartment",
    "ste": "suite",
    "ste.": "suite",
    "fl": "floor",
    "fl.": "floor",
    "bldg": "building",
    "bldg.": "building",
    "dept": "department",
    "opp": "opposite",
    "opp.": "opposite",
    "nr": "near",
    "nr.": "near",
    "ext": "extension",
    "ext.": "extension",
    "sq": "square",
    "sq.": "square",
    "rue": "rue",
    "bd": "boulevard",
    "all": "allee"
}

# LightGBM Classifier Parameters (Apache 2.0 license, compliant with competition constraints)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 400,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}
