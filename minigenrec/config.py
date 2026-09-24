from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

POSITIVE_MIN_RATING = 4

COLD_SPLIT_SEED = 0
TRAIN_SUBSET_SEED = 0
TITLE_SHUFFLE_SEED = 0
BOOTSTRAP_SEED = 0
MODEL_SEEDS = (0, 1, 2)

COLD_RATIO_DEFAULT = 0.10
COLD_RATIO_FALLBACK = 0.20
COLD_MIN_USERS = 1000
COLD_MIN_INTERACTIONS = 2000

N_TRAIN_SIZES = (10_000, 30_000, 100_000)

MAX_EVENTS_K: int = 21

MAX_LEN = 512

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION: str | None = "060db6499f32faf8b98477b0a26969ef7d8b9987"

# Hybrid only: during training, drop each warm item's ID part with this
# probability so text-only (cold) items are scored on the same scale.
ID_DROPOUT = 0.5

TOP_K = 10
BOOTSTRAP_SAMPLES = 1000
