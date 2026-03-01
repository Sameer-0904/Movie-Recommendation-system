# recommend.py
import os
import joblib
import logging
import gdown

# ==============================
# 🔑 GOOGLE DRIVE FILE IDS
# ==============================

DF_FILE_ID = "1zt4D2gkeB4Gg6Ld9RFvh6wRAedqVHKO1"
COSINE_FILE_ID = "1i2JpskuFc-Pf-7fq0-OelYllZDhcoOUM"

DF_PATH = "df_cleaned_compressed.pkl"
COSINE_PATH = "cosine_sim_compressed.pkl"

# ==============================
# Logging Setup
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ==============================
# Download if not exists
# ==============================

def download_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    logging.info(f"⬇️ Downloading {output} from Google Drive...")
    gdown.download(url, output, quiet=False)
    logging.info(f"✅ {output} downloaded successfully.")

if not os.path.exists(DF_PATH):
    download_from_drive(DF_FILE_ID, DF_PATH)

if not os.path.exists(COSINE_PATH):
    download_from_drive(COSINE_FILE_ID, COSINE_PATH)

# ==============================
# Load Data
# ==============================

logging.info("🔁 Loading data...")
try:
    df = joblib.load(DF_PATH)
    cosine_sim = joblib.load(COSINE_PATH)
    logging.info("✅ Data loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise e


# ==============================
# Recommendation Function
# ==============================

def recommend_movies(movie_name, top_n=5):
    logging.info("🎬 Recommending movies for: '%s'", movie_name)
    idx = df[df['title'].str.lower() == movie_name.lower()].index

    if len(idx) == 0:
        logging.warning("⚠️ Movie not found in dataset.")
        return None

    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    logging.info("✅ Top %d recommendations ready.", top_n)

    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."

    return result_df
