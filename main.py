"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

# === 0. Импорты ===
import pandas as pd
import numpy as np
import re
import os
import torch
import random
import tempfile  # ← Добавлено для безопасного создания временной папки
from sklearn.model_selection import GroupShuffleSplit
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMRanker, early_stopping
from catboost import CatBoostRanker, Pool
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Отключаем предупреждения tokenizers о параллелизме при fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === 1. Фиксированный сид ===
RANDOM_SEED = 993
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# === 2. Вспомогательные функции ===
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_for_bm25(text):
    if pd.isna(text) or str(text).lower() in ("none", "nan", ""):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_for_sbert(text):
    if pd.isna(text) or str(text).lower() in ("none", "nan", ""):
        return ""
    return str(text)

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def tokenize(text, min_len=2):
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) >= min_len]

def make_product_shape(row):
    title = safe_str(row["product_title"])
    desc = safe_str(row.get("product_description", ""))
    bullet = safe_str(row.get("product_bullet_point", ""))
    return " ".join([title, desc, bullet]).strip()

# === 3. Классы ===

class InMemoryCrossEncoder:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.device = next(model.parameters()).device

    def predict(self, sentence_pairs, batch_size=32, show_progress_bar=True):
        scores = []
        for i in tqdm(range(0, len(sentence_pairs), batch_size), disable=not show_progress_bar, desc="CrossEncoder"):
            batch = sentence_pairs[i:i+batch_size]
            enc = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**enc)
                batch_scores = outputs.logits.squeeze().cpu().tolist()
                if isinstance(batch_scores, list):
                    scores.extend(batch_scores)
                else:
                    scores.append(batch_scores)
        return np.array(scores)


class PairwiseRankingDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        query, pos_doc, neg_doc = self.pairs[idx]
        pos_enc = self.tokenizer(query, pos_doc, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        neg_enc = self.tokenizer(query, neg_doc, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids_pos": pos_enc["input_ids"].flatten(),
            "attention_mask_pos": pos_enc["attention_mask"].flatten(),
            "input_ids_neg": neg_enc["input_ids"].flatten(),
            "attention_mask_neg": neg_enc["attention_mask"].flatten(),
        }

# === 4. Функция сохранения submission ===
def create_submission(predictions):
    submission = pd.DataFrame({
        "id": test_df["id"],
        "prediction": predictions
    })
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission файл сохранен: {submission_path}")
    return submission_path

# === 5. Основная функция ===
def main():
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    global test_df

    # === Загрузка данных ===
    print("Загружаем данные...")
    df_train = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    df_train["product_brand"] = df_train["product_brand"].fillna("unknown_brand")
    df_train["product_color"] = df_train["product_color"].fillna("unknown_color")
    test_df["product_brand"] = test_df["product_brand"].fillna("unknown_brand")
    test_df["product_color"] = test_df["product_color"].fillna("unknown_color")
    
    df_train["product_text"] = df_train.apply(make_product_shape, axis=1)
    test_df["product_text"] = test_df.apply(make_product_shape, axis=1)
    
    # === Создание ranking-пар ===
    print("Создаём ranking-пары...")
    pairs = []
    for qid, group in df_train.groupby("query_id"):
        if len(group) < 2:
            continue
        group = group.sort_values("relevance", ascending=False).reset_index(drop=True)
        rels = group["relevance"].values
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if rels[i] > rels[j]:
                    pairs.append((
                        group["query"].iloc[i],
                        group["product_text"].iloc[i],
                        group["product_text"].iloc[j]
                    ))
    print(f"Создано {len(pairs)} пар")
    
    # === Дообучение cross-encoder ===
    print("Дообучаем cross-encoder...")
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    dataset = PairwiseRankingDataset(pairs, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    margin = 1.0
    model.train()
    
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids_pos = batch["input_ids_pos"].to(device)
        attention_mask_pos = batch["attention_mask_pos"].to(device)
        input_ids_neg = batch["input_ids_neg"].to(device)
        attention_mask_neg = batch["attention_mask_neg"].to(device)
    
        pos_scores = model(input_ids_pos, attention_mask=attention_mask_pos).logits.squeeze()
        neg_scores = model(input_ids_neg, attention_mask=attention_mask_neg).logits.squeeze()
        loss = torch.mean(torch.clamp(margin - (pos_scores - neg_scores), min=0))
        loss.backward()
        optimizer.step()
    
    print("✅ Дообучение завершено")
    
    # === Инициализация моделей для инференса ===
    cross_encoder = InMemoryCrossEncoder(model, tokenizer)
    sbert_model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
    
    # === Подготовка глобальных эмбеддингов и BM25 ===
    unique_queries = pd.concat([df_train["query"], test_df["query"]]).drop_duplicates().apply(clean_for_sbert).tolist()
    unique_products = pd.concat([df_train["product_text"], test_df["product_text"]]).drop_duplicates().tolist()
    all_texts = list(set(unique_queries + unique_products))
    embeddings = sbert_model.encode(all_texts, batch_size=256, convert_to_numpy=True, device=device)
    text_to_emb = dict(zip(all_texts, embeddings))
    
    def build_global_bm25(df):
        doc_ids = []
        texts = []
        for _, row in df.iterrows():
            full = row["product_text"]
            doc_ids.append(row["product_id"])
            texts.append(full)
        tokenized = [tokenize(clean_for_bm25(t)) for t in texts]
        bm25 = BM25Okapi(tokenized)
        return bm25, doc_ids
    
    global_bm25, global_doc_ids = build_global_bm25(df_train)
    global_doc_id_to_idx = {pid: i for i, pid in enumerate(global_doc_ids)}
    
    # === Разбиение на train/val ===
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss.split(df_train, groups=df_train["query_id"]))
    df_train_split = df_train.iloc[train_idx].reset_index(drop=True)
    df_val_split = df_train.iloc[val_idx].reset_index(drop=True)
    
    # === Функция извлечения признаков ===
    def extract_features_with_cross_encoder(df):
        feature_rows = []
        pairs = []
        for qid, group in df.groupby("query_id"):
            query_raw = group["query"].iloc[0]
            for _, row in group.iterrows():
                pairs.append((query_raw, row["product_text"]))
        cross_scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
        pair_idx = 0
        for qid, group in df.groupby("query_id"):
            query_raw = group["query"].iloc[0]
            query_sbert = clean_for_sbert(query_raw)
            query_bm25 = clean_for_bm25(query_raw)
            q_tokens = tokenize(query_bm25)
            docs_sbert = []
            docs_bm25 = []
            product_ids = []
            doc_embs = []
            cross_vals = []
            for _, row in group.iterrows():
                full_sbert = row["product_text"]
                full_bm25 = clean_for_bm25(full_sbert)
                docs_sbert.append(full_sbert)
                docs_bm25.append(full_bm25)
                product_ids.append(row["product_id"])
                doc_embs.append(text_to_emb[full_sbert])
                cross_vals.append(cross_scores[pair_idx])
                pair_idx += 1
            query_emb = text_to_emb[query_sbert]
            sbert_sims = cosine_similarity([query_emb], doc_embs).ravel()
            tokenized_local = [tokenize(d) for d in docs_bm25]
            bm25_local_scores = BM25Okapi(tokenized_local).get_scores(q_tokens) if any(tokenized_local) else np.zeros(len(docs_bm25))
            tfidf_sims = np.zeros(len(docs_bm25))
            if any(docs_bm25):
                try:
                    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1, max_features=10000)
                    X_docs = tfidf.fit_transform(docs_bm25)
                    X_q = tfidf.transform([query_bm25])
                    tfidf_sims = cosine_similarity(X_docs, X_q).ravel()
                except:
                    pass
            global_scores = []
            for pid in product_ids:
                idx = global_doc_id_to_idx.get(pid, -1)
                score = global_bm25.get_scores(q_tokens)[idx] if idx != -1 else 0.0
                global_scores.append(float(score))
            for i, (_, row) in enumerate(group.iterrows()):
                d_tokens = tokenize(docs_bm25[i])
                overlap = len(set(q_tokens) & set(d_tokens)) / max(1, len(set(q_tokens)))
                brand_match = 1 if row["product_brand"].lower() in q_tokens else 0
                color_match = 1 if row["product_color"].lower() in q_tokens else 0
                qlen = len(q_tokens)
                tlen = len(tokenize(row["product_title"]))
                dlen = len(tokenize(row.get("product_description", "")))
                blen = len(tokenize(row.get("product_bullet_point", "")))
                has_desc = 1 if pd.notna(row.get("product_description")) and safe_str(row["product_description"]).strip() else 0
                has_bullet = 1 if pd.notna(row.get("product_bullet_point")) and safe_str(row["product_bullet_point"]).strip() else 0
                feat = {
                    "bm25_local": float(bm25_local_scores[i]),
                    "bm25_global": float(global_scores[i]),
                    "tfidf_cosine": float(tfidf_sims[i]),
                    "cross_encoder_score": float(cross_vals[i]),
                    "overlap_token_ratio": float(overlap),
                    "brand_in_query": int(brand_match),
                    "color_in_query": int(color_match),
                    "query_len_tokens": int(qlen),
                    "title_len_tokens": int(tlen),
                    "desc_len_tokens": int(dlen),
                    "bullet_len_tokens": int(blen),
                    "text_total_tokens": int(tlen + dlen + blen),
                    "has_desc": int(has_desc),
                    "has_bullet": int(has_bullet)
                }
                feature_rows.append(feat)
        return pd.DataFrame(feature_rows)
    
    # === Извлечение признаков ===
    print("Извлекаем признаки...")
    X_train = extract_features_with_cross_encoder(df_train_split)
    X_val = extract_features_with_cross_encoder(df_val_split)
    X_test = extract_features_with_cross_encoder(test_df)
    
    y_train = df_train_split["relevance"].values.astype(np.float32)
    y_val = df_val_split["relevance"].values.astype(np.float32)
    train_groups = df_train_split.groupby("query_id").size().values
    val_groups = df_val_split.groupby("query_id").size().values
    
    # === LightGBM ===
    lgb_ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.1,
        min_data_in_leaf=15,
        feature_fraction=0.85,
        random_state=RANDOM_SEED,
        verbose=0
    )
    lgb_ranker.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # === CatBoost (исправлено: временная директория) ===
    catboost_train_dir = tempfile.mkdtemp(prefix="catboost_")
    print(f"CatBoost использует временную директорию: {catboost_train_dir}")

    train_pool = Pool(X_train, y_train, group_id=df_train_split["query_id"].values)
    val_pool = Pool(X_val, y_val, group_id=df_val_split["query_id"].values)
    cb_ranker = CatBoostRanker(
        loss_function="YetiRank",
        iterations=500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=RANDOM_SEED,
        verbose=0,
        use_best_model=True,
        early_stopping_rounds=30,
        train_dir=catboost_train_dir  # ← безопасный путь
    )
    cb_ranker.fit(train_pool, eval_set=val_pool)
    
    # === Ансамбль и submission ===
    print("Делаем предсказания на тесте...")
    pred_lgb_test = lgb_ranker.predict(X_test)
    pred_cb_test = cb_ranker.predict(X_test)
    final_pred = 0.91 * pred_cb_test + 0.09 * pred_lgb_test
    
    create_submission(final_pred)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()