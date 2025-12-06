from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from loguru import logger

def evaluate_sts(
    model_path: str,
    csv_path: str = "dataset_JA_Clinical_STS.csv",
) -> tuple[float, float]:
    """
    STSデータセットでモデルを評価
    
    Args:
        model_path: モデルのパス（"intfloat/multilingual-e5-large" または "./trained_model"）
        csv_path: STSデータセットのCSVファイルパス
    """
    # モデルの読み込み
    logger.info(f"モデルを読み込み中: {model_path}")
    model = SentenceTransformer(model_path)
    logger.info("モデル読み込み完了")
    
    # データセットの読み込み
    logger.info(f"データセットを読み込み中: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=['sentence1', 'sentence2', 'score'])
    logger.info(f"データ読み込み完了: {len(df)}件のペア")
    
    # 文のペアを埋め込みベクトルに変換
    logger.info("埋め込みベクトルを計算中...")
    embeddings1 = model.encode(df['sentence1'].tolist(), normalize_embeddings=True, show_progress_bar=True)
    embeddings2 = model.encode(df['sentence2'].tolist(), normalize_embeddings=True, show_progress_bar=True)
    logger.info("埋め込み計算完了")
    
    # コサイン類似度を計算
    logger.info("コサイン類似度を計算中...")
    cosine_similarities = np.sum(embeddings1 * embeddings2, axis=1)
    logger.info("類似度計算完了")
    
    # Spearman相関係数を計算
    true_scores = df['score'].values
    spearman_corr, p_value = spearmanr(cosine_similarities, true_scores)
    
    # 結果を表示
    logger.info("\n=== 評価結果 ===")
    logger.info(f"Spearman相関係数: {spearman_corr:.4f}")
    logger.info(f"p値: {p_value:.4e}")
    logger.info(f"データ数: {len(df)}")
    logger.info(f"類似度の範囲: [{cosine_similarities.min():.4f}, {cosine_similarities.max():.4f}]")
    logger.info(f"正解スコアの範囲: [{true_scores.min()}, {true_scores.max()}]")
    
    return spearman_corr, p_value

if __name__ == "__main__":
    # 事前学習済みモデルで評価
    logger.info("=== 事前学習済みモデル (intfloat/multilingual-e5-large) の評価 ===")
    evaluate_sts("intfloat/multilingual-e5-large")
    
    # 学習済みモデルの評価
    logger.info("\n=== 学習済みモデル (./trained_model) の評価 ===")
    evaluate_sts("./trained_model")

