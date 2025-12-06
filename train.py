from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from loguru import logger
import os

# モデルとデータセットの準備
model = SentenceTransformer("intfloat/multilingual-e5-large")

# CSVファイルから読み込む（存在する場合）、なければ辞書形式で作成
csv_path = "train.csv"
if os.path.exists(csv_path):
    logger.info(f"CSVファイルからデータセットを読み込み中: {csv_path}")
    train_dataset = Dataset.from_csv(csv_path)
    logger.info(f"✓ {len(train_dataset)}件のサンプルを読み込みました")
else:
    logger.info("CSVファイルが見つかりません。デフォルトのデータセットを使用します。")
    train_dataset = Dataset.from_dict({
        "anchor": ["It's nice weather outside today.", "He drove to work."],
        "positive": ["It's so sunny.", "He took the car to the office."],
    })
loss = losses.MultipleNegativesRankingLoss(model)

# 保存先を明示的に指定
output_dir = "./trained_model"
training_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_strategy="epoch",  # エポック終了時に保存
    logging_steps=1,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss=loss,
)

# 学習実行
result = trainer.train()
logger.info(f"result: {result}")

# 学習後のモデルを保存（明示的に保存）
model.save(output_dir)
logger.info(f"学習されたモデルは '{output_dir}' に保存されました")