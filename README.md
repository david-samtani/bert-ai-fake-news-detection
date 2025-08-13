BERT Fake News Detection (Kaggle 2018)


Detects real vs. fake news headlines/articles by fine-tuning a BERT classifier on the Kaggle Fake News dataset. Built and evaluated end-to-end in Google Colab with GPU acceleration.

🔎 Project at a Glance
Task: Binary text classification (real vs. fake news)

Model: BERT for Sequence Classification (Hugging Face Transformers)

Data: Kaggle Fake News

Notebook: bert_fake_news_detection_kaggle2018_colab.ipynb (Colab-ready)

Result: Test accuracy ≈ 0.574 on a held-out split (baseline fine-tune; minimal tuning). 

🧰 Tech Stack
Python, PyTorch

Hugging Face transformers / datasets

Google Colab GPU

Pandas, scikit-learn (for basic metrics/inspection)

📂 Repo Structure
bash
Copy
Edit
.
├── bert_fake_news_detection_kaggle2018_colab.ipynb   # Main Colab notebook
├── Fake_News_Detection_with_BERT.pdf                 # Short paper / write-up
└── README.md
🚀 Quickstart (Colab)
Open the notebook in Colab (badge above).

Install & import deps (cells provided).

Dataset access (Kaggle API)

Upload your kaggle.json when prompted.

The notebook moves it to ~/.kaggle/kaggle.json and sets permissions.

Automatically downloads and unzips train.csv / test.csv.

Preprocess

Keep non-empty text rows.

Rename label column to labels for HF compatibility.

Tokenize with bert-base-uncased, padding='max_length', truncation=True.

Split & Loaders

80/10/10 split via Dataset.train_test_split.

PyTorch DataLoaders for train/val/test.

Train

Fine-tune BERT with Adam (lr=1e-5).

Periodic checkpointing to Drive (optional).

Evaluate

Compute accuracy on the held-out test set (printed at the end).

⚠️ Note: This baseline intentionally uses minimal hyperparameter tuning; accuracy can be improved with better schedules, regularization, and class-imbalance handling.

📊 Results (Baseline)
Accuracy: ~0.574 (held-out test split). 

Interpretation: Slightly above chance for a first-pass fine-tune; a good starting point for iterative improvements.

🧪 Ideas to Improve
Model: Try roberta-base, deberta-v3-base, or domain-adapt via continued pretraining.

Tuning: Warmup + cosine decay; different LRs for encoder vs. classifier; weight decay; early stopping.

Data: Balance classes; longer max sequence length; headline+body concatenation; text cleaning.

Eval: Add precision/recall/F1 and a confusion matrix; k-fold cross-validation.

📝 Paper
A short write-up describing motivation, methods, and results is included: Fake_News_Detection_with_BERT.pdf. 

✅ Requirements (for local runs)
Python 3.8+

pip install transformers datasets pandas torch

Kaggle API credentials (kaggle.json) to download data

🔒 Notes on Colab Upload Widget
The notebook uses Colab’s files.upload() only to provide the Kaggle API key; it does not depend on ipywidgets, so GitHub rendering should work once widget metadata is stripped (handled in repo).
