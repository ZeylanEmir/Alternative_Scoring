# Alternative scoring for sole proprietors (transactions + behavioral) VS classical

## Quick start
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt

# Train
python -m src.main --mode train --data_dir data/raw --artifacts_dir artifacts --seed 42

# Predict (CSV/PDF)
python -m src.main --mode predict --input_file .\samples\disciplined_statement.csv --artifacts_dir artifacts --gauge_png artifacts\credit_gauge.png
