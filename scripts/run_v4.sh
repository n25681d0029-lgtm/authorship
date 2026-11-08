#!/usr/bin/env bash
# Dựng kho v4 (~30 tác giả), chạy baseline và huấn luyện mô hình lai trên cùng cách chia (seed 42).
# Chạy từ thư mục authorship_attribution:  bash scripts/run_v4.sh <số đoạn tối thiểu mỗi tác giả>
set -e
MIN=${1:-40}
PY=../.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONWARNINGS=ignore
$PY main.py build-corpus --src data/raw/wikisource --out data/raw/vi_wikisource_v4.csv --per-author 200 --min-per-author $MIN
$PY main.py baseline --data data/raw/vi_wikisource_v4.csv > outputs/baseline_v4.txt 2>&1
cat outputs/baseline_v4.txt | grep -v Warn
[ -d outputs/checkpoints/best_model ] && [ ! -d outputs/checkpoints/best_model_v3_12tg ] && mv outputs/checkpoints/best_model outputs/checkpoints/best_model_v3_12tg
$PY main.py train --data data/raw/vi_wikisource_v4.csv --lang vi --fusion concat --char-ngram --class-weight \
    --epochs 10 --patience 3 --name best_model > outputs/train_best_model_v4.log 2>&1
grep -E "Dữ liệu:|N-gram|Trọng số lớp|Hiệu chỉnh|TEST \||Đã lưu" outputs/train_best_model_v4.log
