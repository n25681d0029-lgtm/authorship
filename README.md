# Định danh tác giả: Stylometry + mô hình ngôn ngữ

Hệ thống phân loại tác giả của một văn bản bằng hai luồng: đặc trưng phong cách (đếm từ chức năng,
dấu câu, độ dài câu, độ phong phú từ vựng) và vector ngữ cảnh từ mô hình ngôn ngữ (RoBERTa, PhoBERT),
hợp nhất rồi phân loại.

## Cài đặt (Windows, Linux, macOS)

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux, macOS
pip install -r requirements.txt
python main.py check
```

Yêu cầu Python 3.10 trở lên. Không có đường dẫn cố định nào trong mã. Muốn đặt dữ liệu và kết quả
ở ổ khác, đặt biến môi trường `AA_HOME` trỏ tới thư mục đó.

## Chạy thử trong 2 phút (không cần Internet)

```
python main.py demo-data --lang en
python main.py baseline --data demo_en.csv
python main.py tiny-backbone --data demo_en.csv
python main.py train --data demo_en.csv --backbone outputs/tiny_backbone --epochs 5 --name demo
python main.py predict --ckpt demo --text "The old house stood on the hill, and the wind was cold."
```

Dữ liệu demo là dữ liệu nhân tạo, chỉ để kiểm tra đường ống chạy đúng.

## Chạy với dữ liệu thật

Ba kiểu dữ liệu được nhận diện tự động:

| Kiểu | Cấu trúc |
|---|---|
| CSV | một file có cột `text`, `author` (tùy chọn `split`, `topic`) |
| C50 | `C50train/<tác giả>/*.txt` và `C50test/<tác giả>/*.txt` |
| Thư mục | `<thư mục>/<tác giả>/*.txt` hoặc có sẵn `train/`, `test/` bên trong |

```
python scripts/download_c50.py
python main.py baseline --data data/raw/C50 --fmt c50
python main.py train --data data/raw/C50 --fmt c50 --lang en --fusion concat --epochs 5
python main.py ablation --data data/raw/C50 --fmt c50 --epochs 5
python main.py evaluate --ckpt concat_20250101_120000
python main.py predict --ckpt <tên checkpoint> --file bai_can_kiem_tra.txt --threshold 0.5
```

Tiếng Việt: thêm `--lang vi` (dùng `vinai/phobert-base-v2`) và nên `pip install pyvi` để tách từ.

Thử kịch bản đổi chủ đề giữa train và test (CSV có cột `topic`):
`python main.py train --data du_lieu.csv --test-topics the_thao`
hoặc để mỗi tác giả tự giữ lại khoảng 30% số đoạn thuộc các tác phẩm train không thấy:
`python main.py train --data du_lieu.csv --topic-holdout 0.3`

## Kho dữ liệu tiếng Việt thật (Wikisource, phạm vi công cộng)

> Kho văn xuôi cũ này (v1–v4) đã bị xóa; kho dùng thật là kho v5 theo thể loại ở mục dưới. Các lệnh vẫn chạy được nếu cần dựng lại.

```
pip install beautifulsoup4 pyvi
python scripts/fetch_wikisource.py                      # tải văn xuôi, có cache, chạy lại không tải lại
python main.py build-corpus --src data/raw/wikisource --out data/raw/vi_wikisource.csv --per-author 200
python main.py experiment --data data/raw/vi_wikisource.csv --lang vi --name vi_wikisource --epochs 8 --patience 3
```

- `fetch_wikisource.py` chỉ lấy tác giả mất trước 1976 (hết bảo hộ theo Luật SHTT Việt Nam), kiểm tra
  tên tác giả trong header của từng trang để loại bản dịch và bài của người khác, bỏ thơ.
  Kết quả: `data/raw/wikisource/<tác giả>/<tác phẩm>/<trang>.txt` và `manifest.csv` (URL nguồn).
- `build-corpus` ghép đoạn văn tự nhiên thành đoạn 50–200 từ (âm tiết), khử trùng lặp, mỗi tác giả tối đa
  `--per-author` đoạn (mặc định 300) chia đều cho các tác phẩm, loại tác giả dưới 100 đoạn. Cột: `text, author, topic, doc_id, n_words, source`
  (`topic` = tác phẩm, `doc_id` = bài/chương).
- Khi CSV có cột `doc_id`, mọi lệnh chia dữ liệu theo bài: các đoạn của cùng một bài luôn nằm chung một tập.

## Kho v5 theo thể loại (Thơ, Tự sự, Ký, Nghị luận)

Thể loại định nghĩa trong `data/genres.json` (nhãn gán cho từng tác phẩm). Tác giả ứng viên: `data/v5_sources.json`.

```
python scripts/fetch_v5.py survey          # liệt kê tác phẩm + đề xuất thể loại -> data/v5_works.csv
#   duyệt cột genre trong data/v5_works.csv (tho | tu_su | ky | nghi_luan | bo | để trống)
python scripts/fetch_v5.py fetch           # tải -> data/v5_genre/raw/<thể loại>/<tác giả>/<tác phẩm>/*.txt
python main.py build-genre-corpus          # -> data/v5_corpus.csv (+ .stats.json)
```
Nguồn: trang văn bản vi.wikisource, trang "Trang:" của bản chụp sách đã được soát (chất lượng ≥ 3), hoặc văn bản tự đặt vào
`data/v5_genre/raw/...` (ghi nguồn và giấy phép vào `manifest.csv`). Lưu đệm API ở `data/v5_genre/cache` (xóa để tải mới).
Đoạn có hơn 3% âm tiết sai cấu trúc tiếng Việt bị loại (`--max-invalid-ratio`), để bản chép OCR không thành "vân tay" giả.

## Giao diện demo (Streamlit)

```
..\.venv\Scripts\streamlit run app.py
```
Mở http://localhost:8501. Thanh bên chọn checkpoint (mặc định `best_model`), ngưỡng từ chối, số tác giả hiển thị.
- **Đoán tác giả:** thẻ kết quả, thanh xác suất, kết quả từng đoạn khi văn bản dài, nút lưu vào yêu thích; mục
  "Câu nào khiến máy nghĩ là X" tô màu từng câu theo mức ảnh hưởng (bỏ từng câu rồi đoán lại, `src/influence.py`);
  câu giải thích tiếng Việt từ các đặc trưng phong cách lệch nhiều nhất (`src/explain.py`).
- **Xem từng bước:** dán văn bản hoặc bấm “Lấy một đoạn mẫu” (đoạn máy chưa từng học, có đáp án), rồi chọn
  các bước muốn xem: 1 làm sạch văn bản, 2 chia đoạn, 3 tách từ cho PhoBERT, 4 đo cách viết, 5 chấm điểm (mỗi
  phần góp bao nhiêu điểm), 6 tính xác suất, 7 kết luận, 8 so với từng tác giả. Số liệu do `src/pipeline_trace.py`
  tính, trùng với lần dự đoán thật; tác giả, thể loại lấy từ checkpoint nên thêm dữ liệu rồi huấn luyện lại
  không phải sửa giao diện.
- **Tải file:** `.txt` (một văn bản) hoặc `.csv` có cột `text` (nhiều văn bản; nếu có cột `author` thì tính độ chính xác),
  tải kết quả về dạng CSV.
- **Học viết theo tác giả** (`app_style.py`, `src/style_guide.py`): hồ sơ văn phong (từ đặc trưng tính bằng log-odds
  so với tác giả cùng thể loại, cách mở câu, từ nối, độ dài câu, lời thoại, thể thơ), dàn ý cho một đề bài theo độ dài
  người dùng chọn (chữ hoặc ký tự), kiểm tra bài người dùng tự viết (giống tác giả bao nhiêu %, gợi ý sửa, câu nào
  giống nhất), và máy viết thử bằng GPT nhóm tự huấn luyện trong `../style_gpt` (nếu đã có mô hình cho tác giả đó).
- **So sánh hai văn bản** (`app_compare.py`, `src/compare.py`): hai đoạn có cùng người viết không — độ giống hai
  phân bố tác giả, đối chiếu với các cặp đoạn trong tập kiểm tra.
- **Lịch sử** (`app_history.py`, `src/history.py`): mọi lần dùng được ghi vào `outputs/app_data/lich_su.db` (SQLite,
  chỉ trên máy), xem theo 7 ngày / 30 ngày / 1 năm, gom theo ngày / tuần / tháng / năm, tìm kiếm, ghi chú, yêu thích,
  dùng lại. Mục không yêu thích tự xóa sau 7 / 30 / 90 / 365 ngày (người dùng chọn, hoặc không bao giờ).
- **Về mô hình:** accuracy, macro-F1, precision/recall từng tác giả, ma trận nhầm lẫn, thông số hiệu chỉnh, bộ gộp.

Giao diện (`app_theme.py`) chỉ dùng CSS: hiệu ứng hiện dần, nút nổi khi rê chuột, thanh xác suất chạy; tắt được ở
thanh bên. Không gọi dịch vụ AI bên ngoài.

## Bộ gộp (ensemble)

```
python main.py train --data data/v5_corpus.csv --lang vi --fusion concat --epochs 10 --patience 3 --oversample --class-bias --seed 43 --split-seed 42 --name best_model_s43
python main.py ensemble --ckpt best_model --extra best_model_s43 best_model_s44 --save
```
`--split-seed` giữ nguyên cách chia dữ liệu của best_model, chỉ đổi khởi tạo. `ensemble` học trọng số (log-loss trên
val) giữa trung bình các mô hình lai và SVM n-gram ký tự / từ, hiệu chỉnh lại nhiệt độ + ngưỡng, so trên test (có
khoảng tin cậy bootstrap theo bài) và ghi `best_model/ensemble.joblib`. Có file này thì `AuthorPredictor`, app và tab
Xem từng bước tự dùng bộ gộp; `from_checkpoint(..., use_ensemble=False)` để dùng riêng mô hình chính.

## Dự đoán, từ chối "unknown" và n-gram ký tự

- `predict`: văn bản dài hơn 200 âm tiết được cắt thành các đoạn cỡ dữ liệu huấn luyện, xác suất là trung bình các
  đoạn (in kết quả từng đoạn). Xác suất đã được hiệu chỉnh nhiệt độ (temperature scaling) trên tập val.
  `--threshold auto` (mặc định) dùng ngưỡng từ chối học trên val (chấp nhận 95% đoạn val); `--threshold 0` tắt từ chối.
- `openset --ckpt <ckpt> --unknown <thư mục .txt của tác giả ngoài kho>`: đo AUROC và tỷ lệ từ chối đoạn lạ / chấp
  nhận đoạn đã biết theo ngưỡng, ví dụ thư mục .txt của một tác giả không có trong kho.
- `train --oversample --class-bias` (cấu hình của best_model hiện tại): rút đoạn train cân bằng giữa tác giả và học độ
  ưu tiên từng tác giả trên tập val → tăng độ nhạy cho tác giả ít dữ liệu. `--class-weight`: trọng số lớp trong hàm mất mát.
- Giao diện: bước 1 chọn thể loại của văn bản (danh sách tác giả mỗi thể loại lưu trong `meta.json` của checkpoint khi
  huấn luyện trên kho có cột `genre`), có thể thu hẹp thêm bằng cách chọn tay tác giả nghi vấn.

## Thí nghiệm theo thể loại (kho v5)

```
python main.py genre-experiment --data data/v5_corpus.csv --epochs 10 --patience 3 --oversample --class-bias --seeds 42 43 44
python main.py train --data data/v5_corpus.csv --lang vi --fusion concat --oversample --class-bias --epochs 10 --patience 3 --name best_model
```

So sánh trên cùng đoạn test: mô hình riêng từng thể loại, mô hình chung lọc theo thể loại, không lọc, chọn sai thể loại,
và chéo thể loại (mô hình thể loại X đoán văn thể loại Y của cùng tác giả). Kết quả: `outputs/experiments/v5_the_loai/summary.md`.
- `train --char-ngram` (hoặc tên `gated+ng`, `concat+ng`, `style_only+ng` trong `experiment --fusions`): ghép xác suất
  của bộ phân loại n-gram ký tự (out-of-fold theo tác phẩm) vào luồng phong cách.

Nhật ký các vấn đề, chẩn đoán và cách sửa: `../report/nhat_ky_ky_thuat.md`.

## Thí nghiệm đánh giá (`experiment`)

Một lệnh chạy cả 3 baseline và 4 chế độ mạng lai (ablation) trên **cùng một cách chia** cho hai kịch bản:
`random` (chia theo bài, phân tầng theo tác giả) và `cross_topic` (test là tác phẩm train chưa thấy).
Mỗi kịch bản còn đánh giá đoạn test bị cắt còn 10, 25, 50 từ đầu (văn bản ngắn) và độ chính xác theo nhóm độ dài.
Kết quả ở `outputs/experiments/<tên>/`: `summary.md`, `results.csv`, và trong từng kịch bản
`split.csv`, `split_stats.json`, `by_length.csv`, `details.json` (cặp nhầm nhiều nhất, trọng số cổng, thời gian),
`confusion/<mô hình>_confusion.png|csv`. Mặc định xóa trọng số mô hình sau khi đánh giá (thêm `--keep-checkpoints` để giữ).

## Cấu trúc

```
main.py                 điểm vào, các lệnh: check, demo-data, tiny-backbone, baseline, train, ablation,
                        experiment, build-corpus, evaluate, predict, openset
config/settings.py      đường dẫn (tính từ vị trí file) và siêu tham số
src/text_utils.py       chuẩn hóa Unicode, tách từ, tách câu, tách từ tiếng Việt
src/data_loader.py      đọc CSV, C50, thư mục; chia train/val/test theo bài (doc_id), giữ lại chủ đề
src/corpus_builder.py   ghép đoạn 50-200 từ, khử trùng lặp, cân bằng tác giả và tác phẩm
src/experiment.py       baseline + ablation trên cùng cách chia, chéo chủ đề, văn bản ngắn
src/calibration.py      hiệu chỉnh nhiệt độ, chọn ngưỡng từ chối, ECE
src/ngram_features.py   n-gram ký tự out-of-fold cho luồng phong cách
src/openset.py          đánh giá từ chối "unknown" với tác giả ngoài kho
scripts/fetch_wikisource.py  tải văn xuôi tiếng Việt thuộc phạm vi công cộng
src/feature_extractor.py  đặc trưng phong cách, z-score, giải thích
src/dataset.py          Dataset và Collator (token hóa theo batch)
src/models.py           StyleEncoder và HybridAuthorModel (concat, gated, lm_only, style_only)
src/train.py            vòng huấn luyện, dừng sớm, lưu checkpoint tự chứa
src/evaluate.py         accuracy, macro-F1, ma trận nhầm lẫn
src/baselines.py        Burrows' Delta, char n-gram + SVM, stylometry + LogReg
src/predict.py          nạp checkpoint, dự đoán, đánh giá lại
src/demo_data.py        dữ liệu nhân tạo để chạy thử
src/tiny_backbone.py    BERT nhỏ ngẫu nhiên cho kiểm thử offline
src/ensemble.py         bộ gộp mô hình lai + SVM n-gram (học trọng số trên val), thí nghiệm so sánh
src/pipeline_trace.py   số liệu từng bước cho tab Xem từng bước
src/influence.py        câu nào ảnh hưởng nhiều nhất (che từng câu)
src/style_guide.py      hồ sơ văn phong, kiểm tra bài viết, dàn ý
src/compare.py          so sánh hai văn bản
src/history.py          lịch sử + yêu thích (SQLite), tự dọn
app.py, app_*.py        giao diện Streamlit (app_theme: CSS; app_pipeline, app_style, app_compare, app_history: các tab)
scripts/download_c50.py tải Reuters C50
tests/                  kiểm thử tự động
```

## Kiểm thử

```
python -m unittest discover -s tests -v
```

## Chép sang máy khác

Chép nguyên thư mục (có thể bỏ `.venv`). Checkpoint trong `outputs/checkpoints/<tên>/` là thư mục tự
chứa (trọng số, cấu hình, tokenizer, trạng thái đặc trưng) nên dự đoán được mà không cần tải lại backbone.
