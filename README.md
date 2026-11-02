# Định danh tác giả văn bản tiếng Việt: Stylometry + PhoBERT

Bài tập lớn môn Xử lý ngôn ngữ tự nhiên — nhóm Dương Hoài Nam (trưởng nhóm), Nguyễn Minh Công.

Hệ thống đoán tác giả của một đoạn văn tiếng Việt bằng mạng lai hai luồng: đặc trưng phong cách (từ chức năng, dấu câu,
độ dài câu, độ phong phú từ vựng) và vector ngữ cảnh của PhoBERT (`vinai/phobert-base-v2`), so sánh với 3 baseline
(Burrows' Delta, char n-gram + SVM, stylometry + Logistic Regression). Người dùng chọn **thể loại** của văn bản trước khi
định danh; mô hình chỉ xét các tác giả viết thể loại đó.

Dữ liệu: 4.851 đoạn, 28 tác giả Việt Nam đã mất trước 1976 (tác phẩm thuộc phạm vi công cộng theo Luật Sở hữu trí tuệ,
Điều 27), lấy từ [vi.wikisource.org](https://vi.wikisource.org), chia 3 thể loại: Nghị luận (13 tác giả), Thơ (12), Tự sự (8).

## Bắt đầu nhanh

Yêu cầu Python 3.10+; nên có GPU NVIDIA (huấn luyện khoảng 15–20 phút trên RTX 3060 12GB, CPU chậm hơn nhiều).

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # muốn dùng GPU: cài torch bản CUDA theo hướng dẫn ở pytorch.org trước
python main.py check

# 1. Huấn luyện mô hình dùng cho giao diện (tải PhoBERT ~540MB lần đầu)
python main.py train --data data/v5_corpus.csv --lang vi --fusion concat --oversample --class-bias --epochs 10 --patience 3 --name best_model

# 2. Mở giao diện demo: http://localhost:8501
streamlit run app.py
```

Checkpoint được lưu ở `outputs/checkpoints/best_model/` (không đưa lên GitHub vì nặng ~520MB).

Dự đoán bằng dòng lệnh:
```
python main.py predict --ckpt best_model --file van_ban.txt
```

## Giao diện (Streamlit)

- **Bước 1 (thanh bên):** chọn thể loại của văn bản (Nghị luận / Thơ / Tự sự / Không rõ). Có thể thu hẹp thêm bằng cách
  chọn tay các tác giả nghi vấn.
- **Nhập văn bản:** tác giả dự đoán, độ tin cậy (đã hiệu chỉnh), biểu đồ xác suất, kết quả từng đoạn khi văn bản dài,
  câu giải thích tiếng Việt từ các đặc trưng phong cách nổi bật. Độ tin cậy dưới ngưỡng → "không xác định".
- **Tải file:** `.txt` (một văn bản) hoặc `.csv` có cột `text` (nhiều văn bản; có cột `author` thì tính độ chính xác).
- **Về mô hình:** accuracy, macro-F1, precision/recall từng tác giả, ma trận nhầm lẫn, danh sách tác giả theo thể loại.

## Tái lập thí nghiệm

```
python main.py genre-experiment --data data/v5_corpus.csv --epochs 10 --patience 3 --oversample --class-bias --seeds 42 43 44
```

Khoảng 30 phút mỗi seed trên RTX 3060. Trên cùng các đoạn test, so sánh: mô hình riêng từng thể loại, mô hình chung lọc
theo thể loại, mô hình chung không lọc, chọn sai thể loại, và chéo thể loại (mô hình học thể loại X đoán văn thể loại Y
của cùng tác giả). Kết quả ghi vào `outputs/experiments/v5_the_loai/`. Kết quả nhóm đã chạy có sẵn ở
[`results/v5_the_loai/summary.md`](results/v5_the_loai/summary.md).

Tóm tắt (trung bình 3 seed, trung bình 3 thể loại):

| cách làm | char n-gram SVM | mạng lai (concat) |
|---|---|---|
| mô hình riêng từng thể loại — macro-F1 | 0,778 | 0,756 |
| mô hình chung + lọc theo thể loại — macro-F1 | 0,784 | 0,767 |
| mô hình chung + lọc — accuracy | 0,855 | 0,833 |
| mô hình chung không lọc — accuracy | 0,841 | 0,822 |
| chọn sai thể loại — accuracy | 0,144 | 0,143 |

Chênh lệch giữa SVM và mạng lai, giữa mô hình riêng và mô hình chung nhỏ hơn độ lệch chuẩn giữa các seed (0,02–0,05).
Mô hình tụt mạnh khi đổi thể loại (ví dụ học văn nghị luận của Ngô Tất Tố, đoán tiểu thuyết của ông: đúng ~5%).

## Dựng lại kho dữ liệu (không bắt buộc)

`data/v5_corpus.csv` đã có sẵn. Văn bản gốc (1.699 file) được nén trong `data/v5_genre/raw.zip`; muốn dựng lại kho thì
giải nén vào `data/v5_genre/` (tạo thư mục `data/v5_genre/raw/`) rồi chạy:
```
python -c "import shutil; shutil.unpack_archive('data/v5_genre/raw.zip', 'data/v5_genre')"
python main.py build-genre-corpus          # -> data/v5_corpus.csv (+ .stats.json), giống hệt bản có sẵn
```
Muốn tải lại từ Wikisource (cần Internet, `beautifulsoup4`):
```
python scripts/fetch_v5.py survey          # liệt kê tác phẩm + đề xuất thể loại -> data/v5_works.csv
#   duyệt cột genre trong data/v5_works.csv (tho | tu_su | ky | nghi_luan | bo | để trống)
python scripts/fetch_v5.py fetch           # tải -> data/v5_genre/raw/<thể loại>/<tác giả>/<tác phẩm>/*.txt
python main.py build-genre-corpus
```
- Thể loại định nghĩa trong `data/genres.json`, nhãn gán cho từng tác phẩm theo hình thức văn bản. Tác giả ứng viên:
  `data/v5_sources.json`. Quyết định duyệt nhãn và lý do: `scripts/apply_v5_review.py`; báo cáo dữ liệu:
  `data/v5_bao_cao_du_lieu.md`.
- Chỉ nhận trang có header ghi đúng tác giả (loại bản dịch, bài của người khác); đoạn có hơn 3% âm tiết sai cấu trúc tiếng
  Việt bị loại để lỗi OCR không thành "vân tay" giả. Thể loại Ký bị loại vì chỉ 1 tác giả đủ dữ liệu.
- Chia train/val/test theo bài (`doc_id`), phân tầng theo cặp thể loại|tác giả: các đoạn của cùng một bài luôn nằm chung một tập.

## Các lệnh khác

| lệnh | việc |
|---|---|
| `baseline` | chạy 3 baseline |
| `train` | huấn luyện mạng lai (`--fusion concat|gated|lm_only|style_only`, `--char-ngram`, `--oversample`, `--class-bias`, `--class-weight`) |
| `ablation`, `experiment` | so sánh các chế độ hợp nhất; kịch bản `random` và `cross_topic` (tác phẩm chưa thấy), đoạn ngắn |
| `evaluate`, `predict`, `openset` | đánh giá lại checkpoint, dự đoán (`--threshold auto` dùng ngưỡng từ chối học trên val), đo từ chối tác giả ngoài kho |
| `demo-data`, `tiny-backbone` | dữ liệu nhân tạo + BERT nhỏ để chạy thử đường ống không cần Internet |

`python main.py <lệnh> -h` để xem tham số. Đường dẫn dữ liệu/kết quả tính từ thư mục dự án; đặt biến môi trường
`AA_HOME` nếu muốn để ở nơi khác.

## Cấu trúc

```
main.py                   điểm vào dòng lệnh
app.py                    giao diện Streamlit
config/settings.py        đường dẫn và siêu tham số
src/text_utils.py         chuẩn hóa (NFC, quy ước chép văn bản), tách từ, kiểm tra âm tiết tiếng Việt
src/data_loader.py        đọc dữ liệu; chia train/val/test theo bài
src/corpus_builder.py     dựng kho đoạn văn (văn xuôi 50–200 âm tiết, thơ 40–160 âm tiết theo dòng)
src/feature_extractor.py  đặc trưng phong cách
src/models.py             HybridAuthorModel (concat, gated, lm_only, style_only)
src/train.py              huấn luyện, dừng sớm, hiệu chỉnh xác suất, lưu checkpoint tự chứa
src/predict.py            nạp checkpoint, dự đoán (cắt văn bản dài, lọc tác giả ứng viên)
src/baselines.py          Burrows' Delta, char n-gram + SVM, stylometry + LogReg
src/genre_experiment.py   thí nghiệm theo thể loại
src/experiment.py         baseline + ablation, chéo chủ đề, đoạn ngắn
src/calibration.py, src/ngram_features.py, src/openset.py, src/explain.py, src/evaluate.py, src/dataset.py
scripts/fetch_v5.py       tải kho theo thể loại từ vi.wikisource
data/                     kho dữ liệu (v5_corpus.csv), định nghĩa thể loại, nhãn tác phẩm, văn bản gốc (v5_genre/raw.zip)
results/v5_the_loai/      kết quả thí nghiệm của nhóm
tests/                    python -m unittest discover -s tests
```

## Giấy phép dữ liệu

Văn bản trong `data/` là tác phẩm thuộc phạm vi công cộng chép từ vi.wikisource.org; URL nguồn từng trang ghi trong
`raw/manifest.csv` bên trong `data/v5_genre/raw.zip` (tóm tắt: `data/v5_manifest_summary.csv`). Kết quả định danh chỉ mang tính tham khảo, không dùng làm bằng chứng giám định.
