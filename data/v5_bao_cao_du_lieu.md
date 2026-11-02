# Báo cáo dữ liệu kho v5 (theo thể loại)

Ngày dựng: 01/11/2026. Kho: `data/v5_corpus.csv` (+ `data/v5_corpus.stats.json`). Nguồn: vi.wikisource.org (trang thường và trang “Trang:” đã soát), chỉ tác giả mất trước 1976. Các bước: `fetch_v5.py survey` → duyệt nhãn → `fetch_v5.py fetch` → `main.py build-genre-corpus` (ngưỡng mặc định trong `data/genres.json`, không hạ ngưỡng).

## 1. Tổng quan

- **4860 đoạn, 3 thể loại, 28 tác giả khác nhau.** Thể loại **Ký bị loại** (xem mục 3).
- Tác giả có mặt ở nhiều thể loại: Hồ Biểu Chánh (Thơ, Tự sự); Hồ Chí Minh (Nghị luận, Thơ); Ngô Tất Tố (Nghị luận, Tự sự); Phan Bội Châu (Nghị luận, Thơ); Tản Đà (Thơ, Tự sự).
- Đơn vị trùng lặp bị bỏ khi dựng kho: 426.

| Thể loại | Số tác giả | Số đoạn |
|---|---|---|
| Nghị luận (`nghi_luan`) | 13 | 2179 |
| Thơ (`tho`) | 12 | 1259 |
| Tự sự (`tu_su`) | 8 | 1422 |

### Nghị luận — 13 tác giả, 2179 đoạn

| Tác giả | Số đoạn | Số tác phẩm | Số bài | Âm tiết TB/đoạn | Tỷ lệ âm tiết sai TB |
|---|---|---|---|---|---|
| Đào Trinh Nhất | 200 | 2 | 27 | 153.2 | 0.006 |
| Dương Bá Trạc | 200 | 3 | 31 | 162.3 | 0.002 |
| Hồ Chí Minh | 200 | 7 | 19 | 160.4 | 0.004 |
| Huỳnh Thúc Kháng | 125 | 12 | 14 | 146.7 | 0.003 |
| Ngô Tất Tố | 102 | 5 | 10 | 154.8 | 0.002 |
| Nguyễn Trọng Thuật | 98 | 2 | 8 | 149.7 | 0.002 |
| Nguyễn Văn Vĩnh | 122 | 19 | 19 | 149.7 | 0.002 |
| Phạm Quỳnh | 200 | 11 | 27 | 156.7 | 0.004 |
| Phan Bội Châu | 200 | 6 | 48 | 146.8 | 0.003 |
| Phan Châu Trinh | 132 | 2 | 7 | 150.0 | 0.001 |
| Phan Kế Bính | 200 | 5 | 46 | 153.2 | 0.003 |
| Phan Khôi | 200 | 41 | 43 | 155.6 | 0.003 |
| Trần Trọng Kim | 200 | 8 | 59 | 151.3 | 0.002 |

### Thơ — 12 tác giả, 1259 đoạn

| Tác giả | Số đoạn | Số tác phẩm | Số bài | Âm tiết TB/đoạn | Tỷ lệ âm tiết sai TB |
|---|---|---|---|---|---|
| Bích Khê | 78 | 55 | 4 | 151.8 | 0.001 |
| Đông Hồ | 54 | 33 | 3 | 147.7 | 0.000 |
| Hàn Mặc Tử | 153 | 78 | 8 | 150.5 | 0.000 |
| Hồ Biểu Chánh | 100 | 2 | 5 | 152.5 | 0.001 |
| Hồ Chí Minh | 59 | 44 | 4 | 149.7 | 0.006 |
| Nguyễn Bính | 196 | 71 | 10 | 153.2 | 0.000 |
| Nguyễn Công Trứ | 44 | 38 | 3 | 148.4 | 0.002 |
| Nguyễn Đình Chiểu | 197 | 12 | 10 | 153.9 | 0.001 |
| Nguyễn Du | 150 | 3 | 8 | 153.9 | 0.000 |
| Phan Bội Châu | 82 | 57 | 5 | 148.4 | 0.004 |
| Tản Đà | 97 | 18 | 6 | 144.7 | 0.001 |
| Trần Tế Xương | 49 | 46 | 3 | 143.8 | 0.000 |

### Tự sự — 8 tác giả, 1422 đoạn

| Tác giả | Số đoạn | Số tác phẩm | Số bài | Âm tiết TB/đoạn | Tỷ lệ âm tiết sai TB |
|---|---|---|---|---|---|
| Hồ Biểu Chánh | 200 | 2 | 28 | 156.1 | 0.001 |
| Lê Văn Trương | 200 | 3 | 27 | 156.9 | 0.004 |
| Nam Cao | 200 | 15 | 26 | 159.7 | 0.000 |
| Ngô Tất Tố | 200 | 3 | 31 | 157.2 | 0.000 |
| Nguyễn Chánh Sắt | 200 | 2 | 27 | 157.6 | 0.002 |
| Tản Đà | 104 | 2 | 13 | 156.2 | 0.005 |
| Thạch Lam | 200 | 26 | 29 | 154.4 | 0.001 |
| Vũ Trọng Phụng | 118 | 3 | 8 | 155.1 | 0.003 |

**Lưu ý về độ đa dạng tác phẩm (ảnh hưởng tới đánh giá chéo tác phẩm):** ở Thơ, Nguyễn Du gần như chỉ có *Truyện Kiều* (148/150 đoạn), Hồ Biểu Chánh chủ yếu *U tình lục* (86/100), Nguyễn Đình Chiểu chủ yếu *Dương Từ Hà Mậu* (161/197); ở Tự sự, Hồ Biểu Chánh, Nguyễn Chánh Sắt và Tản Đà mỗi người chỉ 2 tác phẩm. Với các tác giả này, đoạn test thường cùng tác phẩm với đoạn train.

## 2. Cặp (thể loại, tác giả) bị loại và lý do

Ngưỡng (`genres.json`): Thơ ≥ 40 đoạn, Tự sự ≥ 60, Ký ≥ 40, Nghị luận ≥ 60; mọi thể loại ≥ 2 tác phẩm; mỗi thể loại ≥ 5 tác giả.

| Thể loại | Tác giả | Lý do |
|---|---|---|
| Ký | Nguyễn Văn Vĩnh | 8 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Ký | Tản Đà | 86 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Ký | Thạch Lam | 114 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Ký | Trần Trọng Kim | 288 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Ký | Trương Vĩnh Ký | 65 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Ký | Vũ Trọng Phụng | 30 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Nghị luận | Bùi Thế Mỹ | 16 đoạn, 3 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Nghị luận | Đạm Phương | 5 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Nghị luận | Diệp Văn Kỳ | 53 đoạn, 2 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Nghị luận | Hải Triều | 8 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Nghị luận | Hoàng Tích Chu | 29 đoạn, 5 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Nghị luận | Ngô Đức Kế | 20 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Nghị luận | Tản Đà | 55 đoạn, 4 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Nghị luận | Vũ Trọng Phụng | 15 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Thơ | Bà Huyện Thanh Quan | 3 đoạn, 3 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Bùi Kỷ | 3 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Bùi Thế Mỹ | 1 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Đạm Phương | 1 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Dương Bá Trạc | 4 đoạn, 4 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Hồ Xuân Hương | 17 đoạn, 17 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Ngô Đức Kế | 2 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Nguyễn An Ninh | 1 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Nguyễn Khuyến | 32 đoạn, 32 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Nguyễn Nhược Pháp | 18 đoạn, 7 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Nhượng Tống | 3 đoạn, 3 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Phan Châu Trinh | 3 đoạn, 3 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Phan Kế Bính | 2 đoạn, 2 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Phan Khôi | 7 đoạn, 4 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Thâm Tâm | 23 đoạn, 16 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Thơ | Trương Minh Ký | 69 đoạn, 1 tác phẩm (cần ≥40 đoạn, ≥2 tác phẩm) |
| Tự sự | Hoàng Ngọc Phách | 34 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Tự sự | Khái Hưng | 389 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Tự sự | Lan Khai | 96 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Tự sự | Nguyễn Bá Học | 29 đoạn, 2 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Tự sự | Nguyễn Trọng Quản | 59 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Tự sự | Phạm Duy Tốn | 34 đoạn, 4 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Tự sự | Phan Kế Bính | 277 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |
| Tự sự | Phan Khôi | 8 đoạn, 1 tác phẩm (cần ≥60 đoạn, ≥2 tác phẩm) |

Nhận xét: nhiều cặp bị loại vì **chỉ 1 tác phẩm** dù đủ số đoạn: Khái Hưng (Tự sự, 389 đoạn, chỉ *Nửa chừng xuân*), Phan Kế Bính (Tự sự, 277 đoạn, chỉ *Hưng Đạo vương*), Lan Khai (Tự sự, 96 đoạn), Trần Trọng Kim (Ký, 288 đoạn, chỉ *Một cơn gió bụi*), Thạch Lam (Ký, 114 đoạn), Tản Đà (Ký, 86 đoạn), Trương Vĩnh Ký (Ký, 65 đoạn), Trương Minh Ký (Thơ, 69 đoạn). Các nhà thơ cổ (Nguyễn Khuyến 32, Hồ Xuân Hương 17, Thâm Tâm 23, Nguyễn Nhược Pháp 18 đoạn) bị loại vì bài thơ ngắn và phần lớn thơ chữ Hán đã bị bỏ (mục 4).

## 3. Thể loại bị loại

- **Ký (`ky`)**: chỉ 1 tác giả (cần ≥5). Chỉ Ngô Tất Tố đạt cả hai ngưỡng (2 tác phẩm: *Việc làng*, *Làm no hay cái ăn…*). Sáu tác giả khác có văn bản ký nhưng mỗi người chỉ 1 tác phẩm trên Wikisource.

## 4. Đoạn bị lọc vì âm tiết sai cấu trúc

Tổng **578 đoạn** (ngưỡng > 3% âm tiết sai cấu trúc): Nghị luận 372, Ký 73, Tự sự 68, Thơ 65.
Nhiều nhất: nghi_luan/tran_trong_kim 86, nghi_luan/pham_quynh 82, nghi_luan/dao_trinh_nhat 57, ky/tran_trong_kim 41, nghi_luan/nguyen_trong_thuat 37, tu_su/tan_da 29, tho/truong_minh_ky 27, nghi_luan/phan_khoi 19.

Lưu ý: bộ lọc này không chỉ bắt lỗi OCR mà còn bỏ các đoạn có trích dẫn chữ Hán, tiếng Pháp hoặc chữ số dày (các bài nghị luận của Trần Trọng Kim, Phạm Quỳnh, Đào Trinh Nhất hay trích nguyên văn). Bộ lọc **không** bắt được phiên âm Hán Việt (mỗi âm tiết phiên âm vẫn đúng cấu trúc tiếng Việt) — phần này được xử lý khi duyệt nhãn (mục 5).

## 5. Đổi nhãn so với đề xuất tự động

Tổng **264 tác phẩm** đổi nhãn (đầy đủ: `data/v5_genre/review_changes.csv`; đề xuất gốc lưu ở `data/v5_genre/v5_works_de_xuat_goc.csv`; quy tắc và danh sách sửa tay: `scripts/apply_v5_review.py`).

### 5.1. Thơ/văn chữ Hán — 170 tác phẩm (quy tắc tự động, rồi soát lại)

Trang có ≥ 10 chữ Hán hoặc có mục *Phiên âm / Dịch nghĩa / Dịch thơ / Nguyên văn chữ Hán* mà đề xuất là `tho` hoặc chưa rõ → `bo`. Lý do: `genres.json` loại thơ chữ Hán và bản dịch của người khác; header Wikisource vẫn ghi tên tác giả gốc nên bước khảo sát không phát hiện. Theo tác giả: Nguyễn Du 113, Nguyễn Khuyến 46, Phan Châu Trinh 3, Hồ Xuân Hương 2, Nguyễn Đình Chiểu 2, Nguyễn Công Trứ 1, Bà Huyện Thanh Quan 1, Hồ Chí Minh 1, Phan Bội Châu 1.
Ngoại lệ (giữ `tho`, vì là thơ Nôm của chính tác giả, chữ Hán chỉ là câu chèn/chú thích): hát nói của Nguyễn Công Trứ (*Chí nam nhi, Kẻ sĩ, Chữ nhàn, Cầm kỳ thi tửu/Bài 2, Chí làm trai, Hàn nho phong vị phú, Đi thi tự vịnh*), Đông Hồ (*Cảnh trăng trên Đông Hồ*), Nguyễn Đình Chiểu (*Trung thần, nghĩa sĩ*), Tản Đà (*Lên sáu, Đài gương truyện*).

### 5.2. Các thay đổi khác — 94 tác phẩm

| Tác giả | Tác phẩm | Đề xuất → Nhãn | Lý do |
|---|---|---|---|
| Bích Khê | Tinh huyết | ? → tho | tập thơ |
| Bùi Thế Mỹ | Quyền lợi và tương tế | ? → nghi_luan | bài luận |
| Đạm Phương | Nữ quốc dân tu tri/Phụ trương-2 | ? → nghi_luan | bài giảng, luận |
| Đào Trinh Nhất | Vương Dương Minh/Phụ lục-5 | ? → nghi_luan | khảo cứu |
| Diệp Văn Kỳ | Nước non nhà | ? → nghi_luan | tập bài luận |
| Đông Hồ | Bội lan hành | ? → tho | tập thơ |
| Đông Hồ | Cô gái xuân | ? → tho | tập thơ |
| Dương Bá Trạc | Chữ nho học lấy | tho → (trống) | không đủ căn cứ |
| Dương Bá Trạc | Gia lễ giản yếu | tho → nghi_luan | sách biên khảo lễ nghi; đề xuất tho sai do dòng tiêu đề |
| Dương Bá Trạc | Nét mực tình | tho → bo | hợp tuyển nhiều tác giả |
| Dương Bá Trạc | Tiếng gọi đàn | ? → nghi_luan | sách chính luận |
| Hồ Biểu Chánh | Tiền bạc, bạc tiền | tu_su → bo | trùng tác phẩm với bản khác trong kho (giữ một bản để tránh đoạn gần giống nhau lọt cả train lẫn test): giữ bản chụp Mục lục:Tien bac bac tien.pdf (151/156 trang đã soát) |
| Hồ Chí Minh | Cảnh khuya | ? → tho | bài thơ quốc ngữ |
| Hồ Chí Minh | Di chúc của Chủ tịch Hồ Chí Minh | ? → nghi_luan | di chúc (genres.json: thuộc nghi_luan) |
| Hồ Chí Minh | Mừng ngày Chúa giáng sinh | ? → nghi_luan | thư, lời chúc gửi đồng bào |
| Hồ Chí Minh | Ngục trung nhật ký | ky → bo | thơ chữ Hán, bản quốc ngữ là bản dịch của người khác |
| Hồ Chí Minh | Nội dung chủ nghĩa anh hùng cách mạng của chúng ta | nghi_luan → (trống) | trang nhiều chữ Hán (128) mà ít văn xuôi (90 từ), có thể là bản song ngữ; không chắc |
| Hồ Chí Minh | Sắc luật cấm chỉ mọi hành động đầu cơ về kinh tế nước Việt Nam Dân chủ Cộng hòa | ? → bo | văn bản pháp luật (genres.json: không thuộc nghi_luan) |
| Hồ Chí Minh | Sắc luật quy định những trường hợp phạm pháp quả tang, những trường hợp khẩn cấp, và những trường hợp khám người phạm pháp quả tang nước Việt Nam Dân chủ Cộng hòa | ? → bo | văn bản pháp luật (genres.json: không thuộc nghi_luan) |
| Hồ Chí Minh | Sắc luật về bầu cử Hội đồng nhân dân và Ủy ban hành chính các cấp nước Việt Nam Dân chủ Cộng hòa | ? → bo | văn bản pháp luật (genres.json: không thuộc nghi_luan) |
| Hồ Chí Minh | Sắc luật về quyền tự do xuất bản nước Việt Nam Dân chủ Cộng hòa | ? → bo | văn bản pháp luật (genres.json: không thuộc nghi_luan) |
| Hồ Chí Minh | Sửa đổi lối làm việc | ? → nghi_luan | sách chính luận |
| Hồ Chí Minh | Thư Chủ tịch Hồ Chí Minh gửi các học sinh nhân ngày khai giảng đầu tiên ở nước Việt Nam Dân chủ Cộng hòa năm 1945 | ? → nghi_luan | thư |
| Hoàng Ngọc Phách | Tố Tâm | ? → tu_su | tiểu thuyết |
| Hoàng Tích Chu | Cái án tú Khôi - cử Kỳ | ? → nghi_luan | bài báo châm biếm, bình luận |
| Hoàng Tích Chu | Cần có bọn học giả ấy của ông Phan Khôi | ? → nghi_luan | bút chiến |
| Hoàng Tích Chu | Nghe quẻ bói của Thông Reo mà Văn Tôi mừng | ? → nghi_luan | bút chiến |
| Huỳnh Thúc Kháng | Việt Nam ta cũng có người hay thi | ? → nghi_luan | bài luận (có trích thơ) |
| Huỳnh Tịnh Của | Chuyện giải buồn | ? → bo | 'dịch rút trong các sách hay' — truyện dịch, phỏng dịch |
| Huỳnh Tịnh Của | Chuyện giải buồn cuốn sau | ? → bo | truyện dịch, phỏng dịch |
| Huỳnh Tịnh Của | Câu hát góp | tho → bo | câu hát dân gian sưu tầm |
| Huỳnh Tịnh Của | Tục ngữ, cổ ngữ, gia ngôn | tho → bo | tục ngữ dân gian sưu tầm |
| Huỳnh Tịnh Của | Đại Nam Quấc âm tự vị | ? → bo | từ điển |
| Khái Hưng | Anh phải sống | tu_su → bo | Khái Hưng và Nhất Linh viết chung |
| Khái Hưng | Anh phải sống/Tình tuyệt vọng | tu_su → bo | thuộc tập 'Anh phải sống' viết chung |
| Ngô Tất Tố | Báo Trung lập đã nhổ đi một cái đinh ở trước mắt ông Phan Khôi | ? → nghi_luan | bút chiến |
| Ngô Tất Tố | Làm no hay cái ăn ... | ? → ky | phóng sự tả cảnh lụt, người thật việc thật |
| Ngô Tất Tố | Phê bình Nho giáo Trần Trọng Kim | tho → nghi_luan | phê bình; đề xuất tho sai do tiêu đề nhiều dòng bị tính là dòng thơ |
| Nguyễn Bá Học | Chuyện cô Chiêu Nhì | ? → tu_su | truyện ngắn |
| Nguyễn Bính | Lỡ bước sang ngang | ? → tho | tập thơ |
| Nguyễn Bính | Mây tần | ? → tho | tập thơ |
| Nguyễn Bính | Mười hai bến nước | ? → tho | tập thơ |
| Nguyễn Bính | Người con gái lầu hoa | ? → tho | tập thơ |
| Nguyễn Bính | Tâm hồn tôi | ? → tho | tập thơ |
| Nguyễn Chánh Sắt | Gái trả thù cha | tho → tu_su | tiểu thuyết; đề xuất tho sai do phần thơ chèn và dòng tiêu đề |
| Nguyễn Đình Chiểu | Lục Vân Tiên | ? → tho | truyện thơ Nôm của tác giả |
| Nguyễn Đình Chiểu | Viếng cụ Phan Thanh Giản | tho → (trống) | có cặp câu chữ Hán (phiên âm) lẫn thơ Nôm; không chắc |
| Nguyễn Du | Hoàng Châu trúc lâu | tho → bo | thơ chữ Hán: 59% dòng là phiên âm Hán Việt |
| Nguyễn Du | Hành lạc từ | ? → bo | thơ chữ Hán |
| Nguyễn Du | Mạnh Tử từ cổ liễu | tho → bo | thơ chữ Hán: 36% dòng là phiên âm Hán Việt, còn lại là bản dịch |
| Nguyễn Du | Thu nhật ký hứng | ky → bo | thơ chữ Hán (đề xuất 'ky' sai vì tiêu đề có chữ 'nhật ký') |
| Nguyễn Nhược Pháp | Ngày xưa | ? → tho | tập thơ |
| Nguyễn Văn Vĩnh | Chết vì gạo | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Chữ Quốc-ngữ | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Chữ nho nên để hay là nên bỏ | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Con xâu đổ đầu nồi canh | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Duy tân (Nguyễn Văn Vĩnh) | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Hội dịch sách | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Lính-tuần, lính-lệ | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Ma to dỗ nhớn | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Nhời đàn bà (Propos de femme) - Lắm vợ | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Phiên mổ bò | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Phán, ký | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Thiếu gạo ăn thừa giấy đốt | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Thói tệ | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Tiếng An-nam | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Truyện ăn mày | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Tật huyền hồ sáo hủ | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nguyễn Văn Vĩnh | Đốt pháo | ky → nghi_luan | luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận) |
| Nhất Linh | Anh phải sống | tu_su → bo | Khái Hưng và Nhất Linh viết chung |
| Phạm Quỳnh | Mục lục:Văn minh luận (1927) by Phạm Quỳnh.pdf | nghi_luan → bo | trùng tác phẩm với bản khác trong kho (giữ một bản để tránh đoạn gần giống nhau lọt cả train lẫn test): giữ trang thường 'Văn minh luận' (như kho v3, v4) |
| Phan Bội Châu | Cao đẳng quốc dân | ? → nghi_luan | sách chính luận |
| Phan Bội Châu | Lời hỏi? Các anh em thanh niên | ? → nghi_luan | bài luận gửi thanh niên |
| Phan Bội Châu | Pháp-Việt đề huề chính kiến thư | ? → bo | nguyên văn chữ Hán kèm bản dịch |
| Phan Bội Châu | Thuốc hoàn hồn | ? → nghi_luan | sách chính luận |
| Phan Bội Châu | Trông mưa | tho → (trống) | 29% dòng là phiên âm Hán Việt; không chắc thơ Hán hay thơ Nôm chèn câu Hán |
| Phan Bội Châu | Tạ ơn người bạn gái cho cái bao đầu | ? → tho | bài thơ |
| Phan Bội Châu | Tờ thông cáo toàn quốc của Phan Bội Châu | ? → bo | nguyên văn chữ Hán kèm bản dịch |
| Phan Châu Trinh | Các kiến nghị vắn tắt của Phan Châu Trinh do Roux dịch chuyển cho bộ trưởng thuộc địa và Sarraut | ? → bo | bản dịch (tiêu đề ghi 'do Roux dịch') |
| Phan Châu Trinh | Giai nhân kỳ ngộ chi ca | tho → bo | diễn ca phỏng dịch tiểu thuyết Nhật 'Kajin no kigū' |
| Phan Kế Bính | Việt Hán văn khảo/II.V7-2 | tho → bo | phiên âm thơ chữ Hán (bài mở đầu Tam quốc) |
| Phan Kế Bính | Đêm trăng chơi Hồ Tây | nghi_luan → (trống) | có thể là bản dịch bài chữ Hán của chính tác giả; không chắc |
| Tản Đà | Giấc mộng con | ? → tu_su | truyện tưởng tượng (du hành trong mộng), văn xuôi chiếm phần lớn; thơ chèn bị tách bỏ |
| Tản Đà | Nhàn tưởng | ? → nghi_luan | tập tạp văn, tiểu luận |
| Tản Đà | Thơ Tản Đà | ? → tho | tập thơ (khi tải chỉ lấy khối thơ) |
| Tản Đà | Đánh bạc | ? → nghi_luan | bài luận ngắn |
| Trần Trọng Kim | Phật giáo | tho → nghi_luan | ba bài diễn thuyết; đề xuất tho sai do dòng tiêu đề |
| Trương Minh Ký | Ca từ diễn nghĩa | tho → bo | diễn nghĩa bài chữ Hán của Tư Mã Quang |
| Trương Minh Ký | Tuồng Joseph | tho → bo | tuồng (kịch) |
| Trương Minh Ký | Ấu học khải mông | tho → bo | diễn nghĩa sách vỡ lòng chữ Hán |
| Trương Vĩnh Ký | Gia Định báo | ? → bo | tờ báo, nhiều người viết |
| Vũ Trọng Phụng | Làm đĩ | tu_su → bo | trùng tác phẩm với bản khác trong kho (giữ một bản để tránh đoạn gần giống nhau lọt cả train lẫn test): giữ bản chụp Mục lục:Lam di 1939.pdf (51/59 trang đã soát, đủ hơn) |
| Vũ Trọng Phụng | Mục lục:Com Thay Com Co va Luc Si 1937.pdf | tu_su → bo | phóng sự (không phải tu_su như gợi ý); chỉ 1/10 trang đã soát và Lục sì đã có ở trang thường |
| Vũ Trọng Phụng | Để đáp lời báo Ngày nay : Dâm hay là không dâm ? | tu_su → nghi_luan | bút chiến trả lời báo Ngày nay |

## 6. Tác giả không tìm thấy

- **Hoàng Đạo** (Nguyễn Tường Long, mất 1948): Wikisource không có trang Tác gia; đã tìm cả tên thật và tiền tố “Tác gia:Hoàng…”, “Tác gia:Nguyễn Tường…” (chỉ có Nguyễn Tường Tam = trang chuyển hướng tới Nhất Linh).
- **Trần Huy Liệu** (mất 1969): không có trang Tác gia; kết quả tìm kiếm toàn văn chỉ là nghị quyết của Quốc hội/văn bản nhà nước (không phải tác phẩm ông viết) → không thêm.
- Ghi chú: **Nhất Linh** có trang nhưng chỉ có *Anh phải sống* — tập truyện viết chung với Khái Hưng → `bo` (cả ở Khái Hưng).

## 7. Lỗi gặp phải và cách xử lý

1. **Thơ chữ Hán và bản dịch lọt vào nhãn `tho`.** Header ghi tác giả gốc; trang gồm nguyên văn, phiên âm và bản dịch của người khác. Viết `scripts/review_v5_features.py` (đếm chữ Hán và dấu hiệu dịch thuật từ bộ đệm survey) và áp quy tắc ở mục 5.1.
2. **Phiên âm Hán Việt không có chữ Hán trên trang (phát hiện SAU khi dựng kho lần 1).** Đoạn đầu kho thơ của Nguyễn Du là phiên âm *Hoàng Châu trúc lâu*. Đo theo từng dòng: dòng không có hư từ thuần Việt và có ≥ 2 từ Hán Việt điển hình. Bỏ *Hoàng Châu trúc lâu*, *Mạnh Tử từ cổ liễu* (`bo`); để trống *Trông mưa* (Phan Bội Châu), *Viếng cụ Phan Thanh Giản* (Nguyễn Đình Chiểu) vì không chắc. Hát nói Nguyễn Công Trứ, thơ Đông Hồ giữ nguyên.
3. **Số thứ tự dòng thơ của Wikisource lẫn vào văn bản.** *Truyện Kiều* có 650 số thứ tự (`<span title="Số thứ tự">`, 56 trang trong bộ đệm dùng bản mẫu này) → chữ số thành tín hiệu giả. **Sửa code** `fetch_v5.py` (DROP_SELECTORS bỏ `.OptionText`, `[title="Số thứ tự"]`), thêm test `LineNumberTest`, tải lại toàn bộ từ bộ đệm. Thư mục tải trước khi sửa từng giữ ở `data/v5_genre/raw_truoc_sua_so_dong/` (đã xóa ngày 02/11/2026 khi dọn phiên bản cũ). Sau sửa còn 103/1.259 đoạn thơ có chữ số — là nội dung thật (ngày tháng trong thơ Hồ Chí Minh, số câu do chính sách *Khối tình con* đánh).
4. **Đổi nhãn sau khi đã tải (`Thu nhật ký hứng` bị gán `ky` vì tiêu đề có “nhật ký”).** Quy tắc chữ Hán ban đầu chỉ áp cho đề xuất `tho`/`?`. File đã tải theo nhãn cũ vẫn nằm trong raw. Viết `scripts/sync_v5_manifest.py` (bỏ dòng manifest và file không còn khớp nhãn; không đụng nguồn `local`) + test `SyncManifestTest`. Đã đồng bộ 5 file.
5. **Wikisource xếp loạt bài 1907–1914 của Nguyễn Văn Vĩnh vào “Phóng sự”.** Đọc phần mở đầu: hình thức là luận bàn (“Bài này là bài luận…”) → 17 bài `nghi_luan`, chỉ *Hội Kiếp-bạc* (kể chuyến đi) giữ `ky`.
6. **Đề xuất `tho` sai vì dòng tiêu đề nhiều dòng (thẻ `<br>`) bị tính là dòng thơ.** Sửa tay: *Phê bình Nho giáo Trần Trọng Kim*, *Phật giáo* (Trần Trọng Kim), *Gia lễ giản yếu*, *Gái trả thù cha*. Chưa sửa code đề xuất (chỉ ảnh hưởng gợi ý, người duyệt quyết định).
7. **Cùng tác phẩm có cả trang thường và bản chụp (*Làm đĩ*, *Tiền bạc, bạc tiền*, *Văn minh luận*).** Giữ một bản để đoạn gần giống nhau không lọt cả train lẫn test.
8. **Văn bản không phải văn chương của tác giả trên trang tác gia Hồ Chí Minh.** Nghị định 17/2023, Luật 2025, điếu văn của Ban Chấp hành… đã bị bước survey loại nhờ kiểm tra header; 4 *Sắc luật* (header ghi ông) được gán `bo` (văn bản pháp luật).
9. **HTTP 429 (quá nhiều yêu cầu) khi tra cứu tay ngoài script.** Tăng giãn cách (3 giây) và thử lại có lùi thời gian.
10. **Môi trường.** beautifulsoup4 4.15.0 đã có sẵn; không có lỗi do Windows; bộ test 51/51 đạt sau mọi thay đổi.

## 8. Đề xuất bổ sung cho thể loại dưới 5 tác giả (KHÔNG tự hạ ngưỡng)

Chỉ **Ký** dưới ngưỡng (1/5 tác giả). Cách bổ sung hợp lệ, xếp theo khả năng:

1. **Thêm tác phẩm ký thứ hai cho các tác giả đã có 1 tác phẩm đủ số đoạn** (Thạch Lam, Trần Trọng Kim, Tản Đà, Trương Vĩnh Ký, Vũ Trọng Phụng) — chỉ cần 1 tác phẩm nữa mỗi người là đạt; 4 trong 5 người là đủ 5 tác giả cùng Ngô Tất Tố.
   - **Vũ Trọng Phụng:** *Cơm thầy cơm cô* (phóng sự) có bản chụp `Mục lục:Com Thay Com Co va Luc Si 1937.pdf` trên Wikisource nhưng mới 1/10 trang được soát. Soát thêm các trang này trên Wikisource (nguồn `ws_trang`, chất lượng ≥ 3) là đủ.
   - **Phạm Quỳnh:** du ký *Mười ngày ở Huế*, *Một tháng ở Nam Kỳ*, *Pháp du hành trình nhật ký* (đăng Nam Phong 1918–1922, phạm vi công cộng). Wikisource hiện chỉ có trích đoạn trong tuyển tập *Quốc văn trích diễm*; cần chép (nguồn `local`, ghi rõ số Nam Phong và giấy phép) hoặc soát trên Wikisource. Phạm Quỳnh sẽ thành tác giả Ký mới.
   - **Tản Đà:** *Giấc mộng lớn* đã có; cân nhắc lại *Giấc mộng con* (đang xếp `tu_su` vì là chuyến du hành trong mộng — nhóm quyết định; nếu xếp `ky` thì Tản Đà đạt 2 tác phẩm).
   - **Thạch Lam, Trần Trọng Kim, Trương Vĩnh Ký:** cần tìm tác phẩm ký thứ hai có bản chép sạch (vd. các bài tùy bút khác của Thạch Lam trong báo *Ngày nay*); hiện chưa có trên Wikisource.
2. **Tác giả ký mới thuộc phạm vi công cộng:** Nguyễn Văn Vĩnh (thêm bài du ký/tường thuật như *Hội Kiếp-bạc*), Phạm Quỳnh (như trên). Không dùng Vũ Bằng (mất 1984), Nguyễn Tuân (1987), Tam Lang (1986) — còn thời hạn bảo hộ.
3. Nguồn hợp lệ: soát trang “Trang:” trên vi.wikisource (đóng góp được cho cộng đồng, bộ tải lấy tự động); văn bản tự chép từ bản in phạm vi công cộng đặt vào `data/v5_genre/raw/<thể loại>/<tác giả>/<tác phẩm>/*.txt` kèm dòng nguồn/giấy phép trong manifest (nguồn `local`). Không dùng lớp chữ OCR chưa soát của PDF.

Các thể loại còn lại đều trên ngưỡng 5 tác giả (Thơ 12, Nghị luận 13, Tự sự 8).

## 9. File liên quan

- `data/v5_works.csv` — nhãn đã duyệt (cột `genre`, `ghi_chu`); `data/v5_genre/v5_works_de_xuat_goc.csv` — bản đề xuất gốc.
- `data/v5_genre/review_changes.csv`, `review_features.csv` — thay đổi và đặc điểm từng trang.
- `scripts/review_v5_features.py`, `scripts/apply_v5_review.py`, `scripts/sync_v5_manifest.py` — công cụ duyệt.
- `data/v5_survey_log.txt`, `data/v5_fetch_log.txt`, `data/v5_build_log.txt` — log từng bước.
- `data/v5_manifest_summary.csv`, `data/v5_genre/raw/manifest.csv` — nguồn từng file.
