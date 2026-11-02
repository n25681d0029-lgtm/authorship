#!/usr/bin/env python3
"""Ghi kết quả duyệt nhãn thể loại vào data/v5_works.csv (cột genre, ghi_chu) — lần duyệt ngày 01/11/2026.

Quy tắc theo data/genres.json; căn cứ: tiêu đề, thể loại Wikisource, phần mở đầu nội dung (đọc từ bộ đệm survey) và
data/v5_genre/review_features.csv (scripts/review_v5_features.py). Mọi thay đổi so với cột de_xuat được ghi vào
data/v5_genre/review_changes.csv để đưa vào báo cáo. Chạy lại được nhiều lần (luôn tính từ de_xuat).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_v5 import V5, WORK_FIELDS, WORKS_CSV, read_csv, write_csv  # noqa: E402

HAN_NOTE = "thơ/văn chữ Hán: trang có nguyên văn chữ Hán, phiên âm, dịch nghĩa hoặc dịch thơ của người khác"
KEEP_HAN = "giữ: thơ Nôm/quốc ngữ của chính tác giả, chữ Hán chỉ là câu chèn hoặc chú thích"

# Thơ Nôm của chính tác giả có chèn câu chữ Hán (hát nói...) — KHÔNG áp quy tắc chữ Hán
HAN_WHITELIST = {
    ("nguyen_cong_tru", "Cầm kỳ thi tửu/Bài 2"), ("nguyen_cong_tru", "Chí làm trai"), ("nguyen_cong_tru", "Chí nam nhi"),
    ("nguyen_cong_tru", "Chữ nhàn"), ("nguyen_cong_tru", "Hàn nho phong vị phú"), ("nguyen_cong_tru", "Kẻ sĩ"),
    ("nguyen_cong_tru", "Đi thi tự vịnh"), ("dong_ho", "Cảnh trăng trên Đông Hồ"),
    ("nguyen_dinh_chieu", "Trung thần, nghĩa sĩ"), ("tan_da", "Lên sáu"), ("tan_da", "Đài gương truyện"),
}

NVV_ESSAY = "luận bàn/xã luận (Wikisource xếp 'Phóng sự' nhưng hình thức là lập luận)"
DUP = "trùng tác phẩm với bản khác trong kho (giữ một bản để tránh đoạn gần giống nhau lọt cả train lẫn test)"

# (tác giả, tác phẩm) -> (nhãn, ghi chú). Nhãn "" = để trống (không chắc, bỏ qua).
OVERRIDES = {
    # --- Vũ Trọng Phụng
    ("vu_trong_phung", "Để đáp lời báo Ngày nay : Dâm hay là không dâm ?"): ("nghi_luan", "bút chiến trả lời báo Ngày nay"),
    ("vu_trong_phung", "Làm đĩ"): ("bo", DUP + ": giữ bản chụp Mục lục:Lam di 1939.pdf (51/59 trang đã soát, đủ hơn)"),
    ("vu_trong_phung", "Mục lục:Com Thay Com Co va Luc Si 1937.pdf"):
        ("bo", "phóng sự (không phải tu_su như gợi ý); chỉ 1/10 trang đã soát và Lục sì đã có ở trang thường"),
    # --- Ngô Tất Tố
    ("ngo_tat_to", "Làm no hay cái ăn ..."): ("ky", "phóng sự tả cảnh lụt, người thật việc thật"),
    ("ngo_tat_to", "Báo Trung lập đã nhổ đi một cái đinh ở trước mắt ông Phan Khôi"): ("nghi_luan", "bút chiến"),
    ("ngo_tat_to", "Gia thế ông Vũ Trọng Phụng"): ("", "lưng chừng giữa ký (hồi ức) và nghị luận, không gán"),
    ("ngo_tat_to", "Những trận đổ máu hồi người Pháp mới sang ta đến ngày nay"): ("", "trang chính không có nội dung để kiểm tra"),
    ("ngo_tat_to", "Phê bình Nho giáo Trần Trọng Kim"): ("nghi_luan", "phê bình; đề xuất tho sai do tiêu đề nhiều dòng bị tính là dòng thơ"),
    # --- Hồ Biểu Chánh, Phạm Quỳnh: trùng bản chụp
    ("ho_bieu_chanh", "Tiền bạc, bạc tiền"): ("bo", DUP + ": giữ bản chụp Mục lục:Tien bac bac tien.pdf (151/156 trang đã soát)"),
    ("pham_quynh", "Mục lục:Văn minh luận (1927) by Phạm Quỳnh.pdf"): ("bo", DUP + ": giữ trang thường 'Văn minh luận' (như kho v3, v4)"),
    # --- Phan Kế Bính
    ("phan_ke_binh", "Nam Hải dị nhân liệt truyện"): ("", "truyện người lạ sưu tập, kể lại; không chắc là tự sự hư cấu của tác giả"),
    ("phan_ke_binh", "Đêm trăng chơi Hồ Tây"): ("", "có thể là bản dịch bài chữ Hán của chính tác giả; không chắc"),
    ("phan_ke_binh", "Việt Hán văn khảo/II.V7-2"): ("bo", "phiên âm thơ chữ Hán (bài mở đầu Tam quốc)"),
    # --- Trần Trọng Kim
    ("tran_trong_kim", "Phật giáo"): ("nghi_luan", "ba bài diễn thuyết; đề xuất tho sai do dòng tiêu đề"),
    # --- Tản Đà
    ("tan_da", "Nhàn tưởng"): ("nghi_luan", "tập tạp văn, tiểu luận"),
    ("tan_da", "Đánh bạc"): ("nghi_luan", "bài luận ngắn"),
    ("tan_da", "Giấc mộng con"): ("tu_su", "truyện tưởng tượng (du hành trong mộng), văn xuôi chiếm phần lớn; thơ chèn bị tách bỏ"),
    ("tan_da", "Lên tám"): ("", "sách học vỡ lòng lẫn thơ và văn, không gán"),
    ("tan_da", "Tản Đà tùng văn"): ("", "tập lẫn thơ và văn xuôi, không gán"),
    ("tan_da", "Thơ Tản Đà"): ("tho", "tập thơ (khi tải chỉ lấy khối thơ)"),
    ("tan_da", "Đài gương kinh"): ("", "không đủ căn cứ"),
    # --- Nguyễn Du, Nguyễn Đình Chiểu (truyện thơ / thơ Hán)
    ("nguyen_du", "Hành lạc từ"): ("bo", "thơ chữ Hán"),
    ("nguyen_du", "Thu nhật ký hứng"): ("bo", "thơ chữ Hán (đề xuất 'ky' sai vì tiêu đề có chữ 'nhật ký')"),
    # phát hiện SAU khi tải, bằng tỷ lệ dòng phiên âm Hán Việt (không có chữ Hán/dấu hiệu 'phiên âm' trên trang)
    ("nguyen_du", "Hoàng Châu trúc lâu"): ("bo", "thơ chữ Hán: 59% dòng là phiên âm Hán Việt"),
    ("nguyen_du", "Mạnh Tử từ cổ liễu"): ("bo", "thơ chữ Hán: 36% dòng là phiên âm Hán Việt, còn lại là bản dịch"),
    ("phan_boi_chau", "Trông mưa"): ("", "29% dòng là phiên âm Hán Việt; không chắc thơ Hán hay thơ Nôm chèn câu Hán"),
    ("nguyen_dinh_chieu", "Viếng cụ Phan Thanh Giản"): ("", "có cặp câu chữ Hán (phiên âm) lẫn thơ Nôm; không chắc"),
    ("ho_chi_minh", "Nội dung chủ nghĩa anh hùng cách mạng của chúng ta"):
        ("", "trang nhiều chữ Hán (128) mà ít văn xuôi (90 từ), có thể là bản song ngữ; không chắc"),
    ("nguyen_dinh_chieu", "Lục Vân Tiên"): ("tho", "truyện thơ Nôm của tác giả"),
    # --- Hàn Mặc Tử, Nguyễn Bính, Bích Khê, Nguyễn Nhược Pháp, Đông Hồ: tập thơ (trang mục lục)
    ("han_mac_tu", "Chơi giữa mùa trăng"): ("", "văn xuôi trữ tình (thơ văn xuôi), không xếp vào 4 thể loại"),
    ("nguyen_binh", "Lỡ bước sang ngang"): ("tho", "tập thơ"),
    ("nguyen_binh", "Mây tần"): ("tho", "tập thơ"),
    ("nguyen_binh", "Mười hai bến nước"): ("tho", "tập thơ"),
    ("nguyen_binh", "Tâm hồn tôi"): ("tho", "tập thơ"),
    ("nguyen_binh", "Người con gái lầu hoa"): ("tho", "tập thơ"),
    ("bich_khe", "Tinh huyết"): ("tho", "tập thơ"),
    ("nguyen_nhuoc_phap", "Ngày xưa"): ("tho", "tập thơ"),
    ("dong_ho", "Cô gái xuân"): ("tho", "tập thơ"),
    ("dong_ho", "Bội lan hành"): ("tho", "tập thơ"),
    ("dong_ho", "Trinh trắng"): ("", "không chắc là thơ hay văn xuôi"),
    # --- Khái Hưng, Nhất Linh: tác phẩm viết chung
    ("khai_hung", "Anh phải sống"): ("bo", "Khái Hưng và Nhất Linh viết chung"),
    ("khai_hung", "Anh phải sống/Tình tuyệt vọng"): ("bo", "thuộc tập 'Anh phải sống' viết chung"),
    ("nhat_linh", "Anh phải sống"): ("bo", "Khái Hưng và Nhất Linh viết chung"),
    # --- Huỳnh Tịnh Của
    ("huynh_tinh_cua", "Đại Nam Quấc âm tự vị"): ("bo", "từ điển"),
    ("huynh_tinh_cua", "Chuyện giải buồn"): ("bo", "'dịch rút trong các sách hay' — truyện dịch, phỏng dịch"),
    ("huynh_tinh_cua", "Chuyện giải buồn cuốn sau"): ("bo", "truyện dịch, phỏng dịch"),
    ("huynh_tinh_cua", "Sách quan chế"): ("", "biên soạn, có thể dịch; không chắc"),
    ("huynh_tinh_cua", "Tục ngữ, cổ ngữ, gia ngôn"): ("bo", "tục ngữ dân gian sưu tầm"),
    ("huynh_tinh_cua", "Câu hát góp"): ("bo", "câu hát dân gian sưu tầm"),
    ("huynh_tinh_cua", "Ca trù thể cách"): ("", "biên khảo kèm bài hát của nhiều người, không gán"),
    # --- Hồ Chí Minh
    ("ho_chi_minh", "Cảnh khuya"): ("tho", "bài thơ quốc ngữ"),
    ("ho_chi_minh", "Di chúc của Chủ tịch Hồ Chí Minh"): ("nghi_luan", "di chúc (genres.json: thuộc nghi_luan)"),
    ("ho_chi_minh", "Sửa đổi lối làm việc"): ("nghi_luan", "sách chính luận"),
    ("ho_chi_minh", "Mừng ngày Chúa giáng sinh"): ("nghi_luan", "thư, lời chúc gửi đồng bào"),
    ("ho_chi_minh", "Thư Chủ tịch Hồ Chí Minh gửi các học sinh nhân ngày khai giảng đầu tiên ở nước Việt Nam Dân chủ "
                    "Cộng hòa năm 1945"): ("nghi_luan", "thư"),
    ("ho_chi_minh", "Sắc luật cấm chỉ mọi hành động đầu cơ về kinh tế nước Việt Nam Dân chủ Cộng hòa"):
        ("bo", "văn bản pháp luật (genres.json: không thuộc nghi_luan)"),
    ("ho_chi_minh", "Sắc luật quy định những trường hợp phạm pháp quả tang, những trường hợp khẩn cấp, và những trường "
                    "hợp khám người phạm pháp quả tang nước Việt Nam Dân chủ Cộng hòa"):
        ("bo", "văn bản pháp luật (genres.json: không thuộc nghi_luan)"),
    ("ho_chi_minh", "Sắc luật về quyền tự do xuất bản nước Việt Nam Dân chủ Cộng hòa"):
        ("bo", "văn bản pháp luật (genres.json: không thuộc nghi_luan)"),
    ("ho_chi_minh", "Sắc luật về bầu cử Hội đồng nhân dân và Ủy ban hành chính các cấp nước Việt Nam Dân chủ Cộng hòa"):
        ("bo", "văn bản pháp luật (genres.json: không thuộc nghi_luan)"),
    ("ho_chi_minh", "Ngục trung nhật ký"): ("bo", "thơ chữ Hán, bản quốc ngữ là bản dịch của người khác"),
    # --- Phan Bội Châu
    ("phan_boi_chau", "Lời hỏi? Các anh em thanh niên"): ("nghi_luan", "bài luận gửi thanh niên"),
    ("phan_boi_chau", "Cao đẳng quốc dân"): ("nghi_luan", "sách chính luận"),
    ("phan_boi_chau", "Tạ ơn người bạn gái cho cái bao đầu"): ("tho", "bài thơ"),
    ("phan_boi_chau", "Thuốc hoàn hồn"): ("nghi_luan", "sách chính luận"),
    ("phan_boi_chau", "Nam quốc dân tu tri"): ("", "không đủ căn cứ về hình thức"),
    ("phan_boi_chau", "Pháp-Việt đề huề chính kiến thư"): ("bo", "nguyên văn chữ Hán kèm bản dịch"),
    ("phan_boi_chau", "Tờ thông cáo toàn quốc của Phan Bội Châu"): ("bo", "nguyên văn chữ Hán kèm bản dịch"),
    # --- Trương Vĩnh Ký, Trương Minh Ký
    ("truong_vinh_ky", "Gia Định báo"): ("bo", "tờ báo, nhiều người viết"),
    ("truong_minh_ky", "Ca từ diễn nghĩa"): ("bo", "diễn nghĩa bài chữ Hán của Tư Mã Quang"),
    ("truong_minh_ky", "Tuồng Joseph"): ("bo", "tuồng (kịch)"),
    ("truong_minh_ky", "Ấu học khải mông"): ("bo", "diễn nghĩa sách vỡ lòng chữ Hán"),
    ("truong_minh_ky", "Nữ nhi ca"): ("", "có thể là diễn ca từ sách Hán, không chắc"),
    ("truong_minh_ky", "Thi pháp nhập môn"): ("", "sách dạy làm thơ lẫn văn và thơ mẫu, không chắc"),
    # --- các tác giả khác
    ("diep_van_ky", "Nước non nhà"): ("nghi_luan", "tập bài luận"),
    ("hoang_tich_chu", "Cái án tú Khôi - cử Kỳ"): ("nghi_luan", "bài báo châm biếm, bình luận"),
    ("hoang_tich_chu", "Nghe quẻ bói của Thông Reo mà Văn Tôi mừng"): ("nghi_luan", "bút chiến"),
    ("hoang_tich_chu", "Cần có bọn học giả ấy của ông Phan Khôi"): ("nghi_luan", "bút chiến"),
    ("nguyen_ba_hoc", "Chuyện cô Chiêu Nhì"): ("tu_su", "truyện ngắn"),
    ("bui_the_my", "Quyền lợi và tương tế"): ("nghi_luan", "bài luận"),
    ("huynh_thuc_khang", "Việt Nam ta cũng có người hay thi"): ("nghi_luan", "bài luận (có trích thơ)"),
    ("phan_chau_trinh", "Các kiến nghị vắn tắt của Phan Châu Trinh do Roux dịch chuyển cho bộ trưởng thuộc địa và Sarraut"):
        ("bo", "bản dịch (tiêu đề ghi 'do Roux dịch')"),
    ("phan_chau_trinh", "Thư cảm ơn Toàn quyền Sarraut đã cho ở thêm một năm"): ("", "có thể là bản dịch từ tiếng Pháp, không chắc"),
    ("phan_chau_trinh", "Người Nam nên học chữ Pháp"): ("", "có thể là bản dịch, không chắc"),
    ("phan_chau_trinh", "Một đoạn đối thoại với cố đạo M. ở Phước Kiều"): ("", "có thể là bản dịch, không chắc"),
    ("phan_chau_trinh", "Cuộc nói chuyện với quan thống soái Sài Gòn ở trên đảo Côn Lôn"): ("", "có thể là bản dịch, không chắc"),
    ("phan_chau_trinh", "Giai nhân kỳ ngộ chi ca"): ("bo", "diễn ca phỏng dịch tiểu thuyết Nhật 'Kajin no kigū'"),
    ("hoang_ngoc_phach", "Tố Tâm"): ("tu_su", "tiểu thuyết"),
    ("dam_phuong", "Nữ quốc dân tu tri/Phụ trương-2"): ("nghi_luan", "bài giảng, luận"),
    ("duong_ba_trac", "Tiếng gọi đàn"): ("nghi_luan", "sách chính luận"),
    ("duong_ba_trac", "Trai lành gái tốt"): ("", "không đủ căn cứ"),
    ("duong_ba_trac", "Nét mực tình"): ("bo", "hợp tuyển nhiều tác giả"),
    ("duong_ba_trac", "Gia lễ giản yếu"): ("nghi_luan", "sách biên khảo lễ nghi; đề xuất tho sai do dòng tiêu đề"),
    ("duong_ba_trac", "Chữ nho học lấy"): ("", "không đủ căn cứ"),
    ("dao_trinh_nhat", "Vương Dương Minh/Phụ lục-5"): ("nghi_luan", "khảo cứu"),
    ("nguyen_chanh_sat", "Gái trả thù cha"): ("tu_su", "tiểu thuyết; đề xuất tho sai do phần thơ chèn và dòng tiêu đề"),
    ("nguyen_van_vinh", "Hội Kiếp-bạc"): ("ky", "tác giả kể chuyến đi dự hội Kiếp Bạc"),
}
NVV_KY_TO_ESSAY = ["Ma to dỗ nhớn", "Thói tệ", "Thiếu gạo ăn thừa giấy đốt", "Lính-tuần, lính-lệ", "Phán, ký",
                   "Duy tân (Nguyễn Văn Vĩnh)", "Chết vì gạo", "Truyện ăn mày", "Đốt pháo", "Tiếng An-nam",
                   "Chữ Quốc-ngữ", "Tật huyền hồ sáo hủ", "Con xâu đổ đầu nồi canh",
                   "Nhời đàn bà (Propos de femme) - Lắm vợ", "Chữ nho nên để hay là nên bỏ", "Hội dịch sách",
                   "Phiên mổ bò"]
for _w in NVV_KY_TO_ESSAY:
    OVERRIDES[("nguyen_van_vinh", _w)] = ("nghi_luan", NVV_ESSAY)

NOT_FOUND = {"hoang_dao": "Wikisource không có trang Tác gia (đã tìm cả tên thật Nguyễn Tường Long)",
             "tran_huy_lieu": "Wikisource không có trang Tác gia; kết quả tìm kiếm chỉ là nghị quyết, văn bản nhà nước"}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    rows = read_csv(WORKS_CSV)
    feat = {(r["author"], r["work"]): r for r in read_csv(V5 / "review_features.csv")}
    changes, used = [], set()
    for r in rows:
        key = (r["author"], r["work"])
        old = r["de_xuat"] if r["de_xuat"] != "?" else ""
        new, note = old, ""
        f = feat.get(key)
        if r["author"] in NOT_FOUND:
            new, note = "", NOT_FOUND[r["author"]]
        elif key in OVERRIDES:
            new, note = OVERRIDES[key]
            used.add(key)
        elif (f and str(r["ok"]) == "1" and r["source"] == "ws" and r["de_xuat"] in ("tho", "?")
              and (int(f["cjk"] or 0) >= 10 or any(m in f["markers"] for m in ("phiên âm", "dịch nghĩa", "dịch thơ",
                                                                                   "nguyên văn")))):
            if key in HAN_WHITELIST:
                new, note = (r["de_xuat"] if r["de_xuat"] != "?" else "tho"), KEEP_HAN
            else:
                new, note = "bo", HAN_NOTE
        r["genre"] = new
        if note:
            r["ghi_chu"] = note
        if new != old:
            changes.append({"author": r["author"], "work": r["work"], "source": r["source"], "de_xuat": r["de_xuat"],
                            "genre": new or "(trống)", "ly_do_doi": note, "url": r.get("url", "")})
    missing = set(OVERRIDES) - used
    if missing:
        print("CẢNH BÁO: override không khớp dòng nào:", sorted(missing))
    write_csv(WORKS_CSV, rows, WORK_FIELDS)
    with open(V5 / "review_changes.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["author", "work", "source", "de_xuat", "genre", "ly_do_doi", "url"])
        w.writeheader()
        w.writerows(changes)
    from collections import Counter
    print("số thay đổi so với đề xuất:", len(changes))
    print(Counter((c["de_xuat"], c["genre"]) for c in changes).most_common())
    return 0


if __name__ == "__main__":
    sys.exit(main())
