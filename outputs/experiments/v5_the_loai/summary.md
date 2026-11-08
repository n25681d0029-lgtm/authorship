# Thí nghiệm theo thể loại: v5_the_loai

Dữ liệu: `D:\apps\programming app\VS code\code data\authorship-NLP\authorship_attribution\data\v5_corpus.csv`

Seed: 42, 43, 44 (trung bình ± độ lệch chuẩn giữa các seed). Cấu hình mạng lai: backbone=vinai/phobert-base-v2, epochs=10, patience=3, batch=16, max_length=256, oversample=True, class_bias=True, char_ngram=False.

Cột `tat_ca` gộp mọi thể loại (nhãn = thể loại|tác giả). `chung_chon_sai` gộp mọi cặp (đoạn, thể loại sai).

## Macro-F1 — đoạn đầy đủ

| cách làm | mô hình | nghi_luan | tho | tu_su | tat_ca | trung bình các thể loại |
|---|---|---|---|---|---|---|
| rieng_the_loai | burrows_delta | 0.5414 ± 0.0818 | 0.3361 ± 0.0137 | 0.5499 ± 0.0416 | 0.4688 ± 0.0283 | 0.4758 |
| rieng_the_loai | char_ngram_svm | 0.8096 ± 0.0582 | 0.6232 ± 0.0322 | 0.9024 ± 0.0380 | 0.7672 ± 0.0281 | 0.7784 |
| rieng_the_loai | stylometry_logreg | 0.6748 ± 0.0498 | 0.5193 ± 0.0589 | 0.7206 ± 0.0331 | 0.6302 ± 0.0275 | 0.6382 |
| rieng_the_loai | concat | 0.7930 ± 0.0536 | 0.6222 ± 0.0591 | 0.8512 ± 0.0445 | 0.7475 ± 0.0545 | 0.7555 |
| chung_loc_the_loai | burrows_delta | 0.4721 ± 0.0176 | 0.2445 ± 0.0476 | 0.5753 ± 0.0521 | 0.4175 ± 0.0165 | 0.4306 |
| chung_loc_the_loai | char_ngram_svm | 0.8140 ± 0.0318 | 0.6414 ± 0.0469 | 0.8965 ± 0.0411 | 0.7736 ± 0.0232 | 0.7840 |
| chung_loc_the_loai | stylometry_logreg | 0.6473 ± 0.0479 | 0.4688 ± 0.0430 | 0.7280 ± 0.0374 | 0.6020 ± 0.0273 | 0.6147 |
| chung_loc_the_loai | concat | 0.7804 ± 0.0429 | 0.6492 ± 0.0851 | 0.8725 ± 0.0167 | 0.7566 ± 0.0341 | 0.7674 |
| chung_khong_loc | burrows_delta | 0.2667 ± 0.0287 | 0.2262 ± 0.0361 | 0.1515 ± 0.0327 | 0.2106 ± 0.0139 | 0.2148 |
| chung_khong_loc | char_ngram_svm | 0.5980 ± 0.0264 | 0.5601 ± 0.0639 | 0.5842 ± 0.1185 | 0.5789 ± 0.0414 | 0.5808 |
| chung_khong_loc | stylometry_logreg | 0.3981 ± 0.0508 | 0.3307 ± 0.0556 | 0.2847 ± 0.0494 | 0.3356 ± 0.0037 | 0.3378 |
| chung_khong_loc | concat | 0.6519 ± 0.0603 | 0.5507 ± 0.0708 | 0.5380 ± 0.0890 | 0.5818 ± 0.0563 | 0.5802 |
| chung_chon_sai | burrows_delta | 0.0221 ± 0.0023 | 0.0463 ± 0.0081 | 0.0429 ± 0.0009 | 0.0371 ± 0.0022 | 0.0371 |
| chung_chon_sai | char_ngram_svm | 0.0366 ± 0.0015 | 0.0449 ± 0.0084 | 0.0476 ± 0.0031 | 0.0526 ± 0.0009 | 0.0430 |
| chung_chon_sai | stylometry_logreg | 0.0316 ± 0.0008 | 0.0398 ± 0.0068 | 0.0472 ± 0.0005 | 0.0489 ± 0.0021 | 0.0395 |
| chung_chon_sai | concat | 0.0472 ± 0.0051 | 0.0680 ± 0.0066 | 0.0668 ± 0.0149 | 0.0621 ± 0.0038 | 0.0607 |

## Accuracy — đoạn đầy đủ

| cách làm | mô hình | nghi_luan | tho | tu_su | tat_ca | trung bình các thể loại |
|---|---|---|---|---|---|---|
| rieng_the_loai | burrows_delta | 0.5618 ± 0.0477 | 0.4623 ± 0.0299 | 0.5756 ± 0.0416 | 0.5397 ± 0.0103 | 0.5332 |
| rieng_the_loai | char_ngram_svm | 0.8248 ± 0.0394 | 0.8095 ± 0.0422 | 0.9112 ± 0.0301 | 0.8459 ± 0.0221 | 0.8485 |
| rieng_the_loai | stylometry_logreg | 0.6975 ± 0.0371 | 0.6962 ± 0.0774 | 0.7351 ± 0.0280 | 0.7081 ± 0.0187 | 0.7096 |
| rieng_the_loai | concat | 0.8098 ± 0.0338 | 0.8060 ± 0.0169 | 0.8602 ± 0.0478 | 0.8235 ± 0.0276 | 0.8254 |
| chung_loc_the_loai | burrows_delta | 0.5102 ± 0.0129 | 0.3166 ± 0.0322 | 0.5723 ± 0.0537 | 0.4775 ± 0.0214 | 0.4664 |
| chung_loc_the_loai | char_ngram_svm | 0.8269 ± 0.0282 | 0.8279 ± 0.0196 | 0.9095 ± 0.0329 | 0.8512 ± 0.0164 | 0.8548 |
| chung_loc_the_loai | stylometry_logreg | 0.6676 ± 0.0436 | 0.6558 ± 0.0471 | 0.7351 ± 0.0280 | 0.6842 ± 0.0204 | 0.6862 |
| chung_loc_the_loai | concat | 0.7883 ± 0.0359 | 0.8258 ± 0.0450 | 0.8849 ± 0.0188 | 0.8263 ± 0.0101 | 0.8330 |
| chung_khong_loc | burrows_delta | 0.4534 ± 0.0077 | 0.3129 ± 0.0298 | 0.3502 ± 0.0506 | 0.3866 ± 0.0225 | 0.3722 |
| chung_khong_loc | char_ngram_svm | 0.8161 ± 0.0226 | 0.8188 ± 0.0171 | 0.8865 ± 0.0172 | 0.8373 ± 0.0121 | 0.8405 |
| chung_khong_loc | stylometry_logreg | 0.6409 ± 0.0438 | 0.6394 ± 0.0496 | 0.6726 ± 0.0383 | 0.6497 ± 0.0158 | 0.6510 |
| chung_khong_loc | concat | 0.7851 ± 0.0359 | 0.8167 ± 0.0430 | 0.8635 ± 0.0155 | 0.8163 ± 0.0133 | 0.8218 |
| chung_chon_sai | burrows_delta | 0.0723 ± 0.0133 | 0.1004 ± 0.0150 | 0.1299 ± 0.0050 | 0.0964 ± 0.0106 | 0.1009 |
| chung_chon_sai | char_ngram_svm | 0.1113 ± 0.0056 | 0.1444 ± 0.0174 | 0.1760 ± 0.0012 | 0.1388 ± 0.0075 | 0.1439 |
| chung_chon_sai | stylometry_logreg | 0.1000 ± 0.0073 | 0.1214 ± 0.0148 | 0.1620 ± 0.0063 | 0.1237 ± 0.0083 | 0.1278 |
| chung_chon_sai | concat | 0.1091 ± 0.0073 | 0.1434 ± 0.0213 | 0.1752 ± 0.0005 | 0.1373 ± 0.0088 | 0.1426 |

## Macro-F1 — đoạn ngắn (first_25w)

| cách làm | mô hình | nghi_luan | tho | tu_su | tat_ca | trung bình các thể loại |
|---|---|---|---|---|---|---|
| rieng_the_loai | burrows_delta | 0.1270 ± 0.0245 | 0.0659 ± 0.0073 | 0.2109 ± 0.0496 | 0.1262 ± 0.0135 | 0.1346 |
| rieng_the_loai | char_ngram_svm | 0.5582 ± 0.0123 | 0.3736 ± 0.0605 | 0.6583 ± 0.0559 | 0.5163 ± 0.0186 | 0.5300 |
| rieng_the_loai | stylometry_logreg | 0.3434 ± 0.0248 | 0.2845 ± 0.0244 | 0.4092 ± 0.0571 | 0.3379 ± 0.0056 | 0.3457 |
| rieng_the_loai | concat | 0.5216 ± 0.0526 | 0.3333 ± 0.0358 | 0.5738 ± 0.0498 | 0.4671 ± 0.0378 | 0.4763 |
| chung_loc_the_loai | burrows_delta | 0.1363 ± 0.0210 | 0.0220 ± 0.0073 | 0.1895 ± 0.0302 | 0.1121 ± 0.0164 | 0.1159 |
| chung_loc_the_loai | char_ngram_svm | 0.5546 ± 0.0239 | 0.3694 ± 0.0181 | 0.6656 ± 0.0563 | 0.5170 ± 0.0016 | 0.5299 |
| chung_loc_the_loai | stylometry_logreg | 0.2944 ± 0.0237 | 0.2496 ± 0.0259 | 0.4053 ± 0.0331 | 0.3050 ± 0.0074 | 0.3164 |
| chung_loc_the_loai | concat | 0.4815 ± 0.0326 | 0.3105 ± 0.0225 | 0.6533 ± 0.0747 | 0.4626 ± 0.0289 | 0.4818 |
| chung_khong_loc | burrows_delta | 0.0042 ± 0.0072 | 0.0220 ± 0.0073 | 0.0000 ± 0.0000 | 0.0080 ± 0.0048 | 0.0087 |
| chung_khong_loc | char_ngram_svm | 0.3435 ± 0.0272 | 0.1646 ± 0.0220 | 0.2505 ± 0.0253 | 0.2488 ± 0.0188 | 0.2529 |
| chung_khong_loc | stylometry_logreg | 0.1360 ± 0.0202 | 0.1143 ± 0.0143 | 0.0982 ± 0.0151 | 0.1162 ± 0.0069 | 0.1162 |
| chung_khong_loc | concat | 0.3240 ± 0.0260 | 0.1792 ± 0.0286 | 0.3208 ± 0.1417 | 0.2636 ± 0.0217 | 0.2747 |
| chung_chon_sai | burrows_delta | 0.0066 ± 0.0020 | 0.0322 ± 0.0072 | 0.0083 ± 0.0047 | 0.0125 ± 0.0025 | 0.0157 |
| chung_chon_sai | char_ngram_svm | 0.0326 ± 0.0042 | 0.0396 ± 0.0009 | 0.0458 ± 0.0023 | 0.0480 ± 0.0010 | 0.0393 |
| chung_chon_sai | stylometry_logreg | 0.0238 ± 0.0025 | 0.0262 ± 0.0028 | 0.0315 ± 0.0037 | 0.0322 ± 0.0018 | 0.0271 |
| chung_chon_sai | concat | 0.0328 ± 0.0059 | 0.0388 ± 0.0061 | 0.0528 ± 0.0104 | 0.0482 ± 0.0039 | 0.0415 |

## Macro-F1 — đoạn ngắn (first_50w)

| cách làm | mô hình | nghi_luan | tho | tu_su | tat_ca | trung bình các thể loại |
|---|---|---|---|---|---|---|
| rieng_the_loai | burrows_delta | 0.2374 ± 0.0526 | 0.1231 ± 0.0087 | 0.3613 ± 0.0414 | 0.2277 ± 0.0133 | 0.2406 |
| rieng_the_loai | char_ngram_svm | 0.6773 ± 0.0384 | 0.4820 ± 0.0588 | 0.7655 ± 0.0334 | 0.6322 ± 0.0210 | 0.6416 |
| rieng_the_loai | stylometry_logreg | 0.4451 ± 0.0097 | 0.4192 ± 0.0360 | 0.5628 ± 0.0644 | 0.4642 ± 0.0234 | 0.4757 |
| rieng_the_loai | concat | 0.6476 ± 0.0466 | 0.4303 ± 0.0633 | 0.7242 ± 0.0529 | 0.5889 ± 0.0513 | 0.6007 |
| chung_loc_the_loai | burrows_delta | 0.1924 ± 0.0059 | 0.0406 ± 0.0264 | 0.3351 ± 0.0496 | 0.1775 ± 0.0201 | 0.1894 |
| chung_loc_the_loai | char_ngram_svm | 0.6745 ± 0.0310 | 0.5397 ± 0.0618 | 0.7673 ± 0.0105 | 0.6521 ± 0.0268 | 0.6605 |
| chung_loc_the_loai | stylometry_logreg | 0.3977 ± 0.0358 | 0.3258 ± 0.0347 | 0.5262 ± 0.0655 | 0.4027 ± 0.0143 | 0.4166 |
| chung_loc_the_loai | concat | 0.6164 ± 0.0324 | 0.4724 ± 0.0591 | 0.7505 ± 0.0622 | 0.5988 ± 0.0193 | 0.6131 |
| chung_khong_loc | burrows_delta | 0.0637 ± 0.0092 | 0.0406 ± 0.0264 | 0.0043 ± 0.0075 | 0.0379 ± 0.0136 | 0.0362 |
| chung_khong_loc | char_ngram_svm | 0.4570 ± 0.0276 | 0.2387 ± 0.0363 | 0.3170 ± 0.0051 | 0.3322 ± 0.0261 | 0.3376 |
| chung_khong_loc | stylometry_logreg | 0.2068 ± 0.0251 | 0.1462 ± 0.0223 | 0.1535 ± 0.0285 | 0.1682 ± 0.0067 | 0.1688 |
| chung_khong_loc | concat | 0.4282 ± 0.0386 | 0.3351 ± 0.0718 | 0.3690 ± 0.1111 | 0.3740 ± 0.0306 | 0.3774 |
| chung_chon_sai | burrows_delta | 0.0119 ± 0.0029 | 0.0452 ± 0.0042 | 0.0120 ± 0.0019 | 0.0195 ± 0.0015 | 0.0231 |
| chung_chon_sai | char_ngram_svm | 0.0326 ± 0.0036 | 0.0462 ± 0.0043 | 0.0489 ± 0.0018 | 0.0507 ± 0.0019 | 0.0426 |
| chung_chon_sai | stylometry_logreg | 0.0290 ± 0.0033 | 0.0302 ± 0.0042 | 0.0399 ± 0.0035 | 0.0410 ± 0.0016 | 0.0331 |
| chung_chon_sai | concat | 0.0396 ± 0.0066 | 0.0450 ± 0.0057 | 0.0578 ± 0.0083 | 0.0541 ± 0.0024 | 0.0474 |

## Chéo thể loại — mô hình riêng của thể loại train đoán đoạn thể loại khác của cùng tác giả

`acc_cung_the_loai`: accuracy của cùng mô hình trên đoạn test cùng thể loại của tác giả đó (mốc so sánh).

| mô hình | train → test | tác giả | số đoạn | acc chéo | acc cùng thể loại | đoán nhiều nhất |
|---|---|---|---|---|---|---|
| burrows_delta | nghi_luan → tho | ho_chi_minh | 59 | 0.017 ± 0.017 | 0.905 | duong_ba_trac |
| burrows_delta | nghi_luan → tho | phan_boi_chau | 82 | 0.280 ± 0.068 | 0.454 | duong_ba_trac |
| burrows_delta | nghi_luan → tu_su | ngo_tat_to | 200 | 0.013 ± 0.003 | 0.407 | phan_ke_binh |
| burrows_delta | tho → nghi_luan | ho_chi_minh | 200 | 0.905 ± 0.061 | 1.000 | ho_chi_minh |
| burrows_delta | tho → nghi_luan | phan_boi_chau | 200 | 0.008 ± 0.003 | 0.275 | ho_chi_minh |
| burrows_delta | tho → tu_su | ho_bieu_chanh | 200 | 0.003 ± 0.006 | 0.925 | ho_chi_minh |
| burrows_delta | tho → tu_su | tan_da | 104 | 0.006 ± 0.006 | 0.100 | ho_chi_minh |
| burrows_delta | tu_su → nghi_luan | ngo_tat_to | 100 | 0.330 ± 0.104 | 0.591 | tan_da |
| burrows_delta | tu_su → tho | ho_bieu_chanh | 98 | 0.071 ± 0.020 | 0.860 | nguyen_chanh_sat |
| burrows_delta | tu_su → tho | tan_da | 96 | 0.531 ± 0.184 | 0.890 | tan_da |
| char_ngram_svm | nghi_luan → tho | ho_chi_minh | 59 | 0.576 ± 0.051 | 0.976 | ho_chi_minh |
| char_ngram_svm | nghi_luan → tho | phan_boi_chau | 82 | 0.346 ± 0.025 | 0.826 | phan_ke_binh |
| char_ngram_svm | nghi_luan → tu_su | ngo_tat_to | 200 | 0.027 ± 0.003 | 0.528 | phan_ke_binh |
| char_ngram_svm | tho → nghi_luan | ho_chi_minh | 200 | 0.610 ± 0.256 | 0.891 | ho_chi_minh |
| char_ngram_svm | tho → nghi_luan | phan_boi_chau | 200 | 0.327 ± 0.249 | 0.625 | phan_boi_chau |
| char_ngram_svm | tho → tu_su | ho_bieu_chanh | 200 | 0.005 ± 0.005 | 0.975 | nguyen_dinh_chieu |
| char_ngram_svm | tho → tu_su | tan_da | 104 | 0.478 ± 0.161 | 0.800 | tan_da |
| char_ngram_svm | tu_su → nghi_luan | ngo_tat_to | 100 | 0.557 ± 0.068 | 0.930 | ngo_tat_to |
| char_ngram_svm | tu_su → tho | ho_bieu_chanh | 98 | 0.020 ± 0.010 | 0.977 | nguyen_chanh_sat |
| char_ngram_svm | tu_su → tho | tan_da | 96 | 0.392 ± 0.057 | 1.000 | ngo_tat_to |
| concat | nghi_luan → tho | ho_chi_minh | 59 | 0.689 ± 0.010 | 0.988 | ho_chi_minh |
| concat | nghi_luan → tho | phan_boi_chau | 82 | 0.252 ± 0.164 | 0.687 | phan_ke_binh |
| concat | nghi_luan → tu_su | ngo_tat_to | 200 | 0.053 ± 0.064 | 0.676 | nguyen_van_vinh |
| concat | tho → nghi_luan | ho_chi_minh | 200 | 0.973 ± 0.038 | 1.000 | ho_chi_minh |
| concat | tho → nghi_luan | phan_boi_chau | 200 | 0.200 ± 0.079 | 0.525 | tan_da |
| concat | tho → tu_su | ho_bieu_chanh | 200 | 0.170 ± 0.170 | 0.900 | ho_bieu_chanh |
| concat | tho → tu_su | tan_da | 104 | 0.510 ± 0.139 | 0.633 | tan_da |
| concat | tu_su → nghi_luan | ngo_tat_to | 100 | 0.363 ± 0.248 | 0.870 | ngo_tat_to |
| concat | tu_su → tho | ho_bieu_chanh | 98 | 0.037 ± 0.036 | 0.931 | nguyen_chanh_sat |
| concat | tu_su → tho | tan_da | 96 | 0.528 ± 0.016 | 0.911 | tan_da |
| stylometry_logreg | nghi_luan → tho | ho_chi_minh | 59 | 0.395 ± 0.035 | 0.965 | ho_chi_minh |
| stylometry_logreg | nghi_luan → tho | phan_boi_chau | 82 | 0.207 ± 0.037 | 0.628 | phan_ke_binh |
| stylometry_logreg | nghi_luan → tu_su | ngo_tat_to | 200 | 0.047 ± 0.021 | 0.500 | tran_trong_kim |
| stylometry_logreg | tho → nghi_luan | ho_chi_minh | 200 | 0.477 ± 0.078 | 0.841 | ho_chi_minh |
| stylometry_logreg | tho → nghi_luan | phan_boi_chau | 200 | 0.220 ± 0.056 | 0.475 | phan_boi_chau |
| stylometry_logreg | tho → tu_su | ho_bieu_chanh | 200 | 0.000 ± 0.000 | 0.825 | tran_te_xuong |
| stylometry_logreg | tho → tu_su | tan_da | 104 | 0.343 ± 0.039 | 0.683 | tan_da |
| stylometry_logreg | tu_su → nghi_luan | ngo_tat_to | 100 | 0.230 ± 0.061 | 0.659 | vu_trong_phung |
| stylometry_logreg | tu_su → tho | ho_bieu_chanh | 98 | 0.231 ± 0.068 | 0.919 | ngo_tat_to |
| stylometry_logreg | tu_su → tho | tan_da | 96 | 0.288 ± 0.043 | 0.866 | ngo_tat_to |
