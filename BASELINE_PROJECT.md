# University-TAMER baseline

Baseline này giữ nguyên kiến trúc TAMER fusion và khởi tạo từ checkpoint `version_3`
(`val_ExpRate=0.6954`). Phần mới gồm MathWriting university subset, paper/camera
augmentation động, 40% HME100K replay và bộ metric thống nhất.

## Cấu trúc dữ liệu sau khi chuẩn bị

```text
data/
├── university/
│   ├── clean/{train,validation,test}/
│   ├── fixed/{validation,test}/
│   ├── splits/
│   │   ├── train_clean.csv
│   │   ├── validation_fixed.csv
│   │   └── test_fixed.csv
│   └── selection_summary.json
├── hme_cache/
│   ├── replay.csv
│   └── validation.csv
└── paper_backgrounds/     # tùy chọn: ảnh giấy trắng thật
```

## 1. Môi trường

Repo gốc ghim PyTorch Lightning 1.4.9. Nên tạo môi trường theo `README.md`
của TAMER, sau đó chạy:

```bash
conda env create -f environment-university.yml
conda activate tamer-university
```

Mọi lệnh bên dưới được chạy từ thư mục chứa `train_university.py`.

## 2. Chuẩn bị 10K/1K/1K MathWriting

```bash
python scripts/prepare_university_data.py \
  --mathwriting "../../mathwriting-2024/mathwriting-2024" \
  --dictionary data/HME100k/dictionary.txt \
  --output data/university
```

Script chỉ đọc split `train` gốc của MathWriting. Các split mới được tách theo
canonical label; nếu có label leakage, script dừng ngay. Mọi token OOV cũng bị loại.

Sinh validation/test cố định đúng một lần:

```bash
python scripts/prepare_paper_backgrounds.py \
  --source data/paper_backgrounds \
  --output data/paper_backgrounds_processed \
  --seed 7

python scripts/generate_fixed_splits.py \
  --data-root data/university \
  --validation-backgrounds data/paper_backgrounds_processed/validation \
  --test-backgrounds data/paper_backgrounds_processed/test \
  --seed 7
```

Train dùng `paper_backgrounds_processed/train` với xác suất 70% nền ảnh
thật và 30% nền procedural. Validation có 60/30/10 và test có
30/40/30 mẫu mild/medium/hard; metric được báo cáo riêng theo severity.

Xem trước augmentation trên một ảnh sạch:

```bash
python scripts/preview_augmentation.py \
  --image data/university/clean/train/MA_MAU.png \
  --output outputs/augmentation_preview
```

## 3. Tạo HME replay cache

Lệnh này đọc file pickle lớn một lần, sau đó training chỉ đọc PNG:

```bash
python scripts/prepare_hme_cache.py \
  --hme100k data/HME100k \
  --output data/hme_cache \
  --replay-size 20000 \
  --validation-size 1000
```

## 4. Đánh giá TAMER version_3 trước fine-tune

University fixed test:

```bash
python eval/evaluate_manifest.py \
  --checkpoint "lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt" \
  --dictionary data/HME100k/dictionary.txt \
  --manifest data/university/splits/test_fixed.csv \
  --data-root data/university \
  --output outputs/evaluations/version3/university
```

HME100K full test:

```bash
python eval/evaluate_hme100k.py \
  --checkpoint "lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt" \
  --hme100k data/HME100k \
  --output outputs/evaluations/version3/hme100k
```

Kiểm tra fail-fast trước khi dùng GPU:

```bash
python scripts/validate_project.py \
  --data-root data/university \
  --hme-cache data/hme_cache \
  --dictionary data/HME100k/dictionary.txt \
  --checkpoint "lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt"
```

## 5. Train baseline hoàn chỉnh

```bash
python train_university.py --config config/university_baseline.yaml
```

Nếu training bị ngắt:

```bash
python train_university.py --config config/university_baseline.yaml \
  --resume outputs/university_baseline_full/checkpoints/last.ckpt
```

- Hai epoch đầu đóng băng encoder và học decoder.
- Sau đó unfreeze toàn bộ với AdamW `5e-5`.
- Batch sampler duy trì trung bình 60% university / 40% HME replay.
- Trước epoch đầu, script tự chạy `version_3` trên HME validation cache và dùng
  chính baseline này làm mốc retention; không ép con số 69.54% của full test
  lên một subset 1.000 ảnh.
- Checkpoint theo `val_retention_score`: University ExpRate cao nhưng phạt nặng khi
  HME validation giảm quá 2 điểm phần trăm so với 0.6954.

## 6. Ablation

```bash
python train_university.py --config config/ablation_no_augmentation.yaml
python train_university.py --config config/ablation_augmentation_only.yaml
python train_university.py --config config/university_baseline.yaml
```

Ba thí nghiệm lần lượt là: fine-tune 10K; thêm augmentation; thêm 40% replay.

## Metric cố định

| Metric | Định nghĩa | Chiều tốt |
|---|---|---:|
| ExpRate | Đúng toàn bộ chuỗi token | Tăng |
| ExpRate <=1 | Khoảng cách edit không quá 1 token | Tăng |
| ExpRate <=2 | Khoảng cách edit không quá 2 token | Tăng |
| Token Error Rate | Tổng edit distance / tổng token ground truth | Giảm |
| Valid LaTeX | Chuỗi có ngoặc và script syntax hợp lệ | Tăng |

Mọi metric được tính bằng `tamer/university/metrics.py`, không được thay
công thức giữa các thí nghiệm. `fixed_test` chỉ chạy sau khi chốt checkpoint.
