# RTX 4060 training handoff

This bundle contains the University-TAMER code, processed training assets and
the official TAMER `version_3` fusion checkpoint. It does not contain the raw
MathWriting or HME100K archives.

## Expected hardware

- NVIDIA RTX 4060 (normally 8 GB VRAM)
- recent NVIDIA driver with CUDA 11.8 runtime support
- at least 16 GB system RAM; 32 GB is preferable
- at least 15 GB free disk space for environment, checkpoints and logs

The CUDA toolkit does not need to be installed separately. Conda installs the
PyTorch CUDA 11.8 runtime.

## 1. Check the extracted folder

Run every command from the folder containing `train_university.py`.

Required files:

```text
data/university/
data/hme_cache/
data/paper_backgrounds_processed/
data/HME100k/dictionary.txt
lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt
```

Do not move individual folders after extraction because manifests use paths
relative to the project root.

## 2. Create the RTX 4060 environment

Install Miniconda, open **Anaconda Prompt**, change to this project directory,
then run:

```bat
conda env create -f environment-rtx4060.yml
conda activate tamer-rtx4060
python -m pip install -e . --no-deps
```

Verify CUDA:

```bat
nvidia-smi
python -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available()); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('capability=',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NONE')"
```

Expected output includes `cuda=True` and `RTX 4060`.

## 3. Validate all assets

```bat
python scripts/validate_project.py --data-root data/university --hme-cache data/hme_cache --dictionary data/HME100k/dictionary.txt --checkpoint "lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt"
```

Do not train unless the report contains:

```json
"status": "PASS"
```

It must also show 10,000/1,000/1,000 University samples, 20,000 HME replay
samples, 1,000 HME validation samples and zero label leakage.

## 4. Run the short smoke test

```bat
python train_university.py --config config/smoke_test_rtx4060.yaml
```

The smoke test runs only 20 training batches and five validation batches. It
checks checkpoint loading, dynamic augmentation, FP16, CUDA and VRAM use.

Stop and report the complete error if any of these occurs:

- `CUDA out of memory`
- `no kernel image is available`
- `NaN` loss
- missing/unexpected checkpoint keys

## 5. Start the real fine-tune

```bat
python train_university.py --config config/university_baseline_rtx4060.yaml
```

The RTX 4060 configuration uses:

```text
micro batch size:       4
gradient accumulation:  8
effective batch size:  32
precision:             FP16
University/HME ratio: 60/40
maximum epochs:        20
early stopping:         4 validation checks without improvement
```

The script first calibrates the original `version_3` checkpoint on the fixed
HME validation cache, then starts fine-tuning.

## 6. Resume an interrupted run

```bat
python train_university.py --config config/university_baseline_rtx4060.yaml --resume outputs/university_baseline_rtx4060/checkpoints/last.ckpt
```

Use `last.ckpt` only for resume. The model used for final evaluation is the
checkpoint printed as `Best retention-aware checkpoint`.

## 7. If RTX 4060 runs out of memory

Edit both RTX 4060 YAML files:

```yaml
data:
  train_batch_size: 2
  eval_batch_size: 1

trainer:
  accumulate_grad_batches: 16
```

This keeps the effective batch size at 32. Close browsers, games and other GPU
applications before retrying.

## Outputs

```text
outputs/university_baseline_rtx4060/
├── checkpoints/
│   ├── last.ckpt
│   └── top retention-aware checkpoints
└── logs/
```

The fixed test set is not used to choose a checkpoint. Run final University
and full HME100K evaluation only after the best checkpoint has been selected.
