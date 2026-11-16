# Sanad OCR Training Handoff — RTX 5060 Ti

## Objective

Continue Sanad's OCR model evaluation and training on the RTX 5060 Ti. Do not change the
Android application yet. First prove that the candidate OCR models outperform the existing
TrOCR models on the same held-out crop sets.

Repository:

```text
git@github.com:CosmicEventHorizon/sanad.git
```

Work on `main`. The Android project uses Clean Architecture; future OCR integration must stay
inside the infrastructure layer and continue implementing the existing domain interfaces.

## Current approach

We replaced the unsuccessful numeric TrOCR experiment with PP-OCRv6 Medium recognition.
The intended sequence is:

1. Fine-tune one shared PP-OCRv6 Medium recognizer on all Latin-script text and number crop
   types in `shared_train.txt`.
2. Evaluate its best checkpoint against the untouched shared and numeric test sets.
3. Fine-tune a second numeric specialization from the best shared checkpoint using
   `numeric_train.txt`.
4. Compare both models with the saved pretrained baseline and the older TrOCR results.
5. Do not modify or package models into Android until the benchmark supports that decision.

Arabic-script samples are excluded from this training corpus. Related crops and generated
variations are grouped into one split to prevent leakage.

## Data and model files are intentionally not in Git

The following are ignored because they are large:

```text
dataset/prepared/ppocrv6_recognition/                       # about 329 MB
ai/pretrained/PP-OCRv6_medium_rec_pretrained.pdparams       # 232,651,336 bytes
ai/runs/                                                    # checkpoints and optimizer state
```

Copy `dataset/prepared/ppocrv6_recognition/` from the RTX 2060 computer into the same path in
the clone. Alternatively, regenerate it with `ai/scripts/prepare_ppocrv6_recognition.py` after
making its external source datasets available. The prepared corpus contains:

| Task | Train | Validation | Test |
|---|---:|---:|---:|
| Shared text + numbers | 9,336 | 1,075 | 1,403 |
| Numeric specialization | 7,121 | 834 | 1,053 |

There is zero source-group overlap between splits. The shared test set includes 10 real
production-shaped fiche-number crops.

Download and validate the official trainable checkpoint with:

```bash
ai/scripts/download_ppocrv6_parallel.sh
```

The downloader is resumable at the chunk level and validates the final file with
`paddle.load`. Do not substitute the 76 MB Hugging Face inference model: training requires the
232.7 MB `.pdparams` checkpoint.

## PaddleOCR environment

Use official PaddleOCR v3.7.0. The previous machine used commit `b03f464`, Python 3.11,
PaddlePaddle GPU 3.3.1, and CUDA 12.9 successfully.

Suggested layout:

```text
sanad/
├── .venv-paddle/
└── .tools/PaddleOCR/     # official v3.7.0 checkout
```

Install the PaddlePaddle GPU wheel matching the RTX 5060 Ti machine's driver/CUDA support,
then install PaddleOCR's requirements. Confirm `paddle.device.is_compiled_with_cuda()` and a
small GPU tensor operation before training.

The committed configuration uses paths relative to `.tools/PaddleOCR`, so run PaddleOCR tools
with that directory as the working directory.

## Saved pretrained baseline

The untouched PP-OCRv6 Medium checkpoint was evaluated on all held-out shared crops:

| Set | CER | Exact match |
|---|---:|---:|
| Shared test (1,403 crops) | 5.21% | 70.9% |
| Real fiche-number subset (10 crops) | 2.5% | 8/10 |
| Numeric targets (1,053 crops) | 11.60% | 52.6% |

The general model often predicts useful units such as `147 cm`, whereas numeric targets expect
only `147`; this partly explains its worse numeric-target score. Use the same normalization and
metric script for all comparisons:

```text
ai/scripts/evaluate_ppocrv6_predictions.py
```

Reports and raw predictions are in `ai/reports/ppocrv6_baseline_*`.

## Train the shared recognizer

Configuration:

```text
ai/config/ppocrv6_shared_handwriting.yml
```

It uses the official PP-OCRv6 Medium architecture, `max_text_length: 40`, multilingual v6
dictionary, multi-scale 32/48/64-pixel training, `RecConAug`, `RecAug`, batch size 16, Adam with
cosine decay, learning rate `1e-4`, and 30 epochs. On the 16 GB RTX 5060 Ti, first test a larger
batch such as 32; use the largest stable batch and consider Paddle AMP only after a short
preflight confirms numerical stability.

Run from `.tools/PaddleOCR`:

```bash
../../.venv-paddle/bin/python -u tools/train.py \
  -c ../../ai/config/ppocrv6_shared_handwriting.yml
```

Expected RTX 5060 Ti duration is roughly 35–90 minutes, potentially less with a stable larger
batch/mixed precision. Save logs outside Git under `ai/runs/`.

The stopped RTX 2060 attempt reached epoch 2 and showed very high validation accuracy, but its
local 1.9 GB checkpoint directory is not in Git. Starting fresh on the 5060 Ti is acceptable.

## Evaluate after shared training

Run inference over `shared_test.txt` using `best_accuracy.pdparams`, write a new predictions
file, and run `evaluate_ppocrv6_predictions.py` twice:

- `shared_test.csv` for general recognition metrics.
- `numeric_test.csv` for numeric-target metrics.

Report overall CER, exact-match accuracy, real fiche-number results, and representative errors.
Do not use validation accuracy as the final result.

## Numeric specialization afterward

Create a numeric config derived from the shared config with these changes:

- Initialize from `ai/runs/ppocrv6_shared/best_accuracy`.
- Use `numeric_train.txt`, `numeric_val.txt`, and later `numeric_test.csv`.
- Start with 15–20 epochs and learning rate `5e-5`.
- Preserve digits and represented punctuation (`. , / : - + %`).
- Select by validation accuracy, then report held-out CER and exact match.

Keep the shared and numeric checkpoints separate. Do not copy either model into Android assets
until their held-out results and deployment format have been reviewed.
