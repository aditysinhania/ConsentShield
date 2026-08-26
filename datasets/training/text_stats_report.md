# MiniLM text training views — statistics

Sources included: `hf_synthetic`, `hf_manual`, `contextdp` (usable text only).
Excluded: B4E2 and any rows without usable OCR/text.

## train

- Samples: **9731**
- Avg text length: **608.95** (median 496.0)
- Min/max length: 3 / 5841

### Class distribution

- `dark_pattern`: 4867
- `no_dark_pattern`: 4864

### Source distribution

- `contextdp`: 400
- `hf_manual`: 922
- `hf_synthetic`: 8409

## val

- Samples: **1214**
- Avg text length: **603.33** (median 483.0)
- Min/max length: 9 / 6594

### Class distribution

- `dark_pattern`: 607
- `no_dark_pattern`: 607

### Source distribution

- `contextdp`: 49
- `hf_manual`: 115
- `hf_synthetic`: 1050

## test

- Samples: **1223**
- Avg text length: **607.81** (median 485.0)
- Min/max length: 14 / 3678

### Class distribution

- `dark_pattern`: 611
- `no_dark_pattern`: 612

### Source distribution

- `contextdp`: 52
- `hf_manual`: 116
- `hf_synthetic`: 1055
