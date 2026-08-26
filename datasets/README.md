# ConsentShield — Multimodal Datasets

Raw corpora live under `raw/`. The **unified processed dataset** (labels normalized, OCR, dedupe, splits) lives under `unified/`.

CookieScanner is documented under raw but **ignored** by the unified builder until manually downloaded.

```text
datasets/
  README.md                          ← this file
  raw/
    cookiescanner/                   ← CookieScanner (Zenodo)
    contextdp/                       ← ContextDP / AidUI evaluation set
    b4e2/                            ← B4E2 Dark Pattern YOLO dataset
    deceptive_patterns/              ← Hugging Face WIPI deceptive-pattern sets
```

---

## Legal / research-use summary

| Dataset | Public? | License | Research use |
|---------|---------|---------|--------------|
| CookieScanner | Yes (Zenodo open) | **CC BY 4.0** | Yes — attribution required |
| ContextDP (AidUI) | Yes (GitHub + Zenodo) | **MIT** (AidUI repo) | Yes |
| B4E2 Dark Pattern | Yes (GitHub LFS / Roboflow / Kaggle) | **CC BY 4.0** (Roboflow export in this tree); Kaggle listing also shows **CC BY-NC-SA 4.0** | Yes for academic research with attribution; prefer Roboflow/GitHub copy terms; avoid commercial redistribution if using the Kaggle NC variant |
| Hugging Face Deceptive Patterns | Yes (HF Hub) | **MIT** (`WIPI/deceptive_patterns_synthetic`) | Yes |

Always cite the original papers/authors. This is **not** legal advice — check each license before redistribution or commercial use.

---

## 1. CookieScanner

### Source
- Paper: *Cookiescanner: An Automated Tool for Detecting and Evaluating GDPR Consent Notices on Websites* (ARES 2023)
- Zenodo: https://zenodo.org/records/8334087  
- DOI: [10.5281/zenodo.8334087](https://doi.org/10.5281/zenodo.8334087)
- Code: https://github.com/UBA-PSI/cookiescanner

### License
**Creative Commons Attribution 4.0 International (CC BY 4.0)** — open access on Zenodo.

### Number of samples
- **1,000** sampled website scans (`02_raw_dataset`)
- Plus BERT classifier assets, banner-detection screenshots, and dark-pattern analysis screenshots (full archive **~6.4 GB**)

### Labels / content
Cookie consent measurement and dark-pattern analysis artifacts, including:
- Raw scan JSON (`results.json`)
- Banner / detection screenshots
- Dark-pattern analysis files and annotated banner screenshots  
Typical consent UI dark-pattern dimensions studied in the paper (e.g. presence/absence and design issues around accept/reject flows). Exact label schemas are inside the archive CSVs/JSON after download.

### File structure (expected after download)

```text
raw/cookiescanner/
  DOWNLOAD_REQUIRED.txt              ← present until manual download
  Cookiescanner_Dataset.zip          ← place archive here
  # after extract:
  01_bert_classifier/
  02_raw_dataset/
  03_banner_detection/
  04_dark_patterns/
```

### Download status on this machine
**Not downloaded automatically.**

**Why:** The Zenodo archive is **~6.4 GB**. This environment had **&lt; 8 GB free disk**, which is insufficient for a safe download + extract.

### Manual download instructions
1. Free **≥ 15 GB** disk (download + extract headroom).
2. Open https://zenodo.org/records/8334087
3. Download `Cookiescanner_Dataset.zip`
4. Save it to `datasets/raw/cookiescanner/Cookiescanner_Dataset.zip` **without renaming contents**.
5. Extract in place (optional):

```powershell
cd "datasets\raw\cookiescanner"
Expand-Archive .\Cookiescanner_Dataset.zip -DestinationPath .
# or: tar -xf .\Cookiescanner_Dataset.zip
```

Direct file URL:

```text
https://zenodo.org/records/8334087/files/Cookiescanner_Dataset.zip?download=1
```

---

## 2. ContextDP (AidUI)

### Source
- Paper: *AidUI: Toward Automated Recognition of Dark Patterns in User Interfaces* (ICSE 2023)
- Evaluation dataset release: https://github.com/SageSELab/AidUI/releases/tag/v-ICSE'23-data  
- Asset: `AidUI-Evaluation-Dataset.tar.gz`
- Zenodo replication package: https://zenodo.org/records/7644966  
- Repo: https://github.com/SageSELab/AidUI (MIT)

### License
**MIT** (AidUI GitHub repository license). Dataset released openly for research / future DP detection work.

### Number of samples (verified locally)
- **501 screenshots total**
  - **339** mobile
  - **162** web
- Paper: **301** dark-pattern instances + **243** non-DP instances across screenshots (localized bounding boxes)

### Labels
COCO-style localization in `bounding-boxes.json`. Mobile categories observed:

| ID | Name |
|----|------|
| 0 | `NO DP` |
| 1 | `DEFAULT CHOICE` |
| 2 | `NAGGING` |
| 3 | `DISGUISED ADS` |
| 4 | `GAMIFICATION` |
| 5 | `ATTENTION DISTRACTION` |

Verified counts (mobile): **339** images, **361** annotations. Web modality uses the same package layout under `evaluation_dataset/web/`.

### File structure (as downloaded)

```text
raw/contextdp/
  AidUI-Evaluation-Dataset.tar.gz
  evaluation_dataset/
    mobile/
      images/                 # *.jpg screenshots
      bounding-boxes.json
    web/
      images/                 # *.png / *.jpg screenshots
      bounding-boxes.json
```

### Download status
**Downloaded and extracted** successfully (~126 MB archive).

---

## 3. B4E2 Dark Pattern Dataset

### Source
- Paper: *Building UI/UX Dataset for Dark Pattern Detection and YOLOv12x-based Real-Time Object Recognition Detection System* (arXiv:2512.18269)
- GitHub (LFS): https://github.com/B4E2/B4E2-DarkLens-YOLO-DataSet  
  (also referenced as `B4E2-DarkPattern-YOLO-DataSet` in the paper)
- Roboflow Universe: https://universe.roboflow.com/darkpattern-2rd6z/b4e2_darkpatterns_dataset-c1rv6  
- Kaggle (alternate host): https://www.kaggle.com/datasets/bobb4e2/b4e2-darkpattern-dataset

### License
- **CC BY 4.0** — recorded in the Roboflow export (`README.dataset.txt`, `data.yaml`) shipped via GitHub LFS
- Kaggle dataset page lists **CC BY-NC-SA 4.0** — if you use that mirror, non-commercial share-alike terms apply

### Number of samples (verified locally)
- **~1,962** images in the GitHub LFS / Roboflow YOLOv12 export (train/valid/test)
- Paper abstract also cites a larger **4,066**-screenshot collection; the public YOLO zip mirrored here is the **~1.96k** labeled export described in the GitHub README

### Labels (YOLO classes)
From `data.yaml`:

| ID | Name |
|----|------|
| 0 | `button` |
| 1 | `checkbox` |
| 2 | `input_field` |
| 3 | `popup` |
| 4 | `qr_code` |

### File structure (as downloaded)

```text
raw/b4e2/
  README.md                                      # upstream GitHub README
  B4E2_DarkPatterns_Dataset.v2i.yolov12.zip      # LFS object (~178 MB)
  extracted/
    data.yaml
    README.dataset.txt
    README.roboflow.txt
    train/   images/ + labels/
    valid/   images/ + labels/
    test/    images/ + labels/
```

### Download status
**Downloaded via Git LFS** and extracted.  
(Kaggle API was unavailable here — no `~/.kaggle/kaggle.json`.)

### Manual fallback (if LFS fails)
1. Install [Git LFS](https://git-lfs.com/), then:

```powershell
git lfs install
git clone https://github.com/B4E2/B4E2-DarkLens-YOLO-DataSet.git datasets/raw/b4e2
```

2. Or download from Roboflow Universe / Kaggle into `datasets/raw/b4e2/` without modifying files.

---

## 4. Hugging Face — Deceptive Patterns

Two related public datasets from the WIPI / UW–Madison *Automatically Detecting Online Deceptive Patterns* work (CCS 2025 / arXiv:2411.07441):

### 4a. `WIPI/deceptive_patterns_synthetic`
- URL: https://huggingface.co/datasets/WIPI/deceptive_patterns_synthetic  
- **License: MIT**  
- **Samples (verified):** train **10,711** + test **2** rows  
- **Fields:** `input` (UI element feature string), `output` (`label|pattern_type|reasoning`), `file_url`  
- **Labels:** e.g. `non-deceptive`, `forced-action`, `interface-interference`, `sneaking`, … with pattern subtypes and free-text reasoning  
- Modality: **text features** distilled from screenshots (not raw images in the parquet)

### 4b. `WIPI/deceptive_patterns_manual_annotations`
- URL: https://huggingface.co/datasets/WIPI/deceptive_patterns_manual_annotations  
- License: not explicitly tagged on the Hub card; treat as research data from the same project — prefer citing the paper; synthetic set is clearly MIT  
- **Samples (verified):** **1,161** rows  
- **Fields:** `input`, `output`, `site_name`

### File structure (as downloaded)

```text
raw/deceptive_patterns/
  synthetic/
    README.md
    train-00000-of-00001.parquet
    test-00000-of-00001.parquet
  manual_annotations/
    README.md
    data-00000-of-00001.parquet
```

### Download status
**Downloaded** (parquet files intact; row counts match Hub metadata).

### Manual re-download

```powershell
# requires: pip install huggingface_hub
huggingface-cli download WIPI/deceptive_patterns_synthetic --repo-type dataset --local-dir datasets/raw/deceptive_patterns/synthetic
huggingface-cli download WIPI/deceptive_patterns_manual_annotations --repo-type dataset --local-dir datasets/raw/deceptive_patterns/manual_annotations
```

---

## Citation notes (minimal)

When publishing, cite at least:

1. Gundelach & Herrmann — CookieScanner / ARES 2023 + Zenodo DOI  
2. Mansur et al. — AidUI / ContextDP, ICSE 2023  
3. B4E2 team — arXiv:2512.18269 + dataset repo  
4. Nayak et al. — Automatically Detecting Online Deceptive Patterns (CCS 2025) + HF datasets  

---

## Unified processed dataset

Built from ContextDP + B4E2 + Hugging Face (CookieScanner ignored).

```bash
python datasets/preprocess/build_unified.py --workers 2 --ocr-sources contextdp,b4e2
```

Outputs land in `datasets/unified/`:

| File | Description |
|------|-------------|
| `train.jsonl` / `val.jsonl` / `test.jsonl` | Unified samples |
| `label_map.json` | Binary / fine / ConsentShield label inventories |
| `ocr_cache.json` | Cached RapidOCR results |
| `preprocessing_report.md` | Full preprocessing report |

Each JSONL row includes `label_binary`, `label_fine`, `label_consentshield`, optional `image_path`, `ocr_text` / `text`, and source metadata.
