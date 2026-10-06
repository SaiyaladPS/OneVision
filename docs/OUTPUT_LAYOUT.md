# Scan archive and training export layout

New scans are stored under `scan/data/YYYYMMDD/` (or the directory set by
`CAR_SCAN_OUTPUT_DIR`). The same layout is used for image uploads, video
uploads, and live cameras.

```text
scan/data/YYYYMMDD/
  thai/                  # confirmed Thai plate evidence
  laos/                  # confirmed Lao plate evidence
    <stem>-full_vehicle.webp
    <stem>-plate_crops.webp
  json/                  # result manifests and plate metadata
  log/                   # detection logs and annotated snapshots
  video/                 # annotated video uploads
```

The `ocr_ready_image` field remains an API/database alias to the plate crop;
no separate OCR image file is written.

`<stem>` is `<camera>-<province>-<prefix>-<number>-<YYYYMMDD>-<sequence>`.

## Create a training export

Run the script without options and it asks for CVAT/Roboflow and the desired
date selection:

```powershell
.\.venv\Scripts\python.exe tools\build_training_dataset.py
```

```powershell
# A whole date range
.\.venv\Scripts\python.exe tools\build_training_dataset.py --format cvat --dates 20260802-20260803

# Specific separate dates
.\.venv\Scripts\python.exe tools\build_training_dataset.py --format roboflow --dates 20260802,20260805
```

The results are written as `cvat/cvat-20260802-20260803/` or
`roboflow/roboflow-20260802-20260803/`. Both contain Thai, Lao, detector, and
OCR sets with `PASS` and `REJECT` folders. `PASS` is a scanner pre-label and
must be reviewed before training; `REJECT` must be manually annotated before
using it as OCR truth.
