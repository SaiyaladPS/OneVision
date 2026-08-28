# Daily scan output

All files produced by a scan are grouped below the configured output folder by
the local scan date:

```text
runs/scan/YYYYMMDD/
  <input>_annotated.jpg
  <input>_result.json
  full_vehicle/
  plate_crops/
  ocr/<province-code>-<plate-prefix>/
  yolo11_ocr_dataset/{thai,lao}/
  dataset_exports/
  debug/                 # only when CAR_SCAN_DEBUG=1
```

Confirmed plate archives use the shared filename format:

```text
<cctv-camera>-<province-code>-<plate-prefix>-<plate-number>-<YYYYMMDD>-<daily-sequence>.jpg
```

For example: `0-BKK-63-7998-20260827-028.jpg`. The daily sequence starts at
`001` and is calculated only from files in that date's `full_vehicle` folder.
The same stem is used for the vehicle image, plate crop, OCR-ready image,
character-annotated image, and OCR metadata.
