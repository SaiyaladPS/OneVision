# Modular vehicle and licence-plate pipeline

The scanner keeps vehicle type, country, plate type, preprocessing, OCR, and
validation as separate stages. Runtime mappings and routes are configured in
`pipeline_config.yaml`; model class names are never rewritten.

## Pipeline modes

- `auto` and `fast`: detect plates first. Vehicle inference is skipped when no
  plate is present.
- `full`: detect/classify vehicles before detecting and associating plates.

Set `CAR_SCAN_PIPELINE_MODE` or pass `--pipeline-mode` to `scan.py`.

## Models and class mappings

Vehicle classes are read from `model.names`, then mapped to `display_name` and
`normalized_class`. The original value is retained in `raw_class`. Plate type
classification is optional and returns `unknown` until
`CAR_SCAN_PLATE_TYPE_MODEL` or `--plate-type-model` points to real trained
weights. Add plate mappings only for class names present in those weights.

## Context routing

Routing keys use `country|vehicle_type|plate_type`, with `*` fallbacks. The
selected preprocessing profile is applied using values under
`preprocessing_parameters`. OCR currently routes to the real shared
`common_ocr`; specific OCR routes can be added when matching trained weights
exist.

Validation rules use the same context keys and may configure:

- `allowed_pattern`
- `allowed_prefix_pattern`
- `allowed_number_pattern`
- `allowed_provinces`
- `allowed_layouts` (`single_line` or `two_line`)

## Debugging

Set `CAR_SCAN_DEBUG=1` to save stage-oriented images and OCR metadata under
`scan/data/YYYYMMDD/debug/`. Debug output includes vehicle/plate annotations, crops,
contextual preprocessing, country and vehicle groups, OCR JSON, and rejected
results.

Each plate result contains the stable nested objects `vehicle`, `plate`, and
`processing`, while the previous flat fields remain available for existing GUI
and database integrations.
