# Git and clone policy

The repository keeps source code, configuration, tests, documentation, the
vendored PaddleOCR source, and the deployable `best.pt` model files needed by
the web scanner.

The following are intentionally excluded from Git:

- `.env` and API keys/passwords
- Python virtual environments and caches
- scan output, logs, archives, and local `Images/`
- downloaded training datasets and generated training runs
- `epoch*.pt`, `last.pt`, and OCR pretraining data

Training entry points remain versionable under `model/train/`. After cloning,
install the runtime dependencies and start the app with:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python run_web.py
```

The runtime model files under `model/*/weights/best.pt` and the Thai/Lao
Tesseract language data under `tools/tesseract/tessdata/` (`tha.traineddata`,
`lao.traineddata`) are kept. A machine with Tesseract installed can use that
data as an optional fallback. Training
datasets are downloaded separately; set `ROBOFLOW_API_KEY` in the environment
before running a dataset downloader. Never commit `.env` or a real API key.
