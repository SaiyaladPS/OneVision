# ສະຫຼຸບລະບົບ OneVision

## 1. ພາບລວມ

OneVision ແມ່ນລະບົບສະແກນ ແລະ ອ່ານປ້າຍທະບຽນລົດໄທ–ລາວ. ລະບົບຫຼັກພັດທະນາດ້ວຍ Python ແລະ FastAPI, ຮອງຮັບການນຳເຂົ້າຮູບ, ວິດີໂອ ແລະ ພາບຈາກກ້ອງສົດ. ຜົນການສະແກນສາມາດບັນທຶກເປັນໄຟລ໌ JSON ແລະ PostgreSQL ເມື່ອຕັ້ງຄ່າຖານຂໍ້ມູນ.

> OneVision ແມ່ນລະບົບຕົ້ນທາງສຳລັບປະມວນຜົນການສະແກນ. ລະບົບລາຍງານ ຫຼື ລະບົບອື່ນອາດຈະດຶງເຫດການ/ຜົນຈາກ OneVision ແຍກຕ່າງຫາກ.

## 2. ຄວາມສາມາດຫຼັກ

- ອັບໂຫຼດຮູບເພື່ອສະແກນ ແລະ ກຳນົດເຂດສົນໃຈ (ROI) ກ່ອນປະມວນຜົນ.
- ອັບໂຫຼດວິດີໂອ, ເບິ່ງພາບຕົວຢ່າງຂະນະສະແກນ ແລະ ສັ່ງຢຸດວຽກໄດ້.
- ໃຊ້ກ້ອງຂອງອຸປະກອນຜ່ານ browser ເພື່ອຖ່າຍຮູບ ຫຼື ບັນທຶກວິດີໂອ ແລ້ວສົ່ງເຂົ້າ pipeline ດຽວກັນ.
- ຮອງຮັບພາສາໄທ, ລາວ ແລະ ອັງກິດ.
- ຄົ້ນຫາປະຫວັດການສະແກນຕາມວັນ, ຜູ້ປະຈຳການ ຫຼື ເລກທະບຽນ; ເປີດຮູບຫຼັກຖານ ແລະ ພິມລາຍການປະຈຳວັນໄດ້.
- ຈັດການບັນຊີຕາມສິດທິ `admin`, `operator` ແລະ `viewer`; ຜູ້ໃຊ້ສາມາດປ່ຽນລະຫັດຜ່ານຂອງຕົນເອງໄດ້.

## 3. ຂັ້ນຕອນການອ່ານປ້າຍ

```text
ຮູບ/ເຟຣມວິດີໂອ
  -> ກວດຫາກອບປ້າຍທະບຽນ
  -> crop ປ້າຍ ແລະ ເພີ່ມຂອບຮູບ
  -> ກວດຈັບຕົວອັກສອນ/ຕົວເລກດ້ວຍໂມເດວໄທ ຫຼື ລາວ
  -> ລວມ box ທີ່ຊ້ຳ ແລະ ຈັດລຳດັບຕາມຕຳແໜ່ງ
  -> ກວດຮູບແບບປ້າຍ ແລະ ປະເມີນປະເທດ
  -> ປັບຄຸນນະພາບຮູບ crop ແລະ ລວມຜົນ OCR
  -> ສະແດງ/ບັນທຶກຜົນ
```

ໂມເດວ `detect_license` ໃຊ້ຫາຕຳແໜ່ງປ້າຍ. ໂມເດວ `thai_license_plate` ແລະ `lao_license_plate` ຊ່ວຍກວດລັກສະນະປ້າຍຂອງແຕ່ລະປະເທດ. OCR ເປັນຕົວຊ່ວຍຢືນຢັນ/ເຕີມຂໍ້ມູນ; ບໍ່ຄວນເອົາ class ທຸກອັນມາຕໍ່ເປັນຂໍ້ຄວາມໂດຍກົງ ເພາະລະຫັດປະເພດ ຫຼື ແຂວງອາດມີຕົວເລກປົນ.

ສຳລັບວິດີໂອ ລະບົບບໍ່ຈຳເປັນຕ້ອງຮັນ inference ທຸກເຟຣມ: ຈະສຸ່ມກວດເຟຣມ, ຕິດຕາມປ້າຍທີ່ພົບ ແລະ ລວມຄະແນນ/ຜົນອ່ານຊ້ຳ. ຈະເລືອກ crop ທີ່ຊັດ ແລະ ຜົນທີ່ໜ້າເຊື່ອຖືກວ່າ.

## 4. ຜົນສະແກນ ແລະ ບ່ອນເກັບຂໍ້ມູນ

ໂດຍປົກກະຕິໄຟລ໌ຈະຖືກຈັດຕາມວັນທີໃນ `scan/data/YYYYMMDD/` (ຫຼື ໂຟນເດີທີ່ລະບຸໃນ `CAR_SCAN_OUTPUT_DIR`):

```text
scan/data/YYYYMMDD/
  thai/     ຫຼັກຖານປ້າຍໄທ
  laos/     ຫຼັກຖານປ້າຍລາວ ແລະ ຮູບ crop
  json/     ຜົນສະແກນ ແລະ metadata
  log/      log ແລະ ຮູບທີ່ມີການວາດຜົນ
  video/    ວິດີໂອທີ່ປະມວນຜົນ
```

ເມື່ອຕັ້ງ PostgreSQL, Prisma ຈັດການຕາຕະລາງຫຼັກ `scan_runs`, `plates` ແລະ `users`. ຄວນສຳຮອງທັງຖານຂໍ້ມູນ ແລະ ໂຟນເດີຮູບຫຼັກຖານ.

## 5. ການຕິດຕັ້ງ ແລະ ເປີດລະບົບ

ລະບົບຕ້ອງການ Python 3.11 ຂຶ້ນໄປ. ຈາກ PowerShell ໃນໂຟນເດີ OneVision:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_web.py
```

ເປີດ `http://127.0.0.1:8000`. ສາມາດໃຊ້ Docker ໄດ້ດ້ວຍ `docker compose up --build` ຫຼັງຈາກຈັດວາງໄຟລ໌ໂມເດວ ແລະ ຕັ້ງຄ່າ `.env` ຄົບ. Docker Compose ຂອງໂຄງການເຊື່ອມ PostgreSQL ພາຍນອກ; ບໍ່ໄດ້ເປີດ PostgreSQL container ໃຫ້ເອງ.

## 6. ຄ່າຕັ້ງຄ່າສຳຄັນ

ໃຫ້ສ້າງ `.env` ຈາກຕົວຢ່າງ `.env.example` ແລະ ໃສ່ຄ່າຕາມ environment ຂອງແຕ່ລະເຄື່ອງ. ຢ່າ commit ລະຫັດຜ່ານ, token ຫຼື secret ຈິງເຂົ້າ Git.

| ຕົວແປ | ໜ້າທີ່ |
| --- | --- |
| `DATABASE_URL` / `CAR_SCAN_DATABASE_URL` | ທີ່ຢູ່ PostgreSQL |
| `CAR_SCAN_OUTPUT_DIR` | ໂຟນເດີບັນທຶກຜົນ ແລະ ຮູບຫຼັກຖານ |
| `CAR_SCAN_WEB_HOST`, `CAR_SCAN_WEB_PORT` | ທີ່ຢູ່ ແລະ port ຂອງ Web app |
| `CAR_SCAN_WEB_MAX_UPLOAD_MB` | ຂະໜາດ upload ສູງສຸດ |
| `CAR_SCAN_ADMIN_USERNAME`, `CAR_SCAN_ADMIN_PASSWORD` | ບັນຊີ admin ເລີ່ມຕົ້ນ; ຄວນປ່ຽນຄ່າເລີ່ມຕົ້ນ |
| `CAR_SCAN_DETECTOR_CONFIDENCE`, `CAR_SCAN_CHARACTER_CONFIDENCE` | ເກນຄວາມໝັ້ນໃຈຂອງການກວດຈັບ |
| `CAR_SCAN_VIDEO_*`, `CAR_SCAN_CAMERA_*` | ຄວາມຖີ່ກວດເຟຣມ ແລະ ຈຳນວນການຢືນຢັນຜົນ |
| `CAR_SCAN_COMPUTE` | ເລືອກວິທີຄຳນວນ `auto`, `gpu`, `cpu` ຫຼື `hybrid` |
| `CAR_SCAN_PIPELINE_CONFIG` | ທີ່ຢູ່ໄຟລ໌ປັບ pipeline |

ຊື່ environment ທີ່ໃຊ້ຈິງໃຫ້ກວດຈາກ `.env.example` ຂອງ repository; ຢ່າຄັດລອກຄ່າລັບຈາກ `.env` ໄປໃສ່ເອກະສານ ຫຼື log.

## 7. API ທີ່ໃຊ້ບ່ອຍ

| Endpoint | ໜ້າທີ່ |
| --- | --- |
| `GET /api/health` | ກວດສະຖານະ Web app ແລະ database |
| `POST /api/auth/login`, `POST /api/auth/logout` | ເຂົ້າ/ອອກຈາກລະບົບ |
| `GET /api/scans` | ຄົ້ນຫາປະຫວັດການສະແກນ |
| `GET /api/scans/{id}` | ເບິ່ງລາຍລະອຽດຜົນສະແກນ |
| `GET /api/scans/{id}/image` | ເປີດຮູບຫຼັກຖານ |
| `POST /api/jobs` | ອັບໂຫຼດ media ແລະ ເລີ່ມສະແກນ |
| `GET /api/jobs/{id}` | ກວດສະຖານະວຽກ |
| `GET /api/jobs/{id}/events` | ຮັບຄວາມຄືບໜ້າຜ່ານ Server-Sent Events |
| `POST /api/jobs/{id}/stop` | ຂໍຢຸດວຽກວິດີໂອ |

ເສັ້ນທາງ API ຂອງກ້ອງ, worker ແລະ real-time ອາດມີຄ່າຕັ້ງຄ່າ/ການຢືນຢັນສິດທິເພີ່ມເຕີມ; ກວດຈາກ source ແລະ config ຂອງ deployment ກ່ອນນຳໄປເປີດໃຊ້ຈາກພາຍນອກ.

## 8. ການສົ່ງອອກຊຸດຂໍ້ມູນເພື່ອຝຶກໂມເດວ

ໃຊ້ `tools/build_training_dataset.py` ເພື່ອສ້າງຊຸດ CVAT ຫຼື Roboflow. ໂຟນເດີ `PASS` ແມ່ນຜົນຄາດເດົາ/label ເບື້ອງຕົ້ນຈາກ scanner ເທົ່ານັ້ນ—ຕ້ອງໃຫ້ຄົນກວດແກ້ກ່ອນນຳໄປຝຶກ. ລາຍການ `REJECT` ຕ້ອງໃຫ້ຄົນໃສ່ annotation ດ້ວຍຕົນເອງ.

```powershell
python tools/build_training_dataset.py --format cvat --dates 20260802-20260803
```

## 9. ການແກ້ບັນຫາເບື້ອງຕົ້ນ

- ເຂົ້າ Web ບໍ່ໄດ້: ກວດວ່າ process ຍັງຮັນ, host/port ຖືກຕ້ອງ ແລະ firewall ອະນຸຍາດ.
- ຖານຂໍ້ມູນຕໍ່ບໍ່ໄດ້: ກວດ `DATABASE_URL`/`CAR_SCAN_DATABASE_URL`, ສະຖານະ PostgreSQL, network route ແລະສິດຂອງບັນຊີ DB.
- ບໍ່ເຫັນກ້ອງ: ກວດສິດກ້ອງໃນ browser, ກວດວ່າກ້ອງບໍ່ຖືກໂປຣແກຣມອື່ນໃຊ້ ແລະ ເລືອກອຸປະກອນຖືກຕົວ.
- ສະແກນຊ້າ ຫຼື ຜິດພາດ: ກວດວ່າ model weights ແລະໄຟລ໌ພາສາ OCR ຄົບ, ກວດ CPU/GPU ແລະຄ່າ confidence ໃນ `.env`.
- ຮູບຫຼັກຖານຫາບໍ່ເຫັນ: ກວດ `CAR_SCAN_OUTPUT_DIR`, ສິດອ່ານ/ຂຽນໂຟນເດີ, ແລະວ່າ path ຂອງ storage ຖືກ mount ເມື່ອໃຊ້ Docker.

## 10. ໂຄງສ້າງ source ໂດຍຫຍໍ້

- `src/car_scan/web.py` — FastAPI web app ແລະ API routes.
- `src/car_scan/pipeline.py`, `service.py`, `scan.py` — ຂັ້ນຕອນປະມວນຜົນຮູບ/ວິດີໂອ.
- `src/car_scan/worker.py` — ວຽກ background ແລະ ການປະມວນຜົນກ້ອງ.
- `src/car_scan/realtime.py` — ສົ່ງເຫດການ real-time ໃຫ້ຜູ້ຕິດຕາມ.
- `src/car_scan/database.py`, `prisma_db.py`, `db/` — ການເຊື່ອມຕໍ່ ແລະ schema ຖານຂໍ້ມູນ.
- `model/` — ໄຟລ໌ໂມເດວກວດຫາປ້າຍ ແລະ ອ່ານລັກສະນະປ້າຍ.
- `scan/` — ຂໍ້ມູນສະແກນ ແລະ ຫຼັກຖານທີ່ສ້າງຂຶ້ນ.
- `tools/` — ເຄື່ອງມືຈັດຊຸດຂໍ້ມູນ, ກວດ label ແລະຊ່ວຍຝຶກ/ນຳເຂົ້າ annotation.

## ເອກະສານອ້າງອີງໃນ repository

- `docs/README_WEB.md`
- `docs/OUTPUT_LAYOUT.md`
- `docs/RECOGNITION_PIPELINE.md`
- `.env.example` (ຊື່ຕົວແປເທົ່ານັ້ນ; ຢ່າເປີດເຜີຍ secret ຈິງ)
