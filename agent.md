# OneVison / Car Scan — Agent Guide

เอกสารนี้สรุปโครงสร้างระบบและกติกาสำหรับ agent หรือนักพัฒนาที่ทำงานต่อในโปรเจกต์นี้

## ภาพรวม

ระบบนี้อ่านป้ายทะเบียนไทยและลาวจากรูปภาพ วิดีโอ กล้องเครื่อง และกล้อง CCTV/RTSP โดยใช้ FastAPI เป็นเว็บ backend, YOLO เป็น detector/character model, OCR เป็นตัวช่วยยืนยันข้อความ และ PostgreSQL/Prisma สำหรับเก็บประวัติผลสแกน

โมเดลหลัก:

1. `model/detect_license/weights/best.pt` ตรวจตำแหน่งป้าย
2. `model/thai_license_plate/weights/best.pt` อ่านอักษร/ตัวเลข/จังหวัดไทย
3. `model/lao_license_plate/weights/best.pt` อ่านอักษร/ตัวเลข/แขวงลาว
4. `model/type_car_license/iruvd_run1/weights/best.pt` จำแนกประเภทรถเมื่อ pipeline ต้องใช้

ห้ามนำ class ทุกตัวมาต่อเป็นทะเบียนโดยตรง เพราะ class อักษรนำหน้าและรหัสจังหวัดอาจมีตัวเลขปะปน ต้องใช้การอ่านแบบมีโครงสร้างและ validation ของ pipeline

## จุดเริ่มต้นและการรัน

`run_web.py` เป็น launcher เท่านั้น: ตรวจ environment แล้วเรียก `car_scan.web:main` จาก `src/` ไม่ควรใส่ logic การอ่านป้ายไว้ในไฟล์นี้

รันจาก source บน Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_web.py
```

เปิด `http://127.0.0.1:8000` แล้วล็อกอินด้วยค่าจาก `.env` ค่าเริ่มต้นคือ `admin` / `changeme` ควรเปลี่ยนก่อนใช้งานจริง

รันด้วย Docker:

```powershell
docker compose up --build
```

หลังติดตั้ง Prisma ครั้งแรก:

```powershell
.\.venv\Scripts\python.exe db\prisma_cli.py generate
.\.venv\Scripts\python.exe db\prisma_cli.py db push
```

## โครงสร้างโค้ดสำคัญ

```text
run_web.py                 launcher
src/car_scan/web.py        FastAPI routes, auth, jobs, SSE
src/car_scan/service.py    scan orchestration, ROI, temporal tracks, OCR refinement
src/car_scan/worker.py     shared CCTV/RTSP worker, camera lanes, live publishing
src/car_scan/pipeline.py   pipeline/config helpers
src/car_scan/config.py     Settings และ environment variables
src/car_scan/database.py   result normalization และ persistence mapping
src/car_scan/auth.py       users, sessions, permissions
scan.py                    detector, character models, OCR และ image utilities
src/car_scan/web_static/   frontend HTML/CSS/JavaScript
model/                     runtime weights และ training scripts
tools/                     dataset/export/audit utilities
tests/                     unittest suites
```

## ลำดับการสแกน

ภาพต้นฉบับ → detector หาป้าย → crop/padding → Thai/Lao character models → แยก prefix/number/province → เลือกประเทศจากคะแนนเชิงโครงสร้าง → validation → preprocess/OCR fusion → แสดงผลและบันทึก JSON/ฐานข้อมูล

วิดีโอและ CCTV ใช้ `ScanService.process_live_frame()` รับเฟรมที่ sample แล้วส่งเข้า `LiveScanSession` ซึ่งเก็บ spatial track ของป้ายแต่ละคัน จับคู่ด้วย IoU/ระยะศูนย์กลาง/ขนาดกรอบ/เวลาที่ห่างจาก observation ก่อนหน้า แล้วสะสม candidate หลายข้อความเพื่อ temporal voting

## กติกา ROI/CCTV

เมื่อป้ายเข้ามาใน ROI ระบบต้องอ่านซ้ำและเก็บผลตลอดช่วงที่ track ยังมีชีวิตอยู่ ไม่ควรใช้ผลจากเฟรมเดียวตัดสินถ้ายังมี observation ต่อได้

การแทนค่าผลเดิมต้องผ่าน `ScanService._should_replace_live_record()`:

- ผลใหม่ที่ confidence หรือ temporal quality ต่ำกว่าจะไม่ทับผลเดิม
- ผลใหม่ที่อ่านข้อความต่างกันต้องมี temporal quality, recognition confidence หรือหลักฐานซ้ำที่ดีกว่า
- ผลที่ดีกว่าเท่านั้นจึงอัปเดต `CameraLane.plates`, live UI และ persistence
- deferred OCR ต้องใช้กฎเดียวกัน และห้ามเขียน crop ใหม่ทับภาพเดิมก่อนผ่าน quality gate

`src/car_scan/worker.py` ส่ง quality fields เช่น `temporal_average_quality`, `recognition_confidence`, `detection_confidence`, `camera_occurrences` และ frame boundaries ไปยัง live layer เพื่อเปรียบเทียบผลได้จริง

ค่าที่เกี่ยวข้องใน `.env`:

```text
CAR_SCAN_CAMERA_FRAME_STRIDE=2
CAR_SCAN_CAMERA_MIN_CONFIRMATIONS=2
CAR_SCAN_CAMERA_INFER_MAX_DIMENSION=1920
CAR_SCAN_TEMPORAL_MIN_QUALITY=0.58
CAR_SCAN_FULL_ACCURACY_MODE=0
CAR_SCAN_COMPUTE=auto
CAR_SCAN_IP_CAMERA_URL=
CAR_SCAN_IP_CAMERA_URLS=
```

ตั้ง `CAR_SCAN_FULL_ACCURACY_MODE=1` เพื่ออ่านทุกเฟรมและทำ OCR เต็มรูปแบบ แต่จะใช้ CPU/GPU สูงขึ้นมาก

## ผลลัพธ์และการ export

ผลใหม่อยู่ใต้ `scan/data/YYYYMMDD/` หรือ `CAR_SCAN_OUTPUT_DIR`:

```text
thai/       หลักฐานป้ายไทยที่ยืนยันแล้ว
laos/       หลักฐานป้ายลาวที่ยืนยันแล้ว
json/       result manifests และ metadata
log/        logs/snapshots
video/      annotated video
```

สร้าง dataset สำหรับ train ด้วย:

```powershell
.\.venv\Scripts\python.exe tools\build_training_dataset.py
```

`PASS` เป็น pre-label ที่ต้องตรวจทานก่อนใช้ฝึกจริง ส่วน `REJECT` ต้อง annotate ใหม่ก่อนใช้เป็น ground truth

สำหรับ archive ป้ายแต่ละรายการ ระบบบันทึกภาพจริงเพียง 2 ไฟล์: full frame และ crop ที่ preprocess แล้วส่งเข้า OCR โดย `ocr_ready_image` เป็น alias ไปยังไฟล์ crop เดียวกันเพื่อรองรับ API/ข้อมูลเก่า

## การเทรนโมเดล

สคริปต์หลัก:

```text
model/train/detect_license-train/dataset/train.py
model/train/thai_license_plate-train/dataset/train.py
model/train/lao_license_plate-train/dataset/train.py
```

พฤติกรรมชื่อ run ของทั้งสามสคริปต์:

- ค่าเริ่มต้นค้นหา run ล่าสุดที่มี `weights/last.pt` แล้ว resume
- `--new-run` สร้างชื่อใหม่ต่อท้าย เช่น `thai_plate_detect`, `thai_plate_detect_1`, `thai_plate_detect_2`
- fresh run ต้องไม่เขียนทับ run เก่า
- `--additional-epochs N` ใช้ต่อ training จาก checkpoint ของ Thai

ตัวอย่าง:

```powershell
cd model\train\thai_license_plate-train\dataset
python train.py --dataset all --epochs 100
python train.py --dataset all --epochs 100 --new-run
```

ถ้ามี label cache ว่างหรือเสีย ให้ย้ายเป็น `.cache.corrupt` และให้ Ultralytics สร้างใหม่ ห้ามลบ dataset/weights โดยไม่จำเป็น

## ฐานข้อมูล ความปลอดภัย และสิทธิ์

- `src/car_scan/auth.py` แบ่งสิทธิ์เป็น `admin`, `operator`, `viewer`
- `src/car_scan/database.py` แปลงผล scanner เป็นข้อมูลสำหรับ Prisma
- `prisma/schema.prisma` เป็น schema หลักของ PostgreSQL
- เก็บ password, RTSP user/password และ API keys ใน `.env` เท่านั้น ห้าม commit
- URL กล้องต้องถูก redact ก่อน log หรือส่งออกสู่ client

## การตรวจสอบก่อนส่งมอบ

ใช้คำสั่งจาก root:

```powershell
python -m unittest discover -s tests -q
python -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in Path('src').rglob('*.py')]; print('Syntax OK')"
git diff --check
```

ถ้า environment ไม่มี `fastapi`, `prisma`, OCR backend หรือ dependency ของกล้อง ให้รายงานเป็นข้อจำกัดของ environment และอย่าปลอมผลการทดสอบ

## กติกาสำหรับ agent

1. อ่านโค้ดและ tests ที่เกี่ยวข้องก่อนแก้ โดยเฉพาะ `service.py`, `worker.py`, `scan.py` และ `tests/test_temporal_scanning.py`
2. เปลี่ยน logic ในชั้นที่รับผิดชอบจริง ไม่ใส่ workaround ใน `run_web.py`
3. รักษา backward compatibility ของ result JSON และ API fields
4. ไม่ลบหรือ reset ไฟล์ของผู้ใช้/weights/dataset เพื่อแก้ปัญหาเฉพาะหน้า
5. เมื่อแก้ temporal recognition ต้องทดสอบทั้งกรณีผลใหม่ดีกว่าและผลใหม่แย่กว่า
6. ตรวจ `git status` และ `git diff --check` ก่อนส่งมอบ และระบุไฟล์ที่เปลี่ยนให้ชัดเจน
