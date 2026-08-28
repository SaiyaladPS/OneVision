# Car Scan Desktop (PyQt6 + PostgreSQL)

แอปเดสก์ท็อปใช้ PyQt6 ครอบ pipeline เดิมใน `scan.py` สำหรับตรวจป้ายทะเบียนไทย/ลาว ผลลัพธ์จะถูกบันทึกเป็น JSON และบันทึกลง PostgreSQL เมื่อกำหนด connection string

## รันจาก source

ต้องใช้ Python 3.11 ขึ้นไป และควรสร้าง virtual environment แยกจาก environment ที่ใช้ train model

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:CAR_SCAN_DATABASE_URL = "postgresql://car_scan:password@localhost:5432/car_scan"
python run_gui.py
```

ถ้าใช้ source checkout สามารถรัน `python run_gui.py` ได้โดยตรง เพราะ entrypoint จะเพิ่ม `src/` ให้อัตโนมัติ ไม่จำเป็นต้องติดตั้ง package ก่อน

ถ้าไม่กำหนด `CAR_SCAN_DATABASE_URL` โปรแกรมยังสแกนและเขียน JSON ได้ แต่จะไม่บันทึกฐานข้อมูล

## สแกนวิดีโอ

ใน GUI กด **เลือกภาพหรือวิดีโอ** แล้วเลือก `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv` หรือ `.webm` จากนั้นกด **เริ่มสแกน** โปรแกรมจะแสดง preview ระหว่างทำงาน และบันทึกวิดีโอที่ตีกรอบป้ายเป็น `<ชื่อวิดีโอ>_annotated.mp4` พร้อม JSON สรุปและภาพ crop ที่ดีที่สุดของแต่ละทะเบียน

ค่าเริ่มต้น `CAR_SCAN_VIDEO_FRAME_STRIDE=1` จะตรวจทุกเฟรมเพื่อความแม่นยำสูงสุด หากวิดีโอยาวและต้องการลดเวลา สามารถเพิ่มเป็น `2` หรือ `3` ได้

## การจัดเก็บภาพที่ยืนยันแล้ว

ภาพที่ยืนยันทะเบียนแล้วจะถูกแยกไว้ใต้โฟลเดอร์ผลลัพธ์เป็น `full_vehicle/` (ภาพรถเต็ม) และ `plate_crops/` (ภาพ crop ป้าย) โดยใช้ชื่อไฟล์ร่วมกันในรูปแบบ:

`เลขกล้อง-จังหวัดหรือแขวง-คำนำหน้า-เลขป้าย-YYYYMMDD-ลำดับสามหลัก.jpg`

ตัวอย่าง `0-กรุงเทพมหานคร-กข-1234-20260825-001.jpg` โดยลำดับจะเริ่มใหม่ตามวัน ตั้งค่าเลขกล้องเริ่มต้นได้ด้วย `CAR_SCAN_ARCHIVE_CAMERA_ID` สำหรับภาพหรือวิดีโอที่ไม่ได้มาจากกล้องสด

ไฟล์ OCR จะถูกจัดกลุ่มเพิ่มที่ `ocr/จังหวัด-คำนำหน้าป้าย/` และระบบจะสร้าง dataset สำหรับ YOLO11 แยกที่ `yolo11_ocr_dataset/thai/` และ `yolo11_ocr_dataset/lao/` โดยแต่ละประเทศมีเพียง `images/`, `labels/` และ `data.yml` (มี `data.yaml` สำรองสำหรับ Roboflow) ไม่มีโฟลเดอร์ `train/` หรือ `val/` ซ้อน ภาพ crop และภาพรถเต็มจะถูกรวมไว้ใน `images/` เดียวกัน เช่น `0-กรุงเทพมหานคร-67-5537-20260825-001.jpg` และ `0-กรุงเทพมหานคร-67-5537-20260825-001_full_vehicle.jpg` พร้อม label คู่กัน โดย label ของภาพรถเต็มถูกแปลงพิกัดจาก crop ให้แล้ว

ระบบสร้างไฟล์ส่งออกเพิ่มที่ `dataset_exports/`: `roboflow_yolo11_<ประเทศ>.zip` สำหรับอัปโหลดเป็น YOLO dataset ใน Roboflow และ `cvat_yolo11_<ประเทศ>.zip` สำหรับ CVAT โดยเลือก importer **YOLO 1.1** ไฟล์จะถูกสร้างใหม่ทุกครั้งที่มีตัวอย่างเพิ่ม

## PostgreSQL

สร้าง database/user ตัวอย่าง:

```sql
CREATE USER car_scan WITH PASSWORD 'change-me';
CREATE DATABASE car_scan OWNER car_scan;
```

เมื่อเปิดใช้ครั้งแรก แอปจะสร้างตาราง `scan_runs` และ `plates` ให้อัตโนมัติ หรือใช้ไฟล์ `db/schema.sql` ด้วย migration tool ขององค์กรก็ได้

หากต้องการสร้าง schema และนำเข้าผล JSON เดิมในโฟลเดอร์ `runs/` เข้า PostgreSQL:

```powershell
.\.venv\Scripts\python.exe db\initialize_database.py
```

คำสั่งนี้ทำงานแบบปลอดภัยเมื่อรันซ้ำ โดยจะนำเข้าผลเดิมเฉพาะตอนที่ฐานข้อมูลยังไม่มี scan

## สร้างไฟล์ติดตั้ง Windows

```powershell
pip install -e ".[build]"
.\build_windows.ps1
```

จะได้ portable build ที่ `dist\CarScan\CarScan.exe` หากติดตั้ง Inno Setup และมี `ISCC.exe` ใน PATH สคริปต์จะสร้าง `installer\output\CarScan-Setup.exe` เพิ่มให้ด้วย

โมเดล deployable ใน `model/` และ Lao OCR data ใน `tools/tesseract/tessdata/` ถูกใส่เข้า one-folder build โดยอัตโนมัติ ส่วน source สำหรับฝึกและข้อมูลที่สร้างระหว่างฝึกจะไม่ถูก bundle จึงควร build บนเครื่องที่มีไฟล์โมเดลครบ
