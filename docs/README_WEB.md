# Car Scan Web (FastAPI)

เว็บแอปใช้ pipeline `ScanService` / `scan.py` สำหรับอ่านป้ายทะเบียนไทยและลาว จากรูป วิดีโอ และกล้องในเบราว์เซอร์

## รันจาก source

ต้องใช้ Python 3.11 ขึ้นไป

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_web.py
```

เปิดเบราว์เซอร์ที่ [http://127.0.0.1:8000](http://127.0.0.1:8000)

เข้าสู่ระบบด้วยบัญชีเริ่มต้น `admin` / `changeme` ตั้งใน `.env`:

```
CAR_SCAN_ADMIN_USERNAME=admin
CAR_SCAN_ADMIN_PASSWORD=changeme
```

สิทธิ์มี 3 ระดับ: `admin` สแกนและจัดการผู้ใช้, `operator` สแกนและบันทึกผล, `viewer` เข้าดูหน้าเว็บอย่างเดียว

ตั้งค่าโฮสต์/พอร์ตใน `.env`:

```
CAR_SCAN_WEB_HOST=127.0.0.1
CAR_SCAN_WEB_PORT=8000
CAR_SCAN_WEB_MAX_UPLOAD_MB=4096
```

ค่าเริ่มต้นรับไฟล์ได้ถึง 4GB หากวิดีโอยาวกว่านั้นให้เพิ่ม `CAR_SCAN_WEB_MAX_UPLOAD_MB` แล้วรีสตาร์ทเซิร์ฟเวอร์

ติดตั้งแล้วสามารถใช้คอนโซลสคริปต์ `car-scan-web` แทน `python run_web.py` ได้

## รันด้วย Docker

ต้องมีไฟล์โมเดล runtime ในโปรเจกต์ (`model/*/weights/best.pt` และ `tools/tesseract/tessdata/lao.traineddata`, `tha.traineddata`) จากนั้นที่โฟลเดอร์โปรเจกต์:

```powershell
docker compose up --build
```

เปิดเบราว์เซอร์ที่ [http://127.0.0.1:8000](http://127.0.0.1:8000)

Compose จะสตาร์ทเฉพาะเว็บแอป และเชื่อมต่อ PostgreSQL ภายนอกผ่าน `DATABASE_URL` หรือ `CAR_SCAN_DATABASE_URL` ใน `.env` จากนั้น Prisma Client Python จะจัดการตาราง `scan_runs`, `plates`, `users` ผลสแกนถูกเขียนไว้ที่ `scan/data` บนเครื่องคุณ โดยไม่สร้าง PostgreSQL container เพิ่ม

เมื่อรันจาก source หลัง `pip install` ให้ generate client แล้วดัน schema ครั้งแรก:

```powershell
.\.venv\Scripts\python.exe db\prisma_cli.py generate
.\.venv\Scripts\python.exe db\prisma_cli.py db push
```

หรือใช้ `python db/initialize_database.py` ซึ่งจะ `db push` แล้วค่อยนำเข้า JSON จาก `runs/` ถ้าฐานยังว่าง

หยุดด้วย `docker compose down`

## สิ่งที่ทำได้บนเว็บ

- อัปโหลดรูป แล้วลากกรอบ ROI ก่อนสแกน
- อัปโหลดวิดีโอ ดูพรีวิวระหว่างสแกน และกดหยุดได้
- เปิดกล้องของเครื่องในเบราว์เซอร์ ถ่ายภาพหรืออัดคลิป แล้วส่งเข้า pipeline เดิม
- สลับภาษาไทย / ລາວ / English
- เปิด **รายการสแกน** ทางซ้าย เพื่อดูผลที่บันทึกแล้วตามวันที่ (เขตเวลากรุงเทพฯ) และผู้ประจำการสแกน ค้นหาทะเบียน เปิดภาพหลักฐาน และพิมพ์รายการวันนั้น
- เปลี่ยนรหัสผ่านของบัญชีตนเองจากปุ่ม **รหัสผ่าน**
- บันทึกผล JSON และเขียน PostgreSQL เมื่อตั้ง `CAR_SCAN_DATABASE_URL`

กล้องบนเว็บใช้กล้องของเครื่องที่เปิดเบราว์เซอร์

## API สั้นๆ

| เส้นทาง | คำอธิบาย |
| --- | --- |
| `GET /` | หน้าเว็บ (ต้องล็อกอิน) |
| `GET /login` | หน้าเข้าสู่ระบบ |
| `POST /api/auth/login` | เข้าสู่ระบบ |
| `POST /api/auth/logout` | ออกจากระบบ |
| `GET /api/auth/me` | ผู้ใช้ปัจจุบันและสิทธิ์ |
| `GET /api/users` | รายชื่อผู้ใช้ (admin) |
| `GET /api/health` | สถานะแอปและฐานข้อมูล |
| `GET /api/scans` | รายการสแกนตามวัน (`day=YYYY-MM-DD`) ผู้สแกน (`operator_id`) หรือค้นหาทะเบียน (`q`) |
| `GET /api/scans/{id}` | รายละเอียดรายการสแกน |
| `GET /api/scans/{id}/image` | ภาพหลักฐานของรายการสแกน |
| `POST /api/auth/password` | เปลี่ยนรหัสผ่านของบัญชีที่ล็อกอิน |
| `POST /api/jobs` | อัปโหลดไฟล์แล้วเริ่มสแกน (`file`, `media_type`, `roi`) |
| `GET /api/jobs/{id}` | สถานะงาน |
| `GET /api/jobs/{id}/events` | Server-Sent Events ระหว่างสแกน |
| `POST /api/jobs/{id}/stop` | ขอหยุดสแกนวิดีโอ |
| `GET /api/jobs/{id}/preview` | เฟรมพรีวิวล่าสุด |
