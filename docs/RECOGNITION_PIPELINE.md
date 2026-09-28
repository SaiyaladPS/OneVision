# ระบบอ่านป้ายทะเบียนไทย–ลาว

## หน้าที่ของโมเดล

1. `detect_license` ตรวจหาตำแหน่งป้ายจากภาพหรือเฟรมวิดีโอเท่านั้น
2. `thai_license_plate` ตรวจเลข 6 หลัก, class อักษรไทยแบบ `Axx` และรหัสจังหวัด
3. `lao_license_plate` ตรวจเลข 4 หลัก, class อักษรลาวที่เก็บเป็น Latin alias และรหัสแขวง
4. OCR ทำงานเฉพาะภาพที่ crop จากกรอบป้ายแล้ว และใช้เป็นตัวช่วยยืนยัน/เติมข้อมูล ไม่ใช้ตัดสินประเทศเพียงลำพัง

## Pipeline สำหรับภาพ

```text
ภาพต้นฉบับ
  -> detect_license
  -> crop + padding
  -> Thai/Lao character models
  -> ลบ box ซ้ำและจัด baseline
  -> แยก digit / prefix / province
  -> ตรวจรูปแบบ (ไทย 2+4, ลาว prefix+4)
  -> เลือกประเทศจากคะแนนเชิงโครงสร้าง
  -> preprocess crop (resize, denoise, deskew, CLAHE, sharpen)
  -> OCR fusion
  -> GUI + PostgreSQL
```

ห้ามนำ class ทุกตัวมาต่อเป็นข้อความโดยตรง เพราะ `A23` และรหัสจังหวัดมีตัวเลข/ตัวอักษรที่ไม่ใช่เลขทะเบียน

## Pipeline สำหรับวิดีโอ

ระหว่างอ่านเฟรมใช้ detector และ structured YOLO เท่านั้น จากนั้นติดตามป้ายด้วยตำแหน่งและสะสมคะแนนของทะเบียนที่อ่านได้ครบ ระบบเลือกข้อความที่มีผลโหวตสูงสุดและ crop ที่คม/มั่นใจที่สุดของแต่ละ track แล้วจึงเรียก OCR เต็มรูปแบบหนึ่งครั้ง วิธีนี้ลดการค้างของตัวเล่นวิดีโอและลดผลลัพธ์ซ้ำจากป้ายเดียวกัน

คะแนนเฟรมประกอบด้วยความมั่นใจของเลข, prefix ลาว, detector, ระยะห่างคะแนนประเทศ และความครบของรูปแบบทะเบียน

ระบบคำนวณ sampling stride จาก FPS ของวิดีโอ โดยตั้งเป้าเริ่มต้นประมาณ 4 ครั้งต่อวินาที จึงไม่สั่ง inference ทุกเฟรมของวิดีโอ 30/60 FPS แต่ยังมีหลาย observation สำหรับโหวต ผลวิดีโอต้องพบข้อความเดิมอย่างน้อย 2 ครั้งก่อนเผยแพร่

## Pipeline สำหรับกล้องสด

GUI เลือกกล้องหมายเลข 0–2 ได้ กล้องทำงานใน worker thread และเปิดผ่าน DirectShow บน Windows เพื่อลดปัญหา backend/permission เฟรม preview ถูกจำกัดอัตราส่งเพื่อไม่ให้ Qt event queue ค้าง

ผลกล้องจะถูกส่งเข้าตาราง GUI เมื่อทะเบียนเดิมมีรูปแบบครบ, ผ่านคะแนนคุณภาพ และพบซ้ำตามค่า `CAR_SCAN_CAMERA_MIN_CONFIRMATIONS` (ค่าเริ่มต้น 2 ครั้ง) ผู้ใช้เลือกให้หยุดอัตโนมัติเมื่อพบทะเบียนแรก หรือสแกนต่อเนื่องและกดหยุดเองได้

ค่าที่ปรับได้จาก `.env` ได้แก่ `CAR_SCAN_VIDEO_TARGET_SCANS_PER_SECOND`, `CAR_SCAN_VIDEO_MIN_CONFIRMATIONS`, `CAR_SCAN_CAMERA_FRAME_STRIDE`, `CAR_SCAN_CAMERA_MIN_CONFIRMATIONS` และ `CAR_SCAN_TEMPORAL_MIN_QUALITY`

## แนวทางเพิ่มความแม่นยำในรอบฝึกโมเดล

- เพิ่มภาพไทยที่แสงน้อย ภาพขาวซีด motion blur และมุมเอียง เพราะเป็นกลุ่มที่ character model พลาดมากที่สุด
- เพิ่มและ balance class prefix ลาวที่มีตัวอย่างน้อย โดยเฉพาะ class ที่สับสนกับ `N` (ບ)
- เก็บ `data.yaml` หรือ class mapping ไว้กับ weight ทุกครั้ง เพื่อไม่ต้องอนุมาน alias จากข้อมูลภายหลัง
- แบ่ง train/validation/test ตามรถหรือคลิป ไม่แบ่งตามเฟรม เพื่อป้องกันภาพเกือบเหมือนกันรั่วไปอยู่คนละชุด
- วัดแยกเป็น plate exact match, digit exact match, country accuracy และ video track accuracy ไม่ใช้ mAP อย่างเดียว
