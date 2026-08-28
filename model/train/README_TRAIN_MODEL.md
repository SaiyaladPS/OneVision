# คู่มือ Train โมเดลตรวจจับป้ายทะเบียน

เอกสารนี้อธิบายการ train โมเดลตรวจจับป้ายทะเบียนต่อจากโมเดลเดิม โดยใช้
`model/detect_license/weights/best.pt` เป็นน้ำหนักเริ่มต้น และใช้ dataset จาก
Roboflow ที่ดาวน์โหลดไว้ใน `model/detect_license/dataset/Lpr-2/`

## สิ่งที่ต้องมี

โครงสร้างไฟล์หลักควรเป็นดังนี้:

```text
model/detect_license/
├── weights/
│   └── best.pt
└── dataset/
    ├── train.py
    └── Lpr-2/
        ├── data.yaml
        ├── train/images/
        ├── train/labels/
        ├── valid/images/
        ├── valid/labels/
        ├── test/images/
        └── test/labels/
```

ใน `Lpr-2/data.yaml` ต้องเป็น dataset แบบ 1 class และชื่อ class ต้องตรงกับ
โมเดลเดิม:

```yaml
nc: 1
names: ['License_Plate']
```

## เตรียม environment

เปิด PowerShell ที่โฟลเดอร์ root ของโปรเจกต์ แล้ว activate virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

ตรวจสอบว่าสคริปต์ใช้งานได้:

```powershell
python model/detect_license/dataset/train.py --help
```

## Train ต่อด้วย Lpr-2

คำสั่งแนะนำ:

```powershell
python model/detect_license/dataset/train.py `
  --dataset Lpr-2 `
  --epochs 50 `
  --batch 16 `
  --imgsz 640
```

ค่าเริ่มต้นของสคริปต์มีดังนี้:

| Option | ค่าเริ่มต้น | ความหมาย |
|---|---:|---|
| `--dataset` | `Lpr-2` | dataset ที่ใช้ train |
| `--epochs` | `50` | จำนวนรอบ train สูงสุด |
| `--batch` | `16` | จำนวนรูปต่อ batch |
| `--imgsz` | `640` | ขนาดรูปที่ใช้ train |
| `--workers` | `0` | จำนวน worker สำหรับโหลดข้อมูล |
| `--device` | ตรวจอัตโนมัติ | GPU เช่น `0` หรือ CPU เช่น `cpu` |
| `--garget` | เลือก model |

เลือกโมเดลด้วย --target ได้แล้ว:
- detect_license — รูปรถเต็มคัน
- lao_license_plate — รูปป้ายลาวที่ crop แล้ว
- thai_license_plate — รูปป้ายไทยที่ crop แล้ว
- type_car_license — รูปรถเต็มคัน

สคริปต์จะโหลด `best.pt` เดิมแล้ว fine-tune รอบใหม่ด้วย learning rate ต่ำ
เพื่อรักษาความสามารถเดิมของโมเดลไว้ การ train นี้เป็นรอบใหม่ จึงใช้
`resume=False` และไม่ได้กู้ optimizer state จากงานเดิม

## เลือกอุปกรณ์ train

ใช้ GPU ตัวที่ 0:

```powershell
python model/detect_license/dataset/train.py --dataset Lpr-2 --device 0
```

บังคับใช้ CPU และลด batch เมื่อหน่วยความจำไม่พอ:

```powershell
python model/detect_license/dataset/train.py `
  --dataset Lpr-2 `
  --device cpu `
  --batch 4 `
  --workers 0
```

ถ้าไม่ระบุ `--device` สคริปต์จะเลือก XPU, CUDA GPU หรือ CPU ตามที่ตรวจพบ

## Train โดยรวม dataset ทั้งหมด

ถ้าต้องการให้โมเดลเรียนรู้ทั้ง `Lpr-2` และ dataset อื่นที่อยู่ข้าง ๆ และมี
`data.yaml` ให้ใช้:

```powershell
python model/detect_license/dataset/train.py `
  --dataset all `
  --epochs 50 `
  --batch 16
```

การรวม dataset ช่วยลดโอกาสที่โมเดลจะลืมรูปแบบจาก dataset เดิม แต่จะใช้เวลา
และพื้นที่ในการ train มากขึ้น

## ไฟล์ `combined_data.generated.yaml`

ไฟล์นี้เป็น manifest ของ Ultralytics YOLO ที่ `train.py` สร้างให้อัตโนมัติ
ทุกครั้งก่อน train ใช้บอกตำแหน่งของ `train`, `val` และ `test` รวมถึงจำนวน
class และชื่อ class

เมื่อใช้ `--dataset Lpr-2` ไฟล์นี้จะชี้ไปที่ `Lpr-2` เท่านั้น เมื่อใช้
`--dataset all` สคริปต์จะเขียนไฟล์เดิมใหม่ให้มี path ของทุก dataset ที่เลือก
จึงไม่ต้องแก้ไฟล์นี้ด้วยตนเอง

## ผลลัพธ์หลัง train

ผลลัพธ์จะถูกบันทึกใต้:

```text
model/detect_license/runs/detect_license_lpr2_finetune/
├── args.yaml
├── results.csv
└── weights/
    ├── best.pt
    └── last.pt
```

- `best.pt`: checkpoint ที่มีผล validation ดีที่สุด ใช้สำหรับนำไปใช้งานจริง
- `last.pt`: checkpoint จาก epoch สุดท้าย
- `results.csv`: loss และ metric ของแต่ละ epoch
- `args.yaml`: configuration ที่ใช้ในรอบ train นั้น

สคริปต์ตั้ง `exist_ok=False` ดังนั้นการ train ซ้ำจะไม่ทับผลลัพธ์เดิม โดย
Ultralytics จะสร้างชื่อโฟลเดอร์ใหม่ เช่น `detect_license_lpr2_finetune2`

## ตรวจสอบโมเดลหลัง train

ตรวจสอบผลบน test split ของ dataset ที่ train ล่าสุด:

```powershell
python -c "from ultralytics import YOLO; model = YOLO('model/detect_license/runs/detect_license_lpr2_finetune/weights/best.pt'); model.val(data='model/detect_license/dataset/combined_data.generated.yaml', split='test')"
```

ผล metric ที่ควรดู ได้แก่ `mAP50`, `mAP50-95`, `precision` และ `recall`
ควรเปรียบเทียบกับโมเดลเดิมก่อนนำโมเดลใหม่ไปใช้งานจริง

## นำโมเดลใหม่ไปใช้งาน

ให้ใช้ไฟล์ `best.pt` จาก run ที่ได้ metric ดีที่สุด:

```text
model/detect_license/runs/detect_license_lpr2_finetune/weights/best.pt
```

ควรเก็บ `model/detect_license/weights/best.pt` เดิมไว้ก่อน แล้วเปลี่ยน path
ของโมดูลตรวจจับให้ชี้ไปยัง checkpoint ใหม่เมื่อตรวจสอบแล้วว่าผลดีขึ้นจริง

## ปัญหาที่พบบ่อย

### หา `best.pt` ไม่พบ

ตรวจสอบว่ามีไฟล์นี้อยู่จริง:

```powershell
Test-Path model/detect_license/weights/best.pt
```

### หา dataset ไม่พบ

ตรวจสอบว่า `Lpr-2/data.yaml` อยู่ในตำแหน่งที่ถูกต้อง:

```powershell
Test-Path model/detect_license/dataset/Lpr-2/data.yaml
```

### หน่วยความจำไม่พอ

ลด batch และใช้ image size ต่ำลง เช่น:

```powershell
python model/detect_license/dataset/train.py `
  --dataset Lpr-2 `
  --batch 4 `
  --imgsz 512
```

### ต้องการเปลี่ยนจำนวน epoch โดยไม่แก้ไฟล์

กำหนดผ่าน command line ได้โดยตรง:

```powershell
python model/detect_license/dataset/train.py --dataset Lpr-2 --epochs 100
```

ระบบมี early stopping (`patience=20`) จึงอาจหยุดก่อนครบจำนวน epoch หากผล
validation ไม่ดีขึ้นเป็นเวลานาน
