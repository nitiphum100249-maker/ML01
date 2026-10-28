# CNN Project: Fruits Plant Leaf Classification

## 1. รายละเอียดโครงงาน
โครงงานนี้ใช้เทคนิค **Convolutional Neural Network (CNN)** เพื่อจำแนกภาพใบพืชผลไม้ตามประเภทของพืช โดยต่อยอดโครงสร้างการทำงานจากไฟล์ `ML-07-CNN.zip` ได้แก่ การโหลดภาพ การเตรียมข้อมูล การแบ่งชุดข้อมูล การฝึกโมเดล และการประเมินผล

> หมายเหตุ: โมเดลนี้จำแนกจาก **ภาพใบไม้** ไม่ใช่ภาพผลไม้โดยตรง

## 2. Dataset
- ชื่อชุดข้อมูล: **Fruits Plant Leaf**
- ขนาดดาวน์โหลดที่ระบุบนหน้า Dataset: ประมาณ 5.95 MB
- จำนวนข้อมูลที่อ้างอิง: 291 ภาพ
- จำนวนคลาส: 10 คลาส ได้แก่ Apple, Banana, Grapes, Guava, Jujube, Orange, Mango, Pineapple, Papaya และ Watermelon
- แหล่งข้อมูล: https://doi.org/10.34740/KAGGLE/DSV/12928903
- ใบอนุญาตที่ระบุโดยแหล่งข้อมูล: CC BY-NC-SA 4.0 ควรตรวจสอบเงื่อนไขก่อนนำไปเผยแพร่หรือใช้เชิงพาณิชย์

Dataset ไม่ได้ถูกรวมไว้ในไฟล์ ZIP ของโปรเจกต์ ต้องดาวน์โหลดและแตกไฟล์แยกต่างหาก

## 3. โครงสร้างโฟลเดอร์
จากโครงสร้างโปรเจกต์ที่ใช้ ให้จัดวางไฟล์ในลักษณะนี้:

```text
Fruits_CNN_Project_Leaf_Dataset/
├── .venv/                         # Python virtual environment (ถ้ามี)
├── dataset/
│   ├── Fruits_Plant_Leaf/         # โฟลเดอร์ภาพที่แตกจาก Dataset
│   │   ├── Apple/
│   │   ├── Banana/
│   │   ├── Grapes/
│   │   └── ...                    # คลาสอื่น ๆ
│   └── Fruits_Plant_Leaf.csv      # ถ้ามีไฟล์ CSV
└── Fruits_CNN_Project/
    ├── main.py
    ├── cnn_model.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── split_data.py
    ├── evaluate.py
    ├── test_cnn.py
    ├── requirements.txt
    ├── README.md
    ├── DATASET_INFO.md
    └── outputs/                   # โปรแกรมสร้างเมื่อรัน
```

ตัวโหลดข้อมูลจะค้นหาโฟลเดอร์คลาสจากโฟลเดอร์ที่กำหนด โดยแต่ละคลาสควรมีภาพอยู่ภายใน เช่น `Apple/*.jpg` และ `Banana/*.jpg`

## 4. สิ่งที่ต้องติดตั้ง
- Python 3.10 หรือ 3.11 แนะนำให้ใช้เวอร์ชันที่เข้ากันได้กับ TensorFlow ที่ติดตั้ง
- ไลบรารีตาม `requirements.txt`

### ติดตั้งไลบรารีบน Windows
เปิด VS Code Terminal ที่โฟลเดอร์ `Fruits_CNN_Project` แล้วรันคำสั่ง:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

หากยังไม่มี `.venv` ที่โฟลเดอร์หลัก ให้สร้างจากโฟลเดอร์ `Fruits_CNN_Project_Leaf_Dataset`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\Fruits_CNN_Project\requirements.txt
```

## 5. วิธีรันโปรแกรมบน Windows
### ขั้นตอนที่ 1: เปิด Terminal ในโฟลเดอร์โปรเจกต์
ตำแหน่ง Terminal ควรลงท้ายด้วย:

```text
...\Fruits_CNN_Project_Leaf_Dataset\Fruits_CNN_Project>
```

### ขั้นตอนที่ 2: ระบุที่อยู่ Dataset
เพราะโฟลเดอร์ `dataset` อยู่ระดับเดียวกับโฟลเดอร์ `Fruits_CNN_Project` ให้ตั้งค่าพาธดังนี้:

```powershell
$env:FRUIT_DATA_PATH = "..\dataset"
```

### ขั้นตอนที่ 3: เริ่มฝึกโมเดล
ถ้า `.venv` อยู่ในโฟลเดอร์หลัก `Fruits_CNN_Project_Leaf_Dataset` ให้รัน:

```powershell
..\.venv\Scripts\python.exe main.py
```

ถ้าใช้ Python ที่ติดตั้งในเครื่องแทน ให้รัน:

```powershell
py main.py
```

### ขั้นตอนที่ 4: ทดลองทำนายภาพ
หลังจาก `main.py` ฝึกโมเดลเสร็จแล้ว ให้รัน:

```powershell
..\.venv\Scripts\python.exe test_cnn.py
```

หากใช้ `py` ในการรัน `main.py` ให้ใช้ `py test_cnn.py` ได้เช่นกัน

## 6. ขั้นตอนการทำงานของโปรแกรม
1. **Load Data:** ค้นหาโฟลเดอร์คลาสและอ่านไฟล์ภาพ
2. **Preprocessing:** ปรับภาพเป็นขนาด 100 × 100 พิกเซล และจัดรูปแบบภาพสำหรับโมเดล
3. **Split Dataset:** แบ่งข้อมูลเป็นชุดฝึก (Training) ประมาณ 70%, ชุดตรวจสอบ (Validation) 15% และชุดทดสอบ (Testing) 15%
4. **Train CNN:** ฝึกโมเดลด้วยภาพในชุดฝึก และใช้ชุด Validation ติดตามผลระหว่างฝึก
5. **Evaluate:** ทดสอบโมเดลด้วยชุด Test และสร้าง Confusion Matrix รวมถึงกราฟประวัติการฝึก
6. **Predict:** ใช้ `test_cnn.py` ทดลองทำนายภาพตัวอย่างจากชุดทดสอบ

สัดส่วนจริงอาจคลาดเคลื่อนเล็กน้อยจากการปัดเศษและจำนวนภาพในแต่ละคลาส

## 7. การตั้งค่าหลัก
ค่าต่าง ๆ แก้ไขได้ที่ส่วนต้นของ `main.py`:

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|---|---:|---|
| `IMG_SIZE` | `100` | ปรับภาพเป็น 100 × 100 พิกเซล |
| `MAX_PER_CLASS` | `None` | ใช้ภาพทั้งหมดที่อ่านได้ในแต่ละคลาส |
| `TEST_SIZE` | `0.15` | สัดส่วนชุดทดสอบ |
| `VAL_SIZE` | `0.15` | สัดส่วนชุด Validation |
| `EPOCHS` | `20` | จำนวนรอบฝึกสูงสุด |
| `BATCH_SIZE` | `32` | จำนวนภาพต่อหนึ่ง batch |

เนื่องจาก Dataset มีภาพจำนวนน้อย ผลการประเมินอาจเปลี่ยนแปลงได้ตามการแบ่งข้อมูล และมีความเสี่ยงเกิด Overfitting

## 8. ไฟล์ผลลัพธ์
เมื่อโปรแกรมทำงานสำเร็จ จะบันทึกผลไว้ในโฟลเดอร์ `Fruits_CNN_Project/outputs/` ตัวอย่างเช่น:

- `cnn_model.keras` — โมเดล CNN ที่ฝึกแล้ว
- `classes.json` — รายชื่อคลาสที่โมเดลรู้จัก
- `X_train.npy`, `X_val.npy`, `X_test.npy` — ข้อมูลภาพที่แบ่งเป็นชุด
- `y_train.npy`, `y_val.npy`, `y_test.npy` — ป้ายกำกับของแต่ละชุด
- `history.json` — ประวัติการฝึก
- `training_history.png` — กราฟผลการฝึก
- `confusion_matrix.png` — ตารางเปรียบเทียบคลาสจริงกับคลาสที่ทำนาย

ไฟล์บางรายการจะปรากฏหลังจากขั้นตอนที่เกี่ยวข้องทำงานสำเร็จเท่านั้น

## สรุป
โปรเจกต์นี้แสดงขั้นตอนการใช้ CNN จำแนกภาพใบพืชผลไม้ ตั้งแต่โหลดและเตรียมข้อมูล แบ่งข้อมูล ฝึกโมเดล ไปจนถึงประเมินผลและทดลองทำนาย เหมาะสำหรับการศึกษาและทดลอง Machine Learning ด้วย Dataset ขนาดเล็ก
