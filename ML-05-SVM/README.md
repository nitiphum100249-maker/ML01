# LAB 05: Support Vector Machine — Shoe Brand Classification

โปรเจกต์นี้ใช้ Support Vector Machine (SVM) เพื่อจำแนกรูปภาพรองเท้า 3 แบรนด์ ได้แก่ **Adidas**, **Converse**, และ **Nike** โดยเปรียบเทียบผลลัพธ์จาก SVM Kernel ทั้ง 3 แบบ คือ **Linear**, **Polynomial**, และ **RBF**

## โครงสร้างโปรเจกต์

```
ML-05-SVM/
├── lab05.py              # โค้ดหลักของโปรเจกต์
├── dataset/               # โฟลเดอร์เก็บรูปภาพ (แยกตาม Class)
│   ├── adidas/
│   ├── converse/
│   └── nike/
└── outputs/                # ผลลัพธ์ที่โปรแกรมสร้างขึ้น (สร้างอัตโนมัติ)
    ├── accuracy_scores.csv
    ├── accuracy_comparison.png
    ├── predictions.csv
    ├── prediction_linear.png
    ├── prediction_polynomial.png
    └── prediction_rbf.png
```

## Dataset

- จำนวนรูปภาพ: 238 รูป/แบรนด์ (รวม 711 รูป)
- แต่ละ Class อยู่ในโฟลเดอร์แยกกันภายใต้ `dataset/` โดยชื่อโฟลเดอร์คือชื่อ Class
- รูปภาพจะถูกแปลงเป็น Grayscale และปรับขนาดเป็น 64x64 พิกเซล ก่อนนำไป Flatten เป็น Feature Vector ขนาด 4096

## การติดตั้ง

ต้องติดตั้งไลบรารีต่อไปนี้ก่อนรันโปรแกรม:

```bash
pip install opencv-python numpy pandas matplotlib scikit-learn
```

## วิธีรัน

ให้ตรวจสอบก่อนว่าอยู่ในโฟลเดอร์เดียวกับไฟล์ `lab05.py` และมีโฟลเดอร์ `dataset` อยู่ในระดับเดียวกัน จากนั้นรันคำสั่ง:

```bash
python lab05.py
```

## ขั้นตอนการทำงานของโปรแกรม

1. **โหลดรูปภาพ** จากโฟลเดอร์ `dataset/` พร้อมกำหนด Label ตามชื่อโฟลเดอร์
2. **แปลงรูปภาพเป็น Feature** โดย Flatten รูป 64x64 ให้เป็นเวกเตอร์ 1 มิติ (4096 Feature)
3. **แบ่งข้อมูล** เป็น Train 80% และ Test 20% แบบ Stratified
4. **Standardize ข้อมูล** ด้วย `StandardScaler`
5. **เทรนโมเดล SVM** ทั้ง 3 Kernel (Linear, Polynomial, RBF)
6. **ประเมินผล** และบันทึก Accuracy, กราฟเปรียบเทียบ, และตัวอย่าง Prediction ลงในโฟลเดอร์ `outputs/`

## ผลลัพธ์ที่ได้

| ไฟล์ | คำอธิบาย |
|---|---|
| `accuracy_scores.csv` | ค่า Accuracy ของแต่ละ Kernel |
| `accuracy_comparison.png` | กราฟแท่งเปรียบเทียบ Accuracy ระหว่าง Kernel |
| `predictions.csv` | ผลการทำนายเทียบกับค่าจริงของข้อมูล Test ทั้งหมด |
| `prediction_<kernel>.png` | ตัวอย่างรูปภาพพร้อมผลทำนายของแต่ละ Kernel |

## หมายเหตุ

- ผลลัพธ์ Accuracy อาจไม่สูงมากนัก เนื่องจากใช้ Raw Pixel เป็น Feature โดยตรง (ไม่มีการทำ Feature Extraction เพิ่มเติม เช่น HOG หรือ CNN) ซึ่งทำให้โมเดลไวต่อความแตกต่างของมุมถ่ายภาพ แสง และพื้นหลังของรูปภาพในแต่ละ Class
- `RANDOM_STATE = 42` ถูกกำหนดไว้เพื่อให้ผลการแบ่งข้อมูลเหมือนเดิมทุกครั้งที่รัน
