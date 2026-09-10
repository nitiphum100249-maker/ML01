# LAB 04 - KNN Classification

## รายละเอียด

โปรเจกต์นี้เป็นการทดลองใช้งาน **K-Nearest Neighbors (KNN)** สำหรับงาน Classification โดยใช้ Dataset `labelnames.csv`

Dataset ประกอบด้วยข้อมูล 2 คอลัมน์ ได้แก่

* `Name` — ชื่อไฟล์รูปภาพ
* `Label` — Class หรือยี่ห้อ เช่น Adidas / Nike

เนื่องจาก Dataset มีเพียงชื่อไฟล์และ Label จึงมีการสร้าง Feature เชิงตัวเลขจากชื่อไฟล์ก่อนนำไปใช้กับ KNN เช่น ความยาวชื่อไฟล์ ตัวเลขในชื่อไฟล์ และจำนวนขีดล่าง `_`

---

## โครงสร้างไฟล์

```text
LAB04/
│
├── lab04.py
├── labelnames.csv
│
└── outputs/
    └── results.csv
```

---

## Library ที่ใช้

โปรแกรมใช้ Python และ Library ดังต่อไปนี้

```text
pandas
scikit-learn
```

สามารถติดตั้ง Library ได้ด้วยคำสั่ง

```bash
pip install pandas scikit-learn
```

---

## ขั้นตอนการทำงาน

### 1. Load Dataset

อ่านข้อมูลจากไฟล์

```python
df = pd.read_csv("labelnames.csv")
```

จากนั้นแสดงข้อมูล 5 แถวแรก ตรวจสอบจำนวนข้อมูล ชื่อ Column ค่า Missing และข้อมูลซ้ำ

### 2. Data Preprocessing

จัดการข้อมูลก่อนนำไปสร้าง Model โดย

* ลบข้อมูลที่เป็นค่าว่าง
* ลบข้อมูลที่ซ้ำกัน

```python
df = df.dropna()
df = df.drop_duplicates()
```

### 3. Feature Engineering

สร้าง Feature จากชื่อไฟล์ ได้แก่

| Feature            | รายละเอียด                  |
| ------------------ | --------------------------- |
| `name_length`      | ความยาวของชื่อไฟล์          |
| `number_in_name`   | ตัวเลขที่พบในชื่อไฟล์       |
| `has_jpg`          | ตรวจสอบว่ามี `.JPG` หรือไม่ |
| `underscore_count` | จำนวน `_` ในชื่อไฟล์        |

Feature เหล่านี้ถูกสร้างขึ้นเพื่อให้สามารถนำข้อมูลไปใช้กับ KNN ได้

### 4. Select Features และ Target

กำหนด Feature สำหรับการทำนาย

```python
X = df[
    [
        "name_length",
        "number_in_name",
        "has_jpg",
        "underscore_count"
    ]
]
```

และกำหนด Target เป็น

```python
y = df["Label"]
```

### 5. Train / Test Split

แบ่ง Dataset เป็น

* Training Data = 80%
* Testing Data = 20%

โดยกำหนด

```python
random_state = 42
```

และใช้ `stratify=y` เพื่อรักษาสัดส่วนของแต่ละ Class

### 6. Standardization

ใช้ `StandardScaler` เพื่อปรับ Scale ของ Feature ก่อนนำไป Train KNN

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 7. KNN Classification

ทดลองค่า K จำนวน 3 ค่า

```python
k_values = [3, 5, 7]
```

โดย K หมายถึงจำนวนเพื่อนบ้านที่ใช้ในการพิจารณา Class ของข้อมูลใหม่

### 8. Accuracy

ประเมินผลการ Classification ด้วย **Accuracy**

```python
accuracy = accuracy_score(y_test, y_pred)
```

จากนั้นเก็บผล Accuracy ของแต่ละค่า K

### 9. เลือกค่า K ที่ดีที่สุด

โปรแกรมจะเลือกค่า K ที่มี Accuracy สูงที่สุด

```python
best_k = max(
    results,
    key=results.get
)
```

และแสดงผล

```text
Best K
Best Accuracy
```

### 10. บันทึกผลลัพธ์

ผลการทดลองจะถูกบันทึกไว้ที่

```text
outputs/results.csv
```

โดยมีข้อมูล

```text
K
Accuracy
```

---

ตัวอย่างผลลัพธ์

```text
First 5 Rows

Dataset Shape
...

Columns
...

Missing Values
...

K = 3 | Accuracy = ...
K = 5 | Accuracy = ...
K = 7 | Accuracy = ...

Best Result
Best K = ...
Best Accuracy = ...

Results saved to outputs/results.csv
```
