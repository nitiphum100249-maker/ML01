# LAB 6: Neural Network Image Classification

## 1. รายละเอียดงาน

Lab 6 เป็นการทดลองสร้างโมเดล **Artificial Neural Network (ANN)** สำหรับจำแนกประเภทของรูปภาพรองเท้า โดยใช้ภาษา Python และ TensorFlow/Keras

Dataset ที่ใช้ในการทดลองประกอบด้วยรูปภาพรองเท้า 3 ประเภท ได้แก่

- Adidas
- Converse
- Nike

โปรแกรมจะทำการอ่านรูปภาพจาก Dataset จากนั้นปรับขนาดรูปภาพ แปลงข้อมูลรูปภาพให้อยู่ในรูปแบบที่ Neural Network สามารถนำไปใช้งานได้ และแบ่งข้อมูลออกเป็น Training Dataset และ Testing Dataset

หลังจากนั้นจะทำการสร้าง Neural Network หลายรูปแบบ เพื่อเปรียบเทียบผลลัพธ์จากจำนวน Hidden Layer, จำนวน Neurons และจำนวน Epoch ที่แตกต่างกัน

---

## 2. วัตถุประสงค์

วัตถุประสงค์ของ Lab นี้คือ

1. ศึกษาการเตรียมข้อมูลรูปภาพสำหรับ Machine Learning
2. ศึกษาการใช้งาน Artificial Neural Network สำหรับ Classification
3. ทดลองปรับจำนวน Hidden Layer และจำนวน Neurons
4. ทดลองเปรียบเทียบจำนวน Epoch ที่แตกต่างกัน
5. ประเมินประสิทธิภาพของโมเดลด้วย Accuracy และ Loss
6. แสดงผลการทำนายของโมเดล
7. วิเคราะห์ผลลัพธ์ด้วย Confusion Matrix
8. เปรียบเทียบประสิทธิภาพของ Neural Network แต่ละรูปแบบ

---

## 3. Dataset

Dataset ที่ใช้ประกอบด้วยรูปภาพรองเท้า 3 Classes

```text
train/
│
├── adidas/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── converse/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── nike/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Dataset ที่ใช้ในงานนี้มีทั้งหมด **48 รูป**

แบ่งเป็น

```text
Adidas      : 16 รูป
Converse    : 16 รูป
Nike        : 16 รูป
----------------------
รวม         : 48 รูป
```

ข้อมูลแต่ละ Class มีจำนวนเท่ากัน ทำให้ Dataset มีความสมดุลระหว่างแต่ละประเภท

---

## 4. Libraries ที่ใช้

โปรแกรมใช้ Libraries หลักดังต่อไปนี้

### OpenCV

ใช้สำหรับอ่านและจัดการรูปภาพ

```python
import cv2
```

เช่น

- อ่านไฟล์รูปภาพ
- แปลงสี BGR เป็น RGB
- Resize รูปภาพ

### NumPy

ใช้สำหรับจัดการข้อมูลในรูปแบบ Array

```python
import numpy as np
```

### Pandas

ใช้สำหรับสร้างตารางสรุปผลการทดลอง และบันทึกผลเป็น CSV

```python
import pandas as pd
```

### Matplotlib

ใช้สำหรับสร้างกราฟและแสดงผลรูปภาพ

```python
import matplotlib.pyplot as plt
```

### Scikit-learn

ใช้สำหรับ

- แบ่ง Training และ Testing Dataset
- Standardize ข้อมูล
- สร้าง Confusion Matrix

### TensorFlow / Keras

ใช้สำหรับสร้างและ Train Artificial Neural Network

---

## 5. การเตรียม Dataset

โปรแกรมจะอ่านรูปภาพจากแต่ละ Folder ตาม Class

```python
class_names = ["adidas", "converse", "nike"]
```

จากนั้นทำการอ่านรูปภาพด้วย OpenCV

```python
img = cv2.imread(file_path)
```

หากพบไฟล์ที่ไม่สามารถอ่านเป็นรูปภาพได้ โปรแกรมจะข้ามไฟล์นั้น

```python
if img is None:
    continue
```

---

## 6. การแปลงสีของรูปภาพ

OpenCV จะอ่านรูปภาพในรูปแบบ

```text
BGR
```

แต่ Matplotlib และการใช้งานรูปภาพทั่วไปนิยมใช้

```text
RGB
```

ดังนั้นโปรแกรมจึงแปลงรูปภาพจาก BGR เป็น RGB

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

เพื่อให้สีของรูปภาพถูกต้องเมื่อนำมาแสดงผล

---

## 7. การ Resize รูปภาพ

รูปภาพทั้งหมดจะถูกปรับขนาดให้เท่ากัน

```python
IMG_SIZE = 64
```

ดังนั้นรูปแต่ละรูปจะมีขนาด

```text
64 × 64 Pixels
```

คำสั่งที่ใช้คือ

```python
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
```

หลังจาก Resize แล้ว รูปภาพหนึ่งรูปจะมี Shape

```text
64 × 64 × 3
```

โดย `3` หมายถึงช่องสี

```text
Red
Green
Blue
```

---

## 8. การแปลงรูปภาพเป็น Feature

Artificial Neural Network ที่ใช้ใน Lab นี้รับข้อมูล Input แบบ 1 มิติ

ดังนั้นรูปภาพขนาด

```text
64 × 64 × 3
```

จะถูก Flatten เป็น

```text
64 × 64 × 3 = 12,288 Features
```

ด้วยคำสั่ง

```python
X = X_images.reshape(len(X_images), -1)
```

ดังนั้นรูปภาพแต่ละรูปจะถูกแทนด้วยข้อมูลจำนวน **12,288 Features**

---

## 9. การแบ่ง Training และ Testing Dataset

ข้อมูลจะถูกแบ่งเป็น

```text
Training Dataset = 80%
Testing Dataset  = 20%
```

โดยใช้

```python
train_test_split()
```

และกำหนด

```python
test_size=0.20
random_state=42
stratify=y
```

การใช้

```python
stratify=y
```

ช่วยรักษาสัดส่วนของแต่ละ Class ให้ใกล้เคียงกันระหว่าง Training Dataset และ Testing Dataset

---

## 10. Standardization

ก่อนนำข้อมูลเข้า Neural Network โปรแกรมจะใช้

```python
StandardScaler()
```

เพื่อปรับข้อมูล Feature ให้อยู่ใน Scale ที่เหมาะสม

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Scaler จะเรียนรู้ค่า Mean และ Standard Deviation จาก Training Dataset เท่านั้น

จากนั้นจึงใช้ค่าที่ได้จาก Training Dataset มาปรับ Testing Dataset

วิธีนี้ช่วยป้องกันข้อมูลจาก Testing Dataset รั่วไหลเข้าไปในขั้นตอน Training

---

## 11. Neural Network

โปรแกรมสร้าง Neural Network ด้วย

```python
Sequential()
```

และใช้ Fully Connected Layer หรือ

```python
Dense
```

Hidden Layer ใช้ Activation Function

```text
ReLU
```

หรือ

```python
activation="relu"
```

---

## 12. Model Configurations

โปรแกรมทดลอง Neural Network ทั้งหมด 3 รูปแบบ

### Model 1

```text
Input
 ↓
64 Neurons
 ↓
Output
```

Configuration

```python
[64]
```

มี Hidden Layer จำนวน 1 Layer

---

### Model 2

```text
Input
 ↓
128 Neurons
 ↓
64 Neurons
 ↓
Output
```

Configuration

```python
[128, 64]
```

มี Hidden Layer จำนวน 2 Layers

---

### Model 3

```text
Input
 ↓
256 Neurons
 ↓
128 Neurons
 ↓
64 Neurons
 ↓
Output
```

Configuration

```python
[256, 128, 64]
```

มี Hidden Layer จำนวน 3 Layers

---

## 13. Output Layer

Dataset มีทั้งหมด 3 Classes

```text
Adidas
Converse
Nike
```

ดังนั้น Output Layer จะมีจำนวน Neurons เท่ากับจำนวน Classes

```python
Dense(
    len(class_names),
    activation="softmax"
)
```

Softmax จะคำนวณ Probability ของแต่ละ Class

ตัวอย่าง

```text
Adidas      : 5%
Converse    : 10%
Nike        : 85%
```

ดังนั้น Model จะทำนายว่ารูปภาพดังกล่าวคือ

```text
Nike
```

---

## 14. Loss Function

โปรแกรมใช้ Loss Function

```python
sparse_categorical_crossentropy
```

เนื่องจาก Label ของ Dataset ถูกเก็บในรูปตัวเลข

ตัวอย่าง

```text
0 = adidas
1 = converse
2 = nike
```

---

## 15. Optimizer

Optimizer ที่ใช้คือ

```python
adam
```

Adam เป็น Optimizer ที่ใช้ปรับ Weight ของ Neural Network ระหว่าง Training

---

## 16. Accuracy

Metric ที่ใช้วัดประสิทธิภาพของ Model คือ

```python
accuracy
```

ซึ่งแสดงสัดส่วนของข้อมูลที่ Model สามารถจำแนกได้ถูกต้อง

ตัวอย่าง

```text
Accuracy = 0.80
```

หมายความว่า Model ทำนายถูกประมาณ

```text
80%
```

---

## 17. Epoch

โปรแกรมทดลองจำนวน Epoch ที่แตกต่างกันทั้งหมด 3 ค่า

```python
epoch_list = [
    10,
    30,
    50
]
```

ดังนั้นแต่ละ Model จะถูกทดลองด้วย

```text
10 Epochs
30 Epochs
50 Epochs
```

มี Neural Network 3 รูปแบบ

และ Epoch 3 รูปแบบ

ดังนั้นจะมีการทดลองทั้งหมด

```text
3 × 3 = 9 Experiments
```

---

## 18. Batch Size

โปรแกรมกำหนด

```python
batch_size=16
```

หมายความว่าในการ Training แต่ละครั้ง Model จะประมวลผลข้อมูลครั้งละ 16 Samples ก่อนทำการปรับ Weight

---

## 19. Validation Dataset

ระหว่าง Training โปรแกรมกำหนด

```python
validation_split=0.20
```

หมายความว่าจะนำข้อมูล Training จำนวน 20% ไปใช้เป็น Validation Dataset

Validation Dataset ใช้สำหรับตรวจสอบว่า Model สามารถทำงานกับข้อมูลที่ไม่ได้ใช้ปรับ Weight ได้ดีเพียงใด

---

## 20. การเลือก Best Model

หลังจาก Training แต่ละ Model โปรแกรมจะนำ Model ไปทดสอบกับ Testing Dataset

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)
```

จากนั้นเปรียบเทียบค่า

```text
Test Accuracy
```

Model ที่มี Test Accuracy สูงที่สุดจะถูกเก็บเป็น

```python
best_model
```

พร้อมกับเก็บ

```text
best_history
best_name
best_epoch
best_accuracy
```

สำหรับนำไปสร้างผลลัพธ์ขั้นสุดท้าย

---

## 21. Results

ผลการทดลองทั้งหมดจะถูกจัดเก็บใน Pandas DataFrame

ข้อมูลที่บันทึกประกอบด้วย

```text
Model
Hidden Layers
Epochs
Test Accuracy
Test Loss
Final Train Accuracy
Final Validation Accuracy
```

จากนั้นบันทึกเป็นไฟล์

```text
outputs/results.csv
```

ไฟล์นี้สามารถนำไปใช้เปรียบเทียบประสิทธิภาพของ Neural Network แต่ละ Configuration ได้

---

## 22. Training History

โปรแกรมสร้างกราฟเปรียบเทียบ

```text
Training Accuracy
```

กับ

```text
Validation Accuracy
```

ของ Best Model

และบันทึกเป็น

```text
outputs/training_history.png
```

กราฟนี้ช่วยให้สามารถสังเกตพฤติกรรมของ Model ระหว่างการ Training ได้

ตัวอย่างเช่น หาก Training Accuracy สูงขึ้นเรื่อย ๆ แต่ Validation Accuracy ลดลง อาจเป็นสัญญาณของ

```text
Overfitting
```

---

## 23. Confusion Matrix

โปรแกรมนำ Best Model มาทำนาย Testing Dataset

```python
prediction_probability = best_model.predict(X_test)
```

จากนั้นเลือก Class ที่มี Probability สูงที่สุด

```python
y_pred = np.argmax(
    prediction_probability,
    axis=1
)
```

แล้วสร้าง Confusion Matrix

```python
confusion_matrix(
    y_test,
    y_pred
)
```

ผลลัพธ์จะถูกบันทึกเป็น

```text
outputs/confusion_matrix.png
```

Confusion Matrix ช่วยให้สามารถตรวจสอบได้ว่า Model จำแนก Class ใดถูกหรือผิดมากน้อยเพียงใด

---

## 24. Prediction Sample

โปรแกรมจะสุ่มรูปภาพจาก Testing Dataset จำนวนสูงสุด 4 รูป

จากนั้นแสดง

```text
Pred
True
Confidence
```

ตัวอย่าง

```text
Pred: nike (92%)
True: nike
```

หมายความว่า

```text
ค่าที่ Model ทำนาย = Nike
ค่าจริง             = Nike
ความมั่นใจ          = 92%
```

หากทำนายถูก ข้อความจะแสดงเป็นสีเขียว

หากทำนายผิด ข้อความจะแสดงเป็นสีแดง

ผลลัพธ์จะถูกบันทึกเป็น

```text
outputs/prediction_sample.png
```

---

## 25. Output Files

หลังจากรันโปรแกรมเสร็จ โปรแกรมจะสร้าง Folder

```text
outputs/
```

ภายในประกอบด้วย

```text
outputs/
│
├── results.csv
├── training_history.png
├── confusion_matrix.png
└── prediction_sample.png
```

รายละเอียดของแต่ละไฟล์

### results.csv

เก็บผลการทดลองของ Neural Network ทั้ง 9 Configurations

### training_history.png

กราฟ Training Accuracy และ Validation Accuracy ของ Model ที่ดีที่สุด

### confusion_matrix.png

Confusion Matrix ของ Model ที่ดีที่สุด

### prediction_sample.png

ตัวอย่างรูปภาพจาก Testing Dataset พร้อมผล Prediction

---

## 26. การติดตั้ง Library

สามารถติดตั้ง Library ที่จำเป็นได้ด้วยคำสั่ง

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow
```

---

## 27. วิธีใช้งาน

จัด Folder ให้มีโครงสร้างดังนี้

```text
project/
│
├── lab06_dataset_fixed.py
│
└── train/
    ├── adidas/
    ├── converse/
    └── nike/
```

จากนั้นเปิด Terminal ใน Folder ของ Project

และใช้คำสั่ง

```bash
python lab06_dataset_fixed.py
```

โปรแกรมจะเริ่มอ่าน Dataset และแสดงจำนวนรูปภาพ

จากนั้น Train Model ทั้งหมด 9 การทดลอง

เมื่อเสร็จแล้วโปรแกรมจะแสดง

```text
Best Model
Best Epoch
Best Accuracy
```

พร้อมสร้างผลลัพธ์ไว้ใน Folder

```text
outputs/
```

---

## 28. ลำดับการทำงานของโปรแกรม

ภาพรวมการทำงานสามารถสรุปได้ดังนี้

```text
Dataset
   ↓
Read Images
   ↓
BGR → RGB
   ↓
Resize 64 × 64
   ↓
Convert to NumPy Array
   ↓
Flatten Images
   ↓
12,288 Features
   ↓
Train / Test Split
   ↓
StandardScaler
   ↓
Create Neural Network
   ↓
Train Model
   ↓
Evaluate Model
   ↓
Compare Accuracy
   ↓
Select Best Model
   ↓
Confusion Matrix
   ↓
Prediction Sample
   ↓
Save Results
```

---

## 29. สรุป

Lab 6 เป็นการทดลองนำ Artificial Neural Network มาประยุกต์ใช้กับงาน Image Classification โดยใช้ Dataset รูปภาพรองเท้าประเภท Adidas, Converse และ Nike

ก่อนนำข้อมูลเข้า Neural Network รูปภาพทั้งหมดจะถูก Resize เป็นขนาด 64 × 64 Pixels และแปลงจากข้อมูลรูปภาพแบบ 3 มิติให้เป็น Feature แบบ 1 มิติ จำนวน 12,288 Features ต่อรูป

จากนั้นข้อมูลจะถูกแบ่งออกเป็น Training Dataset และ Testing Dataset และทำ Standardization ก่อนนำไป Train Neural Network

ในการทดลองมีการสร้าง Neural Network จำนวน 3 รูปแบบ และทดสอบด้วยจำนวน Epoch 10, 30 และ 50 Epochs ทำให้มีการทดลองทั้งหมด 9 รูปแบบ

ระบบจะนำค่า Test Accuracy ของแต่ละการทดลองมาเปรียบเทียบกัน และเลือก Model ที่มี Accuracy สูงที่สุดเป็น Best Model

สุดท้ายโปรแกรมจะแสดง Training History, Confusion Matrix และตัวอย่าง Prediction เพื่อช่วยในการวิเคราะห์ประสิทธิภาพของ Model

การทดลองนี้ทำให้สามารถศึกษากระบวนการพื้นฐานของ Image Classification ตั้งแต่ขั้นตอนการเตรียม Dataset การสร้าง Neural Network การ Training การทดสอบ และการประเมินผลของโมเดลได้ครบถ้วน