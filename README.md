# ML01

ขั้นตอนที่ 0: การเตรียมข้อมูลและไลบรารี (Setup & Data Generation)
ก่อนเริ่มกระบวนการ เราจำเป็นต้องนำเข้าไลบรารีที่จำเป็นและสร้างชุดข้อมูลจำลองที่ครอบคลุมปัญหาต่างๆ (เช่น ค่าว่าง, ข้อมูลซ้ำ) ตามโจทย์ของใบงาน
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # ปิดการแจ้งเตือนที่ไม่จำเป็
import pandas as pd: เรียกใช้ไลบรารี Pandas สำหรับจัดการข้อมูลในรูปแบบตาราง (DataFrame)
import numpy as np: เรียกใช้ Numpy สำหรับการคำนวณทางคณิตศาสตร์และการจัดการค่าว่าง (np.nan)
import matplotlib.pyplot as plt และ import seaborn as sns: เรียกใช้ไลบรารีสำหรับสร้างกราฟและ Data Visualization

LAB 1: การสำรวจชุดข้อมูล (Dataset Exploration)
ขั้นตอนนี้คือการทำความเข้าใจโครงสร้างและปัญหาเบื้องต้นของชุดข้อมูล
# 1. โหลดข้อมูลจากไฟล์ CSV เข้าสู่ตัวแปร df (DataFrame)
df = pd.read_csv('lab2_machine_learning_dataset.csv')

# 2. ตรวจสอบมิติของตาราง (จำนวนแถว x จำนวนคอลัมน์)
print("Shape of dataset:", df.shape)

# 3. ตรวจสอบชนิดของข้อมูลในแต่ละคอลัมน์ (เช่น int64, float64, object)
print(df.dtypes)

# 4. แสดงค่าสถิติเชิงพรรณนาเบื้องต้น (เช่น ค่าเฉลี่ย, ค่าต่ำสุด, ค่าสูงสุด, จำนวนข้อมูล)
print(df.describe(include='all'))

# 5. นับจำนวนค่าว่าง (Missing Values) ในแต่ละคอลัมน์
print(df.isnull().sum())

# 6. นับจำนวนแถวที่มีข้อมูลซ้ำซ้อนกันทุกประการ (Duplicate Records)
print("Total duplicates:", df.duplicated().sum())

# 7. ดูการกระจายตัวของคลาสเป้าหมาย (นับจำนวนคนได้เลื่อนขั้น vs ไม่ได้เลื่อนขั้น)
print(df['Is_Promoted'].value_counts())

LAB 2: การทำภาพกราฟิกข้อมูล (Data Visualization)
การนำข้อมูลที่ทำความสะอาดแล้วมาแสดงผลเพื่อหาความสัมพันธ์
