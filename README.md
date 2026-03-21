# โปรเจคทำนายการยกเลิกบริการของลูกค้า (Customer Churn Prediction)

## ภาพรวมของโปรเจค

โปรเจคนี้มีเป้าหมายเพื่อ **ทำนายว่าลูกค้าธนาคารจะยกเลิกการใช้บริการหรือไม่ (Customer Churn)** โดยใช้เทคนิค Machine Learning

การทำนาย Customer Churn เป็นปัญหาที่สำคัญในธุรกิจ เนื่องจากการหาลูกค้าใหม่มักมีค่าใช้จ่ายสูงกว่าการรักษาลูกค้าเดิม หากสามารถทำนายลูกค้าที่มีแนวโน้มจะยกเลิกบริการได้ ธุรกิจสามารถวางแผนการรักษาลูกค้าได้ล่วงหน้า

ในโปรเจคนี้ใช้โมเดล **Random Forest Classifier** ในการสร้างโมเดลทำนาย

## Dataset ที่ใช้

Dataset ที่ใช้คือ **Churn Modelling Dataset**

แหล่งที่มา:
https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

ข้อมูลประกอบด้วยรายละเอียดของลูกค้าธนาคาร เช่น

* Credit Score (คะแนนเครดิต)
* Geography (ประเทศ)
* Gender (เพศ)
* Age (อายุ)
* Tenure (ระยะเวลาที่เป็นลูกค้า)
* Balance (ยอดเงินคงเหลือ)
* Number of Products (จำนวนผลิตภัณฑ์ที่ใช้)
* Has Credit Card (มีบัตรเครดิตหรือไม่)
* Is Active Member (เป็นลูกค้าที่ใช้งานอยู่หรือไม่)
* Estimated Salary (รายได้โดยประมาณ)

ตัวแปรเป้าหมาย (Target Variable)

**Exited**

* 1 = ลูกค้ายกเลิกบริการ (Churn)
* 0 = ลูกค้ายังใช้บริการอยู่

## ขั้นตอนการวิเคราะห์ข้อมูล (EDA)

ก่อนการสร้างโมเดล มีการสำรวจข้อมูลเบื้องต้น เช่น

* ตรวจสอบขนาดของ dataset
* วิเคราะห์สถิติพื้นฐานของข้อมูล
* วิเคราะห์การกระจายของลูกค้าที่ churn และไม่ churn
* ตรวจสอบ missing values
* ตรวจสอบข้อมูลซ้ำ
* ตรวจสอบ outlier
การทำ EDA ช่วยให้เข้าใจข้อมูลก่อนนำไปสร้างโมเดล Machine Learning

## การเตรียมข้อมูล (Data Preprocessing)

มีการเตรียมข้อมูลก่อน train โมเดล ดังนี้

* ลบคอลัมน์ที่ไม่จำเป็น ได้แก่

  * RowNumber
  * CustomerId
  * Surname

* แปลงข้อมูลประเภทหมวดหมู่ (Categorical Data) ให้เป็นตัวเลขด้วย **One-Hot Encoding**

* แบ่งข้อมูลเป็น

Training Set : 80%
Test Set : 20%

## การสร้างโมเดล Machine Learning

มีการทดลองใช้โมเดลหลายแบบ เช่น

* Logistic Regression
* Decision Tree
* Random Forest

โมเดลที่เลือกใช้เป็นโมเดลหลักคือ

**Random Forest Classifier**

เหตุผลที่เลือกโมเดลนี้

* สามารถจัดการข้อมูลที่มีความสัมพันธ์ซับซ้อนได้
* ลดปัญหา Overfitting
* เหมาะกับข้อมูลประเภทตาราง (Tabular Data)

## การประเมินผลโมเดล

ใช้หลาย Metrics ในการประเมินโมเดล เช่น

* Accuracy
* Confusion Matrix
* Classification Report
* ROC Curve
* AUC Score

Metrics เหล่านี้ช่วยวัดความสามารถของโมเดลในการทำนายว่าลูกค้าจะ churn หรือไม่

## โครงสร้างโปรเจค

คำอธิบายไฟล์

* **model_artifacts** → โฟลเดอร์เก็บไฟล์โมเดลและไฟล์ที่เกี่ยวข้องหลังจาก train โมเดล
* **churn_model.pkl** → ไฟล์โมเดล Random Forest ที่ใช้สำหรับทำนาย
* **feature_names.json** → รายชื่อ features ที่ใช้ในการ train โมเดล
* **metadata.json** → ข้อมูลเกี่ยวกับโมเดล เช่น accuracy และจำนวน features
* **appds.py** → โปรแกรม Streamlit สำหรับทำนาย churn
* **Churn_Modelling.csv** → dataset ที่ใช้
* **requirements.txt** → รายชื่อ library ที่ใช้ในโปรเจค

## Web Application (Streamlit)

ในโปรเจคนี้มีการสร้าง **Streamlit Web Application**

ผู้ใช้สามารถกรอกข้อมูลลูกค้า เช่น

* อายุ
* ยอดเงิน
* รายได้

จากนั้นระบบจะใช้โมเดล Machine Learning เพื่อ

**ทำนายว่าลูกค้ามีโอกาส churn หรือไม่**

## วิธีรันโปรเจค

ติดตั้ง library ที่จำเป็น

pip install -r requirements.txt

รัน Streamlit application

streamlit run appds.py

## เทคโนโลยีที่ใช้

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## ผู้จัดทำ

รหัสนิสิต: **67160333**

โปรเจครายวิชา Data Science / Machine Learning
