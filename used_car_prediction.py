import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# MEMANGGIL DATASET
df = pd.read_csv('toyota.csv')
print(df.head())

# df.info()
# sns.heatmap(df.isnull())
# df.describe()

# VISUALISASI DATA
# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(),annot=True)
# print(plt.show())

# JUMLAH MOBIL BERDASARKAN MODEL
models = df.groupby('model').count()[['tax']].sort_values(by='tax',ascending=True).reset_index()
models = models.rename(columns={'tax':'numberOfCars'})

fig = plt.figure(figsize=(15,5))
sns.barplot(x=models['model'],
            y=models['numberOfCars'],
            color='royalblue')
plt.title("Jumlah Mobil Berdasarkan Model")
plt.xticks(rotation=60)
plt.show()

# UKURAN MESIN
engine = df.groupby('engineSize').count()[['tax']].sort_values(by='tax').reset_index()
engine = engine.rename(columns={'tax':'count'})

plt.figure(figsize=(15,5))
sns.barplot(x=engine['engineSize'], y=engine['count'], color='royalblue')
plt.title("Ukuran Mesin Mobil")
plt.show()

# DISTRIBUSI HARGA MOBIL
plt.figure(figsize=(15,5))
sns.distplot(df['price'])
plt.title("Distribusi Harga Mobil")
plt.show()

# SELEKSI FITUR
features = ['year','mileage','tax','mpg','engineSize']
x = df[features]
y = df['price']
x.shape, y.shape

# SPLIT DATA TRAINING DAN DATA TESTING
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape

# MEMBUAT MODEL REGRESI LINIER
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)

score = lr.score(x_test, y_test)
print("Akurasi model regresi linier = ", score)

# MEMBUAT INPUTAN MODEL REGRESI LINIER
# terdiri dari 'year','mileage','tax','mpg','engineSize'
input_data = np.array([[2019,5000,145,30.2,2]])

prediction = lr.predict(input_data)
print("Estimasi harga mobil dalam EUR : ", prediction)

# SAVE MODEL MENGGUNAKAN PICKLE
import pickle
filename = 'estimasi_mobil.sav'
pickle.dump(lr,open(filename,'wb'))
