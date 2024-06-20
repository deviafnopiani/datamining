import streamlit as st
import numpy as np

from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title('Aplikasi Web Data Mining')
st.write("""
# Menggunakan beberapa algoritma dataset yang berbeda
yang mana terbaik?
""") 

nama_dataset = st.sidebar.selectbox(
    'Pilih Dataset',
    ('Bunga IRIS', 'Kanker Payudara', 'Digit Angka')
)

st.write(f"## Dataset {nama_dataset}")

algoritma = st.sidebar.selectbox(
    'Pilih Algoritma',
    ('KNN','SVM', 'Random Forest')
)

def pilih_dataset(nama):
    data = None
    if nama == 'Bunga IRIS':
        data = datasets.load_iris()
    elif nama == 'Kanker Payudara':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_digits()
    X = data.data
    y = data.target
    return X,y

X,y = pilih_dataset(nama_dataset)
st.write('Jumlah Baris dan Kolom : ', X.shape)
st.write('Jumlah Kelas : ', len(np.unique(y)))

def tambah_parameter(nama_algoritma):
    params = dict()
    if nama_algoritma == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif nama_algoritma == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st. sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimator', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = tambah_parameter(algoritma)

def pilih_klasifikasi(nama_algoritma, params):
    algo = None
    if nama_algoritma == 'KNN':
        algo = KNeighborsClassifier(n_neighbors=params['K'])
    elif nama_algoritma == 'SVM':
        algo = SVC(C=params['C'])
    else:
        algo = RandomForestClassifier(n_estimators=params['n_estimators'],
                max_depth=params['max_depth'], random_state=1234)
    return algo


algo = pilih_klasifikasi(algoritma,params)

##proses klasifikasi

X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=0.2, random_state=1234)

algo.fit(X_train, y_train)
y_pred = algo.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Algoritma = {algoritma}')
st.write(f'Akurasi   =', acc)

##plot dataset
#memproyeksikan data dalam 2 komponen PCA

pca = PCA (2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Component 1')
plt.xlabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)