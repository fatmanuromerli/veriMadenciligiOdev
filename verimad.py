# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:52:18 2023

@author: omerli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


#excel verisini okuyorum
dosya_yolu = "C:\\Users\\omerli\\Downloads\\23_24_Proje_1_Veri_Kumesiiiisonhali.xlsx"
veri_seti = pd.read_excel(dosya_yolu)
#print(veri)

# sutunlarda %50 den fazla eksik veri var mı kontrol?
sutunlar= veri_seti.columns
"""
#for sutun in sutunlar:
    #eksik_veri_orani = veri_seti[sutun].isnull().mean()

    #if eksik_veri_orani > 0.5:
        #print(f"{sutun} sütunu %50'den fazla eksik veri içeriyor.") #plasmayı verdi
    #else:
        #print(f"{sutun} sütunu %50'den fazla eksik veri içermiyor.")
"""
veri= pd.DataFrame(veri_seti)
df = veri.copy()



# Veri kümesinin temel istatistiklerini incele
istatistikler = df.describe().T
print(istatistikler)


    
label_encoder = preprocessing.LabelEncoder()
df['Cinsiyet']= label_encoder.fit_transform(df['Cinsiyet'])
df['Cinsiyet'].unique()
df['Bulantı']= label_encoder.fit_transform(df['Bulantı'])
df['Bulantı'].unique()
df['ağrı_2']= label_encoder.fit_transform(df['ağrı_2'])
df['ağrı_2'].unique()
df['sıçrama']= label_encoder.fit_transform(df['sıçrama'])
df['sıçrama'].unique()
df['kayma']= label_encoder.fit_transform(df['kayma'])
df['kayma'].unique()
df['Ağrı']= label_encoder.fit_transform(df['Ağrı'])
df['Ağrı'].unique()
df['Yeme Bozukluğu']= label_encoder.fit_transform(df['Yeme Bozukluğu'])
df['Yeme Bozukluğu'].unique()
df['ateş']= label_encoder.fit_transform(df['ateş'])
df['ateş'].unique()
df['Lökosit örneği']= label_encoder.fit_transform(df['Lökosit örneği'])
df['Lökosit örneği'].unique()    
df['Teşhis']= label_encoder.fit_transform(df['Teşhis'])
df['Teşhis'].unique()


# Verisetindeki belirli değişkenleri seç
selected_columns = ["Yaş", "Cinsiyet", "Ağrı", "Yeme Bozukluğu", "Bulantı", "ağrı_2", "ateş", "ateş(1)", "sıçrama", "Lökosit örneği", "kayma", "skor", "Lökositoz ", "Nötrofil", "Lenfosit", "N/L", "CRP ", "Procalcitonin"]

# Aykırı değerleri düzelt


def sutun_ortanca_degerleri(dataframe, sutun_adi):
    return dataframe[sutun_adi].median()
def sutun_mod_degerleri(dataframe, sutun_adi):
    return dataframe.mode()
def sutun_ortalama_degeri(dataframe, sutun_adi):
    return dataframe[sutun_adi].mean()
def sutun_standart_sapma(dataframe, sutun_adi):
    return dataframe[sutun_adi].std()
def sutun_varyans(dataframe, sutun_adi):
    return dataframe[sutun_adi].var()

print("-------ortalama değerleri--------------")
for i in selected_columns:
    a = sutun_ortalama_degeri(df, i)
    print(i,a)
print("-------ortanca değerleri--------------")
for i in selected_columns:
    ortanca = sutun_ortanca_degerleri(df, i)
    print(i,ortanca)
print("-------mod değerleri--------------")
for i in selected_columns:
    m = sutun_ortalama_degeri(df, i)
    print(i,m)    
print("-------standart sapma değerleri--------------")
for i in selected_columns:
    s = sutun_standart_sapma(df, i)
    print(i,s)  
print("-------varyans değerleri--------------")
for i in selected_columns:
    v = sutun_varyans(df, i)
    print(i,v)
print("-------mod değerleri--------------")  
for i in selected_columns:
    v = sutun_varyans(df, i)
    print(i,v)

    
def iqr_aykiri_degerleri_bul(dataframe, sutun_adi, aykiri_esik=1.5):
    # Belirtilen sütunu seç
    sütun = dataframe[sutun_adi]
    # Sütunun Q1 (1. çeyrek) ve Q3 (3. çeyrek) değerlerini bul
    q1 = sütun.quantile(0.25)
    q3 = sütun.quantile(0.75)
    # IQR (Interquartile Range) hesapla
    iqr = q3 - q1
    # Aykırı değerleri belirle
    aykiri_degerler = (sütun < (q1 - aykiri_esik * iqr)) | (sütun > (q3 + aykiri_esik * iqr))
    return aykiri_degerler  

def aykiri_degerleri_sil(dataframe, sutun_adi, aykiri_esik=1.5):
    aykiri_degerler = iqr_aykiri_degerleri_bul(dataframe, sutun_adi, aykiri_esik)
    temizlenmis_dataframe = dataframe[~aykiri_degerler]
    return temizlenmis_dataframe

print("-------iqr ile aykırı değerleri bulma ve silme --------------")
for i in selected_columns:
    b = iqr_aykiri_degerleri_bul(df, i, 1.5)
    aykiri_degerleri_sil(df, i, 1.5)



def min_max_normalizasyonu(dataframe, sutun_adi):
    sütun = dataframe[sutun_adi]
    normalleştirilmis_sütun = (sütun - sütun.min()) / (sütun.max() - sütun.min())
    dataframe[sutun_adi] = normalleştirilmis_sütun
    return dataframe[sutun_adi]

def decimal_scaling(dataframe, sutun_adi, ondalik_basamak):
    sütun = dataframe[sutun_adi]
    normalize_edilmis_sütun = sütun / 10 ** ondalik_basamak
    dataframe[sutun_adi] = normalize_edilmis_sütun
    return dataframe[sutun_adi]

def z_score_normalizasyonu(dataframe, sutun_adi):
    sütun = dataframe[sutun_adi]
    normalize_edilmis_sütun = (sütun - sütun.mean()) / sütun.std()
    dataframe[sutun_adi] = normalize_edilmis_sütun
    return dataframe[sutun_adi]

dfnormminmax = df.copy()
dfnormdecimal = df.copy()
dfnormzscore = df.copy()

for i in selected_columns:
    min_max_normalizasyonu(dfnormminmax, i)

for i in selected_columns:
    decimal_scaling(dfnormdecimal, i,3)
    
for i in selected_columns:
    z_score_normalizasyonu(dfnormzscore, i)    





"""
# Yeni DataFrame'i göster
#print(veri)
label_encoder = preprocessing.LabelEncoder()
df['Cinsiyet']= label_encoder.fit_transform(df['Cinsiyet'])
df['Cinsiyet'].unique()
df['Bulantı']= label_encoder.fit_transform(df['Bulantı'])
df['Bulantı'].unique()
df['ağrı_2']= label_encoder.fit_transform(df['ağrı_2'])
df['ağrı_2'].unique()
df['sıçrama']= label_encoder.fit_transform(df['sıçrama'])
df['sıçrama'].unique()
df['kayma']= label_encoder.fit_transform(df['kayma'])
df['kayma'].unique()
df['Teşhis']= label_encoder.fit_transform(df['Teşhis'])
df['Teşhis'].unique()
df['Ağrı']= label_encoder.fit_transform(df['Ağrı'])
df['Ağrı'].unique()
df['Yeme Bozukluğu']= label_encoder.fit_transform(df['Yeme Bozukluğu'])
df['Yeme Bozukluğu'].unique()
df['ateş']= label_encoder.fit_transform(df['ateş'])
df['ateş'].unique()
df['Lökosit örneği']= label_encoder.fit_transform(df['Lökosit örneği'])
df['Lökosit örneği'].unique()

# teşhis KOLONUNUN AYIRILMASI
df1 = df.drop(['Teşhis'],axis=1)
x=df1
X=x.values
y=df['Teşhis']
Y=y.values


#MODEL İNŞASI
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state=0)



#DOĞRUSAL REGRESYON 
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)
score = lr.score(x_test, y_test)
print("Linear regression-> ACC: %", score* 100)



#burada modelleri bir listenin içerisine alıp parametreleri ile beraber tanımlıyoruz.
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('K-NN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('Gradient Boosting Classifier', GradientBoostingClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('MLPClassifier', MLPClassifier()))




#burada bir döngü vasıtasıyla tek tek bütün modelleri deneyerek sonuçları karşılaştırıyoruz. 
for name, model in models:
    model = model.fit(x_train, y_train)
    yy_pred = model.predict(x_test)
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(y_test, yy_pred)*100))
    
    #modellerin karmaşıklık matrislerini oluşturuyoruz
    cf_matrix = confusion_matrix(y_test, yy_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='icefire')
    plt.title(name)
    plt.show()
    
# ısı haritası oluşturuyoruz
sns.heatmap(df1.corr(),annot = True)
plt.show()

"""







































