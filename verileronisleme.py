#kutuphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


veriler = pd.read_csv("veriler.csv")1

Imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
Yas = veriler.iloc[:,3:4].values
Imputer = Imputer.fit(Yas[:,0:1])
Yas[:,0:1] = Imputer.transform(Yas[:,0:1])


ulke = veriler.iloc[:,0:1].values
cinsiyet = veriler.iloc[:,4:5].values


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])



from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories="auto", handle_unknown="ignore")
ulke = ohe.fit_transform(ulke).toarray()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

boyKiloYas = veriler.iloc[:,1:4]
ulke = pd.DataFrame(data=ulke, index=range(22), columns=["tr","us","fr"])
cinsiyet = pd.DataFrame(data=cinsiyet, index=range(22),columns=["e","k"])


s1 = pd.concat([ulke,boyKiloYas],axis=1)
YENI_VERILER = pd.concat([s1,cinsiyet],axis=1)
YENI_VERILER = YENI_VERILER.iloc[:,0:7]



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(YENI_VERILER.iloc[:,0:6],YENI_VERILER.iloc[:,6:7],test_size=0.33)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test5)
