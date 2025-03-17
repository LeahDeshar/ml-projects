import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

sonar_data = pd.read_csv('data.csv',header=None)

sonar_data.head()

sonar_data.shape

sonar_data.describe()

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

X = sonar_data.drop(columns=60,axis=1)
y = sonar_data[60]

X_train,X_test,Y_train,Y_test = train_test_split(X,y , test_size=0.1,stratify=y,random_state = 42)

X.shape, X_train.shape , X_test.shape

y.shape, Y_train.shape , Y_test.shape

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled,X_test_scaled

model = LogisticRegression(C=1.0, solver='liblinear', random_state=1)
model.fit(X_train_scaled,Y_train)



x_train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(x_train_pred,Y_train)
train_accuracy

x_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(x_test_pred,Y_test)
test_accuracy

input_data = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)

input_data_as_numpy_array = np.asarray(input_data)


data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(data_reshaped)

prediction

if prediction[0] == 'R':
  print("ROCK")
elif prediction[0] == 'M':
  print("MINE")
else:
  print("Invalid")

