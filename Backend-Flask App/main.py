import pandas as pd 
import numpy as np 
from collections import Counter as c 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression 
import pickle 


data = pd.read_csv(r"C:/Users\DEEKSHITH\Downloads/mini_project_akki/mpcopy/Chronic-Kidney-Disease-Detection-Using-Machine-Learning-main/Datasets/chronickidneydisease.csv")
data.head(10)
data.drop(['id'],axis=1,inplace=True)
data.columns
data.columns = ['age','blood_pressure','specific_gravity','albumin','sugar','red_blood_cells','pus_cell',
                'pus_cell_clumps','bacteria','blood glucose random','blood_urea','serum_creatinine','sodium','potassium','hemoglobin','packed_cell_volume','white_blood_cell_count','red_blood_cell_count','hypertension','diabetesmellitus','coronary_artery_disease','appetite','pedal_edema','anemia','class']
data.columns
data.info()
data['class'].unique()


data['class'] = data['class'].replace("ckd\t","ckd")
data['class'].unique()

catcols = set(data.dtypes[data.dtypes=='O'].index.values)
print(catcols)
for i in catcols:
    print("Columns :",i)
    print(c(data[i])) 
    print('*'*120+'\n')

catcols.remove('red_blood_cell_count')
catcols.remove('packed_cell_volume')
catcols.remove('white_blood_cell_count')
print(catcols)

contcols = set(data.dtypes[data.dtypes!='O'].index.values)
print(contcols)
for i in contcols:
    print("Continous Columns:",i)
    print(c(data[i]))
    print('*'*120+'\n')

contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')
print(contcols)

contcols.add('red_blood_cell_count')
contcols.add('packed_cell_volume')
contcols.add('white_blood_cell_count')
print(contcols)

catcols.add('specific_gravity')
catcols.add('albumin')
catcols.add('sugar')
print(catcols)

data['coronary_artery_disease']=data.coronary_artery_disease.replace('\tno','no')
c(data['coronary_artery_disease'])
data['diabetesmellitus']=data.diabetesmellitus.replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
c(data['diabetesmellitus'])

data.isnull().any()

data.isnull().count()
data.packed_cell_volume = pd.to_numeric(data.packed_cell_volume , errors='coerce')
data.red_blood_cell_count = pd.to_numeric(data.red_blood_cell_count  , errors='coerce')
data.white_blood_cell_count = pd.to_numeric(data.white_blood_cell_count , errors='coerce')

data['blood glucose random'].fillna(data['blood glucose random'].mean(),inplace = True)
data['blood_pressure'].fillna(data['blood_pressure'].mean(),inplace = True)
data['blood_urea'].fillna(data['blood_urea'].mean(),inplace = True)
data['hemoglobin'].fillna(data['hemoglobin'].mean(),inplace = True)
data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(),inplace = True)
data['potassium'].fillna(data['potassium'].mean(),inplace = True)
data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(),inplace = True)
data['serum_creatinine'].fillna(data['serum_creatinine'].mean(),inplace = True)
data['sodium'].fillna(data['sodium'].mean(),inplace = True)
data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(),inplace = True)

data['age'].fillna(data['age'].mode()[0], inplace=True)
data['specific_gravity'].fillna(data['specific_gravity'].mode()[0], inplace=True)
data['albumin'].fillna(data['albumin'].mode()[0], inplace=True)
data['sugar'].fillna(data['sugar'].mode()[0], inplace=True)
data['red_blood_cells'].fillna(data['red_blood_cells'].mode()[0], inplace=True)
data['pus_cell'].fillna(data['pus_cell'].mode()[0], inplace=True)
data['pus_cell_clumps'].fillna(data['pus_cell_clumps'].mode()[0], inplace=True)
data['bacteria'].fillna(data['bacteria'].mode()[0], inplace=True)
data['blood glucose random'].fillna(data['blood glucose random'].mode()[0], inplace=True)
data['hypertension'].fillna(data['hypertension'].mode()[0], inplace=True)
data['diabetesmellitus'].fillna(data['diabetesmellitus'].mode()[0], inplace=True)
data['coronary_artery_disease'].fillna(data['coronary_artery_disease'].mode()[0], inplace=True)
data['appetite'].fillna(data['appetite'].mode()[0], inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0], inplace=True)
data['anemia'].fillna(data['anemia'].mode()[0], inplace=True)
data['class'].fillna(data['class'].mode()[0], inplace=True)
data.isna().sum()

from sklearn.preprocessing import LabelEncoder

for i in catcols:  
    print("LABEL ENCODING OF :",i)
    LEi = LabelEncoder()
    print(c(data[i])) 
    data[i] = LEi.fit_transform(data[i]) 
    print(c(data[i]))  
    print('*'*100)

features_name = ['blood_urea','blood glucose random','coronary_artery_disease','anemia','pus_cell',
    'red_blood_cells','diabetesmellitus','pedal_edema']
x = pd.DataFrame(data, columns = features_name)
y = pd.DataFrame(data, columns = ['class'])
print(x.shape)
print(y.shape)
data.isna().sum()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(solver='lbfgs', max_iter=1000)

lgr.fit(x_train.values,y_train.values.ravel())
y_pred = lgr.predict(x_test)

def lgpredict(p1,p2,p3,p4,p5,p6,p7,p8):
    x_new=np.array([p1,p2,p3,p4,p5,p6,p7,p8])
    x_new=x_new.reshape(1,8)
    ll_predict= lgr.predict(x_new)
    y_pred=ll_predict
    return int(ll_predict)

c(y_pred)
accuracy_score(y_test,y_pred)

conf_mat = confusion_matrix(y_test,y_pred)
conf_mat
pickle.dump(lgr, open('CKD.pkl','wb'))
