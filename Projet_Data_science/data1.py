import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,chi2



from sklearn.metrics import confusion_matrix







df=pd.read_csv("diabetes.csv")





df_yes=df[df['Outcome']==1]
df_non=df[df['Outcome']==0]

x=df.drop('Outcome',axis=1)
y=df['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model=KNeighborsClassifier()

model.fit(x_train,y_train)
c=model.score(x_test,y_test)

# print('le modele donne \n ')
# print('une efficacité de ',c*100,'% \n')
# print('une erreur de',100-c*100,'%')


y_pred=model.predict(x_test)

X=StandardScaler().fit_transform(x)
Y=y


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

model=KNeighborsClassifier(19)

model.fit(X_train,Y_train)
b=model.score(X_test,Y_test)
# print('le modele donne \n ')
# print('une efficacité de ',b*100,'% \n')
# print('une erreur de',100-b*100,'%')




modele= mode=make_pipeline(StandardScaler(),PolynomialFeatures(7),KNeighborsClassifier(10))
modele.fit(x_train,y_train)
d=modele.score(x_test,y_test)
# print('le modele donne \n ')
# print('une efficacité de ',d*100,'% \n')
# print('une erreur de',100-d*100,'%')


chi2(x,y)

df.drop(['BloodPressure','SkinThickness','DiabetesPedigreeFunction'],axis=1,inplace=True)

x=df.drop('Outcome',axis=1)
y=df['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print(df['Age'].max())

modele=make_pipeline(StandardScaler(),PolynomialFeatures(8),KNeighborsClassifier(6))
modele.fit(x_train,y_train)
d=score=modele.score(x_test,y_test)
# print('le modele donne \n ')
# print('une efficacité de ',d*100,'% \n')
# print('une erreur de',100-d*100,'%')

def Les_parametre():
    Pregnancies=st.sidebar.slider('Pregnancies',df['Pregnancies'].min(),df['Pregnancies'].max())
    Glucose=st.sidebar.slider('Glucose',df['Glucose'].min(),df['Glucose'].max())
    Insulin=st.sidebar.slider('Insulin',df['Insulin'].min(),df['Insulin'].max())
    BMI=st.sidebar.slider('BMI',df['BMI'].min(),df['BMI'].max())
    Age=st.sidebar.number_input('Age',0,100,50,1)
    
    listeDict=[{'Pregnancies':Pregnancies,'Glucose':Glucose,'Insulin':Insulin,'BMI':BMI,'Age':Age,}]
    
    data=pd.DataFrame(listeDict,index=[0])
    return data


st.sidebar.header('les Paramettre')





# sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)







st.write('''
          # Know Your Diabetes Risk
          
Découvrez votre risque de diabète en quelques clics ! Entrez vos informations et
obtenez une évaluation personnalisée. Prenez soin de votre santé avec "Know Your Diabetes Risk".
Cette version plus courte conserve l'essentiel de l'application et
présente brièvement ses avantages. Vous pouvez toujours ajuster la description
en fonction de la place disponible et des éléments que vous souhaitez mettre en avant. L'objectif est de donner
aux utilisateurs un aperçu rapide de l'utilité de l'application tout en les incitant à 
l'utiliser pour obtenir leur évaluation de risque de diabète.
          ''')

df=Les_parametre()

st.subheader('Vos données')
st.write(df)


prediction=modele.predict(df)

st.subheader('Prediction')

if prediction[0]==1:
   st.write('Vous avez de fortes chances d\'avoir le diabète') 
else:
    st.write('Vous avez de faibles chances d\' avoir le diabète') 

st.subheader('Probabilité de Prediction')

st.write(modele.predict_proba(df))