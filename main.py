import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.write('''
# App Simple pour la prévision des matieres premières
Cette application prédit la variation des prix et des stocks
''')
st.sidebar.header("Les parametres d'entrée")
def user_input():
    Matieres_premieres=st.sidebar.selectbox('La matiere premiere ',['Blé','Mais','Viande'])
    Horizon_de_prevision=st.sidebar.slider('Horizon de prévision',1.0,3.0,12.0)
    Annee_de_prevision = st.sidebar.slider('Année de prévision', 2010, 2015, 2020)
    modele_prediction=st.sidebar.selectbox('Le modèle de prediction',['Machine Learning', 'Deep Learning'])
    prodcuction = st.sidebar.slider('Production', 0.0,49678.0,995356.0 )
    #petal_width=st.sidebar.slider('La largeur du Petal',0.1,2.5,1.3)
    data={'Matieres_premieres':Matieres_premieres,
    'Horizon_de_prevision':Horizon_de_prevision,
    'modele_prediction':modele_prediction,
     'Annee_de_prevision':Annee_de_prevision,
     'prodcuction ':prodcuction
          }
    matieres_parametres=pd.DataFrame(data,index=[0])
    return matieres_parametres

df=user_input()
st.subheader('on souhaite ')
st.write(df)
Matières_premieres=['Blé','Mais','Viandes']

if Matières_premieres =='Blé':
    
        Wheat=pd.read_csv("fichierWheatprix.csv", sep = ';')
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        dfe=Wheat
        dfe.Area=le.fit_transform(dfe.Area)
        dfe.Indice_PrixProd=le.fit_transform(dfe.Indice_PrixProd)
        y=dfe.Indice_PrixProd
        dfe1=dfe.fillna(0)
        #Normalisation données
        X=dfe1[[ 'Area_harvested_Wheat', 'Yield_Wheat',
               'Production_Wheat']]#'Area','Year',
        y=dfe1['Indice_PrixProd']
        scaler= StandardScaler()
        X_scaled=scaler.fit_transform(X)
        y_scaled=scaler.fit_transform(y.array.reshape(-1,1))
        X_scaled
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled, test_size=0.2, random_state=0)
        clf=RandomForestRegressor()
        clf.fit(X_train,y_train)
        prediction=clf.predict(df)
        st.subheader("La prédiction des prix est:")
        st.write(y_train[prediction])
