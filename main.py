import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.write('''
# App Simple pour la prévision des matieres premières
Cette application prédit la variation des prix et des stocks
''')
st.sidebar.header("Les parametres d'entrée")
def user_input():
    Matières_premieres=st.sidebar.selectbox('La matiere premiere ',['Blé','Mais','Viande'])
    Horizon_de_prévision=st.sidebar.slider('Horizon de prévision',1.0,3.0,12.0)
    modele_prediction=st.sidebar.selectbox('Le modèle de prediction',['Machine Learning', 'Deep Learning'])
    #petal_width=st.sidebar.slider('La largeur du Petal',0.1,2.5,1.3)
    data={'Matières_premieres':Matières_premieres,
    'Horizon_de_prévision':Horizon_de_prévision,
    'modele_prediction':modele_prediction,
    }
    matieres_parametres=pd.DataFrame(data,index=[0])
    return matieres_parametres

df=user_input()
st.subheader('on veut trouver la catégorie de cette fleur')
st.write(df)
