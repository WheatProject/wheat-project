import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.ensemble import RandomForestRegressor

st.write('''
# Wheat Project
## Application web simple pour la prévision des matières premières
Cette application prédit la variation des prix pour une matière première donnée.
''')
st.sidebar.header("Paramètres d'entrée")


def user_input():
    Matieres_premieres = st.sidebar.selectbox('Matière première', ['Blé', 'Maïs', 'Viande'])
    Horizon_de_prevision = st.sidebar.slider('Horizon de prévision (mois)', 1, 24, 6)
    Annee_de_prevision = st.sidebar.slider('Année de prévision', 2010, 2015, 2020)
    modele_prediction = st.sidebar.selectbox('Modèle de prediction', ['Machine Learning', 'Deep Learning'])
    prodcuction = st.sidebar.slider('Production', 0.0, 49678.0, 995356.0)

    data = {'Matiere_premiere': Matieres_premieres,
            'Horizon_de_prevision': Horizon_de_prevision,
            'modele_prediction': modele_prediction,
            'Annee_de_prevision': Annee_de_prevision,
            'prodcuction ': prodcuction
            }
    matieres_parametres = pd.DataFrame(data, index=[0])
    return matieres_parametres


df = user_input()
st.subheader('Paramètres souhaités')
st.write(df)

extract_matprem = df['Matiere_premiere']
choix_matprem = extract_matprem.iloc[0]
extract_horizon = df['Horizon_de_prevision']
choix_horizon = extract_horizon.iloc[0]

if choix_matprem == 'Blé':

    df = pd.read_csv("FileBleUS.csv", sep=';')
    df = df.fillna(0)

    # Preprocessing
    sc = MinMaxScaler(feature_range=(0, 1))
    data_sc = sc.fit_transform(df[['Dernier']])

    # Séparation en input et output
    X_data = []
    y_data = []
    look_back = 12
    for i in range(len(data_sc) - look_back):
        X_data.append(data_sc[i:(i + look_back), 0])
        y_data.append(data_sc[i + look_back, 0])

    X_data, y_data = np.array(X_data), np.array(y_data)

    # Redimensionnement
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

    # Séparation en jeu d'entraînement et jeu de test
    train_size = int(len(X_data) * 0.7)
    test_size = len(X_data) - train_size

    X_train = X_data[0:train_size, :]
    X_test = X_data[train_size:len(X_data), :]
    y_train = y_data[0:train_size]
    y_test = y_data[train_size:len(X_data)]

    # Construction du RNN
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # return_sequences=True pour avoir un retour de la sortie vers l'entrée
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    # Ajout de la couche de sortie
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32)

    y_pred = model.predict(X_test)


    # Affichage des résultats
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(100.0 * y_pred[0:choix_horizon + 1] / y_pred[0])
    ax.set_title("Prévision des prix du blé")
    ax.set_xlabel('Horizon de prévision (mois)')
    ax.set_ylabel('Indice (base 100)')
    st.pyplot(fig)

elif choix_matprem == 'Maïs':

    df = pd.read_csv('FinalMaizeIndex.csv', sep=';')
    data_maize = df[['MaizeIndex']].values

    # Preprocessing
    sc = MinMaxScaler(feature_range=(0, 1))
    data_sc = sc.fit_transform(data_maize)

    # Standardisation avec StandardScaler
    scaler2 = StandardScaler()
    data_sc2 = scaler2.fit_transform(data_maize)

    # Séparation en input et output
    X_data = []
    y_data = []
    look_back = 12
    for i in range(len(data_sc) - look_back):
        X_data.append(data_sc[i:(i + look_back), 0])
        y_data.append(data_sc[i + look_back, 0])
    X_data, y_data = np.array(X_data), np.array(y_data)

    # Redimensionnement
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

    # Séparation en jeu d'entraînement et jeu de test
    train_size = int(len(X_data) * 0.7)
    test_size = len(X_data) - train_size
    X_train = X_data[0:train_size, :]
    X_test = X_data[train_size:len(X_data), :]
    y_train = y_data[0:train_size]
    y_test = y_data[train_size:len(X_data)]

    # Construction du RNN
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # return_sequences=True pour avoir un retour de la sortie vers l'entrée
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    # Ajout de la couche de sortie
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32)
    y_pred = model.predict(X_test)

    # Affichage des résultats
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(100.0 * y_pred[0:choix_horizon + 1] / y_pred[0])
    ax.set_title("Prévision des prix du maïs")
    ax.set_xlabel('Horizon de prévision (mois)')
    ax.set_ylabel('Indice (base 100)')
    st.pyplot(fig)

elif choix_matprem == 'Viande':

    df = pd.read_csv('FAO_food_price_indices_clean.csv', sep=';')
    data_meat = df[['Meat']].values

    # Preprocessing
    sc = MinMaxScaler(feature_range=(0, 1))
    data_sc = sc.fit_transform(data_meat)

    # Séparation en input et output
    X_data = []
    y_data = []
    look_back = 12
    for i in range(len(data_sc) - look_back):
        X_data.append(data_sc[i:(i + look_back), 0])
        y_data.append(data_sc[i + look_back, 0])
    X_data, y_data = np.array(X_data), np.array(y_data)

    # Redimensionnement
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

    # Séparation en jeu d'entraînement et jeu de test
    train_size = int(len(X_data) * 0.7)
    test_size = len(X_data) - train_size

    X_train = X_data[0:train_size, :]
    X_test = X_data[train_size:len(X_data), :]
    y_train = y_data[0:train_size]
    y_test = y_data[train_size:len(X_data)]

    # Construction du RNN

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # return_sequences=True pour avoir un retour de la sortie vers l'entrée
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))

    # Ajout de la couche de sortie
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32)
    y_pred = model.predict(X_test)

    # Affichage des résultats
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(100.0*y_pred[0:choix_horizon+1]/y_pred[0])
    ax.set_title("Prévision des prix de la viande")
    ax.set_xlabel('Horizon de prévision (mois)')
    ax.set_ylabel('Indice (base 100)')
    st.pyplot(fig)

st.markdown("""
    <iframe width="600" height="606" src="https://app.powerbi.com/groups/me/reports/2c09f392-36a7-4263-b716-cbb391e09268?ctid=373016f8-79a9-4eed-80d2-100ce948d960&pbi_source=linkShare&bookmarkGuid=81d813ce-dd5d-4c60-8c97-de830b922272" frameborder="0" style="border:0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)
