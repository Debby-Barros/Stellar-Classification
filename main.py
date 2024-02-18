import streamlit as st
import joblib
import pandas as pd
from objects import galaxy, qso, star, astronomical_objects
from PIL import Image
import warnings


# Desativa todas as warnings
warnings.filterwarnings("ignore")

# carregando o modelo pre-treinado
loaded_model_RF = joblib.load('random_forest_model.pkl')

# carregando o dataset
df = pd.read_csv('stellar.csv')


# streamlit
def main():
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://w.forfun.com/fetch/df/dfcb69a695dddebdf6412a4ab25a796f.jpeg");
        background-color: rgba(0, 0, 0, 0.5);
        background-size: cover;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title('Classificação de objetos astronômicos')

    # coletando as features dos usuários
    st.sidebar.subheader("Testando o modelo random forest:")
    
    alpha = st.sidebar.slider("alpha", df['alpha'].min(), df['alpha'].max())
    delta = st.sidebar.slider("delta", df['delta'].min(), df['delta'].max())
    u = st.sidebar.slider("Ultraviolet filter", df['u'].min(), df['u'].max())
    g = st.sidebar.slider("Green filter", df['g'].min(), df['g'].max())
    r = st.sidebar.slider("Red filter", df['r'].min(), df['r'].max())
    i = st.sidebar.slider("Near Infrared filter", df['i'].min(), df['i'].max())
    z = st.sidebar.slider("Infrared filter", df['z'].min(), df['z'].max())
    cam_col = st.sidebar.slider("Camera", df['cam_col'].min(), df['cam_col'].max())
    spec_obj_ID = st.sidebar.slider("spectroscopic object ID", df['spec_obj_ID'].min(), df['spec_obj_ID'].max())
    redshift = st.sidebar.slider("redshift", df['redshift'].min(), df['redshift'].max())
    plate = st.sidebar.slider("plate", df['plate'].min(), df['plate'].max())
    MJD = st.sidebar.slider("MJD", df['MJD'].min(), df['MJD'].max())

    # realizando a previsão
    test_data = [[alpha, delta, u, g, r, i, z, cam_col, spec_obj_ID, redshift, plate, MJD]]

    prediction = loaded_model_RF.predict(test_data)

    if st.sidebar.button('Verificar'):
        if prediction[0] == 0: # galaxy
            st.subheader(f"A classe prevista é: GALÁXIA")
            st.write(galaxy)
            img1 = Image.open(astronomical_objects['galaxy'])
            st.image(img1, caption='galáxia', use_column_width=True)

        elif prediction[0] == 1: # qso
            st.subheader(f"A classe prevista é: QUASAR")
            st.write(qso)
            img2 = Image.open(astronomical_objects['qso'])
            st.image(img2, caption='quasar', use_column_width=True)

        else: # star
            st.subheader(f"A classe prevista é: ESTRELA")
            st.write(star)
            img3 = Image.open(astronomical_objects['star'])
            st.image(img3, caption='estrela', use_column_width=True)


if __name__ == '__main__':
    main()