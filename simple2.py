import streamlit as st
import plotly_express as px
import numpy as np
import pickle
# import joblib
# import dill

def main():
    with open('model.pkl','rb') as file:
        model = pickle.load(file)
    
    st.title('Webpage prediksi bunga iris')
    
    # sl = st.number_input(label="Masukkan Sepal Length", value=5.2, min_value=0.0, max_value=8.0, step=0.1)
    # sw = st.number_input(label="Masukkan Sepal Width", value=3.2, min_value=0.0, max_value=8.0, step=0.1)
    # pl = st.number_input(label="Masukkan Petal Length", value=1.2 , min_value=0.0, max_value=8.0, step=0.1)
    # pw = st.number_input(label="Masukkan Petal Width", value=0.2 , min_value=0.0, max_value=8.0, step=0.1)
    
    sl = st.slider(label="Masukkan Sepal Length", value=5.2, min_value=0.0, max_value=8.0, step=0.1)
    sw = st.slider(label="Masukkan Sepal Width", value=3.2, min_value=0.0, max_value=8.0, step=0.1)
    pl = st.slider(label="Masukkan Petal Length", value=1.2 , min_value=0.0, max_value=8.0, step=0.1)
    pw = st.slider(label="Masukkan Petal Width", value=0.2 , min_value=0.0, max_value=8.0, step=0.1)
    
    if st.button(label='Click to Predict'):
        user_data = np.array([sl,sw,pl,pw]).reshape(1,-1)
        # st.write(user_data)
        prediction = model.predict(user_data)[0]
        st.sidebar.header('Hasil Prediksi dari input adalah: ')
        # st.write(prediction)
        if prediction ==1:
            st.write('Iris-Setosa')
        elif prediction ==2:
            st.write('Iris-Versicolor')
        elif prediction == 3:
            st.write('Iris-Virginica')
        

if __name__=='__main__':
    main()