import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('iris_pipeline_model.pkl','rb'))
# df = pickle.load(open('df.pkl','rb'))

st.title("Species of Iris")

# weight
Sepal_length = st.number_input('Length of the sepal')

# Touchscreen
Sepal_width = st.number_input('width of the sepal')


petal_length = st.number_input('Length of the petal')

petal_width = st.number_input('width of the petal')



if st.button('Predict specie'):
    # query
    ppi = None

    query = np.array([Sepal_length,Sepal_width,petal_length,petal_width])

    query = query.reshape(1,4)
    value  = int(pipe.predict(query)[0])
    flower_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    names = list(flower_mapping.keys())[list(flower_mapping.values())[value]]
    st.title("The predicted species is " + names)