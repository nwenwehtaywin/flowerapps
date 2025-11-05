import streamlit as st 
import pickle
import os

def load_model():
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


st.sidebar.title("👤 Student Info")


# Display university logo (use local file or URL)
# Example: put 'university_logo.png' in your project folder
logo_path = "images/logo.jpg"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("**Student Name:** David")
st.sidebar.markdown("**Student ID:** PU100000")




st.title("Iris Flower Classification App")
st.write("Enter the flower measurements below:")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)


flower_images = {
    'Setosa': 'images/setosa.jpg',
    'Versicolor': 'images/versicolor.jpg',
    'Virginica': 'images/virginica.jpg'
}



if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    model=load_model()
    prediction = model.predict(input_data)
    labels_names=['Setosa','Versicolor','Virginica']
    flower_name = labels_names[prediction[0]]
    print(flower_images[flower_name])
    st.success(f"The predicted flower is **{flower_name.capitalize()}**")
    st.image(flower_images[flower_name], caption=f"{flower_name}",width=400)
