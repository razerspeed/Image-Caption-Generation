import streamlit as st
import time
import client


app_formal_name = "Image Caption"
# Start the app in wide-mode
st.set_page_config(
    layout="wide", page_title=app_formal_name,
)


st.title("Image Captioning using CNN and Transformer ")

st.markdown('#### In this project we have implemented the encoder decoder network for image caption generation')
st.markdown('#### The CNN model act as Encoder and the Transformer Decoder act as an Decoder')
st.text('')
st.text('')

from PIL import Image

image = Image.open('cnn_transformer.jpg')
st.image(image, use_column_width=True)

st.markdown('#### How the model is deployed ?')
st.markdown('#### The encoder part i.e. CNN is deployed as a REST service in SERVER 1')
st.markdown('#### The decoder part i.e. Transformers Decoder  is deployed as a REST service in SERVER 2')

html_string = "<br><br>"

st.markdown(html_string, unsafe_allow_html=True)
image = Image.open('diagram.jpg')
image=image.resize((500,500))
st.image(image,caption='API Diagram')


def call_function(a):
    print('value of a',a)
    prediction=client.predict(a)
    print('prediction',prediction)
    st.markdown('#### Generated Caption ')
    st.markdown('### '+prediction)


html_string = "<br><br>"

st.markdown(html_string, unsafe_allow_html=True)
col1, col2 = st.beta_columns(2)


a=None
with col1:
    st.markdown('### Image 1')
    # st.header("Image 1")
    st.image(Image.open('1.jpg'), caption='Predicted output')
    if st.button('Select this image ', key=1):
        a='1.jpg'
        call_function(a)

with col2:
    st.markdown('### Image 2')
    st.image(Image.open('2.jpg'), caption='Predicted output' )
    if st.button('Select this image ',key=2):
        a='2.jpg'
        call_function(a)

col1, col2 = st.beta_columns(2)

a=None
with col1:

    st.markdown('### Image 3')
    st.image(Image.open('3.jpg'), caption='Predicted output')
    if st.button('Select this image ', key=3):
        a='3.jpg'
        call_function(a)

with col2:
    st.markdown('### Image 4')
    st.image(Image.open('4.jpg'), caption='Predicted output' )
    if st.button('Select this image ',key=4):
        a='4.jpg'
        call_function(a)

html_string = "<br><br>"

st.markdown(html_string, unsafe_allow_html=True)
# st.markdown('## Upload a photo to generate caption')
# uploaded_file = st.file_uploader("Upload an image...", type="jpg")
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     # print(image)
#     st.image(image, caption='uploaded image')
#     image.save('upload.jpg')
#     call_function('upload.jpg')
