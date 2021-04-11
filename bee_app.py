#Import required packages and modules
import streamlit as st
from keras.models import model_from_json
from PIL import Image
import pandas as pd
import SessionState
import skimage
import skimage.io
import skimage.transform
import numpy as np

#Load Keras CNN
file = open("model_config.json", 'r')
model_json = file.read()
file.close()

model = model_from_json(model_json)
model.load_weights("model_weights.h5")

#Image preparation for predictions
def read_image():
    image = skimage.io.imread(state.path)
    image = skimage.transform.resize(image, (100, 100))
    return image[:,:,:3]

#Load dataframe
df = pd.read_csv('test_df_strat.csv', index_col=0)

#Load and set icon
Logo = Image.open("photos/Logo_full_Red_white_letters.png")
st.set_page_config(page_title="Bee classifier", 
                   page_icon=None, layout='centered', 
                   initial_sidebar_state='auto')
red_line = Image.open("photos/Red_line.png")

#Custom header
st.image(red_line)
col1, col2 = st.beta_columns((3, 1))
col1.markdown(
            "<h1 style='color: #CE1B28;'>Environmental Data Science Playground</h1>",
            unsafe_allow_html=True,
        )
col2.image(Logo, width=100)
st.image(red_line)
#Set title
st.title('Bee classifier')

#Descriptive text
intro = """With this little app I want to demonstrate the potential of 
convolutional neural networks (CNNs) for image classification. It makes use of 
[Kaggle's BeeImage Dataset](https://www.kaggle.com/jenny18/honey-bee-annotated-images).\n
This dataset includes labeled low-quality images of four bee subspecies:"""

st.write(intro)

#Load bee pics of the four species
col1, col2, col3, col4 = st.beta_columns(4)

pic1 = Image.open("photos/Carniolan_honey_bee.jpg")
col1.markdown("**Carniolan honey bee**")
col1.image(pic1, use_column_width=True)

pic2 = Image.open("photos/Italian_honey_bee.jpg")
col2.markdown("**Italian honey bee**")
col2.image(pic2, use_column_width=True)

pic3 = Image.open("photos/Russian_honey_bee.jpg")
col3.markdown("**Russian honey bee**")
col3.image(pic3, use_column_width=True)

pic4 = Image.open("photos/Western_honey_bee.jpg")
col4.markdown("**Western honey bee**")
col4.image(pic4, use_column_width=True)

#Descriptive text
intro2 = """The CNN (more info about its architecture and hyperparameters can 
be found on [my homepage](https://zubrod-eds.de/en/2021/04/11/bienen-klassifizieren-mit-cnns/)) had an accuracy higher than
99%, meaning that from the more than 850 bee pictures that were set aside for testing 
the CNN only 8 were classified wrong.\n
Below you can test if you can compete with the predictive power of the CNN. 
Try to guess from the pictures above which subspecies the bee on the image 
belongs to and check the respective box. You can repeat this as often as you wish 
by clicking the button below. Good luck 😊"""

st.write(intro2)

#New image button
state = SessionState.get(img=None, path=None, label=None, choice=None)
col1, col2, col3, col4, col5 = st.beta_columns(5)

if col3.button("Load a pic"):
    test_pic = df.sample(1)
    test_pic_file_name = list(test_pic.file)
    state.label = test_pic.subspecies.to_string(index = False)
    state.path = "test_images/"+test_pic_file_name[0]
    state.img = Image.open(state.path)
    
try: 
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    col3.image(state.img, width = 100)
except:
    pass

#Choice button
subspecies_options = ["Carniolan honey bee", "Italian honey bee", 
                      "Russian honey bee", "Western honey bee"]
col1, col2, col3 = st.beta_columns(3)
radio = col2.radio("Choose a subspecies:", subspecies_options)
if radio:
    state.choice = radio

#Submission and prediction
col1, col2, col3, col4, col5 = st.beta_columns(5)
check = col3.button("Submit")

all_subspecies = ['Carniolan honey bee', 'Italian honey bee', 'Russian honey bee',
       'VSH Italian honey bee', 'Western honey bee']
if check:
    try:
        prediction = np.asscalar(np.argmax(model.predict(np.expand_dims(read_image(), 
                                                                        axis=0)), 
                                           axis = 1))
        col1, col2, col3, col4, col5 = st.beta_columns((1,2,2,2,1))
        col2.write("True subspecies:")
        col3.write(state.label)
        col1, col2, col3, col4, col5 = st.beta_columns((1,2,2,2,1))
        col2.write("Model prediction:")
        col3.write(all_subspecies[prediction])
        if state.label == all_subspecies[prediction]:
            col4.markdown(":white_check_mark:")
        else:
            col4.markdown(":x:")
        col1, col2, col3, col4, col5 = st.beta_columns((1,2,2,2,1))
        col2.write("Your guess:")
        col3.write(state.choice)
        if state.label == state.choice or (state.label == "VSH Italian honey bee" 
                                           and state.choice == "Italian honey bee"):
            col4.markdown(":white_check_mark:")
        else:
            col4.markdown(":heavy_exclamation_mark:")
    except:
        pass

st.markdown('##')
st.image(red_line)
st.write("Created by Jochen Zubrod as part of the [Environmental Data Science \
         Playground](https://zubrod-eds.de/en/playground/)")
st.image(red_line)
st.write("Picture sources")
col1, col2, col3, col4 = st.beta_columns(4)
col1.write("[Carniolan honey bee](https://upload.wikimedia.org/wikipedia/commons/1/14/Carnica_bee_on_Hylotelephium_%27Herbstfreude%27.jpg)")
col2.write("[Italian honey bee](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Honeybee-27527-1.jpg/1280px-Honeybee-27527-1.jpg)")
col3.write("[Russian honey bee](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Drinking_Bee2.jpg/1280px-Drinking_Bee2.jpg)")
col4.write("[Western honey bee](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Apis_mellifera_Western_honey_bee.jpg/1024px-Apis_mellifera_Western_honey_bee.jpg)")
