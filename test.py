
import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from gtts import gTTS
import os
from twilio.rest import Client
import sounddevice as sd

# Function to load the model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('model/audio_classification.hdf5')
    return model

# Set page title and favicon
st.set_page_config(
    page_title="Audio Classification Web App",
    page_icon="🔊"
)

# Loading the model
with st.spinner('Model is being loaded..'):
    model = load_model()

# Set app title and description
st.title("Audio Classification Web App")
st.write("Upload an audio file or record audio and the app will predict its class.")

# Sidebar option to choose between file upload and audio recording
option = st.sidebar.selectbox("Choose Input Method", ["Upload File", "Record Audio"])

# Initialize label encoder
labelencoder = LabelEncoder()

# Assuming you have a list of target labels, let's call it `target_labels`
target_labels = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner',
       'street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer',
       'drilling', 'Bell', 'Saurav', 'Sumit', 'Chirag ', 'Chirag']
labelencoder.fit(target_labels)

# Define function to import and predict
def import_and_predict(uploaded_file, model, labelencoder):
    audio, sample_rate = librosa.load(uploaded_file , res_type='soxr_hq')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_label = model.predict(mfccs_scaled_features)
    prediction_class = labelencoder.inverse_transform(np.argmax(predicted_label, axis=1))
    return prediction_class[0]

# Function to send SMS
def send_sms(class_name):
    account_sid = 'AC3c9f03c6cd28df08212e81683d3b8b26'
    auth_token = '35307fce148840ea36a85ff9f3d5fa21'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
      from_='+12512209585 ',
      body=f'The predicted class is {class_name}',
      to='+919594541002'
    )

    print(message.sid)

 


if option == "Upload File":
    # Upload file and classify
    uploaded_file = st.sidebar.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Predict the class and convert prediction to speech
        predicted_class = import_and_predict(uploaded_file, model, labelencoder)
        tts = gTTS(text=f"  {predicted_class}", lang='en')
        tts.save("prediction.mp3")

        # Play the audio
        os.system("start prediction.mp3")

        # Display the predicted class with enhanced styling
        st.markdown(f"<h1 style='text-align: center; color: blue;'>Predicted Class: {predicted_class}</h1>", unsafe_allow_html=True)

        # Send SMS
        message_sid = send_sms(predicted_class)
        st.write(f"SMS Sent. Message SID: {message_sid}")

elif option == "Record Audio":
    # Define the sample rate (sr) for audio processing
    sr = 44100  # You may need to adjust this based on your specific use case

    
# Function to record audio
    def record_audio(duration):
        try:
            audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            return audio.flatten()
        except Exception as e:
            print(f"Error occurred while recording audio: {e}")
            return None



    # Record audio button
    duration = 10  # Adjust the recording duration as needed
    audio_data = record_audio(duration)
    print("Audio recorded.")

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfcc_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    # Make a prediction using your model
    predicted_label = model.predict(mfccs_scaled_features)

    # Assuming `labelencoder` is already defined
    prediction_class = labelencoder.inverse_transform(np.argmax(predicted_label, axis=1))[0]

    # Convert prediction to speech
    tts = gTTS(text=f" {prediction_class}", lang='en')
    tts.save("prediction.mp3")

    # Play the audio
    os.system("start prediction.mp3")

    # Send SMS
    message_sid = send_sms(prediction_class)
    print(f"SMS Sent. Message SID: {message_sid}")
