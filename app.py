import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from tensorflow.keras.optimizers import Adam  # Importing Adam optimizer

# Skin disease classes
SKIN_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'
}

# Load model function with optimizer handling
def load_model_from_file():
    # Load the entire model without compiling
    model = load_model('modelnew.h5', compile=False)  # Disabling compilation
    model.compile(optimizer=Adam(learning_rate=0.000625), loss='categorical_crossentropy', metrics=['accuracy'])  # Manually compile with the correct optimizer
    return model

# Streamlit app interface
def main():
    st.title('Skin Disease Prediction')
    st.markdown("Upload an image of skin to predict its disease.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Load the model
        model = load_model_from_file() 

        # Process the image for prediction
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img = np.array(img)
        img = img.reshape((1, 224, 224, 3))
        img = img / 255
        
        # Prediction
        prediction = model.predict(img)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred] * 100

        # Display results
        st.write(f"Prediction: {disease}")
        st.write(f"Affected Percentage: {accuracy:.2f}%")
        
        # Clear session
        K.clear_session()

if __name__ == "__main__":
    main()