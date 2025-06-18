import streamlit as st
import tensorflow as tf
import numpy as np
from skimage.transform import resize
import cv2
from PIL import Image

# Import Keras for serialization (crucial for loading models with custom layers)
import keras
from keras import saving # This is needed for the decorator to be active

# --- Define the custom preprocessing function used in your model ---
# This MUST be identical to how it's defined in your training script
@saving.register_keras_serializable(package="CustomLayers")
def mobilenet_v2_custom_preprocess_input(inputs):
    """
    A custom Lambda layer equivalent for MobileNetV2's preprocess_input.
    This allows the model to be saved and loaded properly.
    """
    return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# --- Configuration ---
# IMPORTANT: Update this path to your actual trained melanoma model
MODEL_PATH = "melanoma_models\model_finetuned_best_20250618_201223.keras" # <--- Ensure this path is correct!
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180

# Define your melanoma classes based on the dataset
MELANOMA_LABELS = [
    "Actinic keratosis",
    "Basal cell carcinoma",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Pigmented benign keratosis",
    "Seborrheic keratosis",
    "Squamous cell carcinoma",
    "Vascular lesion"
]

# --- Preprocessing Functions (adjusted for single image and model requirements) ---
# NOTE: This preprocess_image_for_model is for your *input* to the Streamlit app
# It is separate from the 'mobilenet_v2_custom_preprocess_input' layer *inside* the model.
# Make sure this function prepares the image in [0, 255] range as mobilenet_v2.preprocess_input expects.
def preprocess_image_for_model(image_array_rgb):
    """
    Applies preprocessing to a single image for the melanoma detection model.
    Expects an RGB image (NumPy array, typically uint8 [0, 255]).
    Outputs an image ready for the model (e.g., float32, in [0, 255] range before the model's Lambda layer).
    """
    # Resize the image to the model's expected input size
    # skimage.transform.resize often outputs [0,1] floats. We need [0,255] for mobilenet_v2.preprocess_input
    resized_image_0_255 = resize(image_array_rgb, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=True) * 255.0
    
    # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
    final_image = np.expand_dims(resized_image_0_255, axis=0)

    return final_image.astype(np.float32) # Ensure float32 dtype


@st.cache_resource
def load_melanoma_model():
    """Loads the pre-trained Keras model and caches it."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.warning(f"Please ensure '{MODEL_PATH}' is in the same directory as this script and it's a valid Keras model file.")
        st.info("Example: `model.save('my_melanoma_model.keras')` during training.")
        return None

# --- Rest of your Streamlit App Layout ---
st.set_page_config(
    page_title="Melanoma Cancer Detection App",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Melanoma Cancer Detection")
st.markdown("Upload a skin lesion image to check for signs of melanoma or other related conditions.")

tab1, tab2 = st.tabs(["🚀 Disease Predictor", "💡 Methodology Explained"])

with tab1:
    st.header("Upload Image for Prediction")
    st.write("Upload a single image of a skin lesion for analysis.")

    model = load_melanoma_model()

    if model:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption='Uploaded Image', use_container_width=True)
            st.write("") # Add a bit of space

            if st.button("Predict"):
                st.write("Analyzing image...")
                try:
                    # Convert PIL Image to NumPy array (RGB)
                    image_array_rgb = np.array(image_pil.convert('RGB'))

                    # Preprocess the image for the model
                    processed_image = preprocess_image_for_model(image_array_rgb)

                    # Make prediction
                    prediction = model.predict(processed_image, verbose=0)
                    predicted_class_idx = np.argmax(prediction)
                    confidence = np.max(prediction) * 100

                    predicted_disease = MELANOMA_LABELS[predicted_class_idx]

                    st.success(f"**Predicted Disease: {predicted_disease}**")
                    st.info(f"Confidence: {confidence:.2f}%")

                    st.markdown("---")
                    st.subheader("All Class Probabilities:")
                    # Display probabilities for all classes
                    # Adjusting this part to handle numpy array directly, as tf.data.Dataset.from_tensor_slices
                    # with dictionary input might not be what you intend or need here.
                    # It's better to just iterate over labels and predictions directly.
                    prob_data = []
                    for i, label in enumerate(MELANOMA_LABELS):
                        prob_data.append({'Class': label, 'Probability': prediction[0][i]})
                    st.dataframe(prob_data, hide_index=True)


                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.write("Please ensure the uploaded image is valid and try again.")
        else:
            st.info("Please upload an image to get a prediction.")

    else:
        st.error("Model could not be loaded. Please check the console for errors and ensure 'melanoma_detection_model.keras' is in the same directory.")


with tab2:
    st.header("💡 Methodology Explained")
    st.markdown("---")
    st.subheader("Problem Statement: Melanoma Cancer Detection")
    st.write(
        """
        The objective is to build a **multiclass classification model** using a custom convolutional neural network in TensorFlow to accurately detect various oncological skin diseases, including melanoma. Melanoma is a severe form of skin cancer that, if not detected early, can be deadly. It accounts for a significant portion of skin cancer-related deaths. A robust automated solution can assist dermatologists by evaluating images and flagging potential cases, thereby reducing manual effort in diagnosis and potentially saving lives.
        """
    )

    st.subheader("Dataset Characteristics")
    st.write(
        """
        The model is trained on a dataset comprising **2357 images** of both malignant and benign oncological skin diseases. This dataset was curated from the **International Skin Imaging Collaboration (ISIC)**. The images are categorized according to ISIC's classification system. Subsets for each disease type are balanced in terms of image count, with a slight dominance of melanoma and moles (Nevus) images.

        The dataset includes the following nine disease categories:
        * **Actinic keratosis**
        * **Basal cell carcinoma**
        * **Dermatofibroma**
        * **Melanoma**
        * **Nevus**
        * **Pigmented benign keratosis**
        * **Seborrheic keratosis**
        * **Squamous cell carcinoma**
        * **Vascular lesion**
        """
    )

    st.subheader("Data Preprocessing Pipeline")
    st.write(
        f"""
        Effective preprocessing is critical to prepare raw images for consumption by a deep learning model. Based on standard practices for image classification and your previous model's requirements, the following steps are typically applied to each image:

        1.  **Resizing:** All images are resized to a uniform dimension of **{IMAGE_HEIGHT}x{IMAGE_WIDTH} pixels**. This is essential because neural networks, especially convolutional layers, require fixed-size inputs.
        2.  **Normalization:** Pixel values, originally ranging from 0 to 255 (for standard image formats), are normalized. The model internally uses MobileNetV2's specific preprocessing, which scales values to **-1 to 1**.
        3.  **No Grayscale Conversion**: Unlike the gesture model, skin lesion detection usually benefits from retaining color information (RGB channels) as color is a crucial diagnostic feature. The preprocessing here assumes an RGB input of shape `({IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3)`.
        """
    )

    st.subheader("Model Architecture (Custom Convolutional Neural Network)")
    st.write(
        """
        The core of this solution is a custom **Convolutional Neural Network (CNN)** built using TensorFlow, leveraging **Transfer Learning** with a pre-trained **MobileNetV2** base.

        * **Transfer Learning with MobileNetV2:** Instead of building a CNN from scratch, the model utilizes MobileNetV2, which has been pre-trained on a massive dataset (ImageNet). This allows the model to leverage powerful, pre-learned feature extraction capabilities for general image recognition. The model is fine-tuned in two stages for optimal performance on skin lesion images.
        * **2D Convolutional Layers (`layers.Conv2D`):** These layers apply filters across the width and height of the image, learning to detect hierarchical features like edges, textures, and more complex patterns relevant to skin lesions.
        * **Pooling Layers (`layers.GlobalAveragePooling2D`):** After the MobileNetV2 base, a Global Average Pooling layer efficiently reduces the spatial dimensions of the feature maps, preparing them for the final classification layers while significantly reducing the number of parameters.
        * **Activation Functions:** Rectified Linear Unit (ReLU) is commonly used after dense layers to introduce non-linearity.
        * **Dropout Layers (`layers.Dropout`):** These layers randomly set a fraction of input units to 0 at each update during training. This prevents overfitting by making neurons less dependent on specific inputs.
        * **Dense (Fully Connected) Layers (`layers.Dense`):** These layers perform the final classification based on the extracted features.
        * **Output Layer:** A `softmax` activation function is used in the final layer to output probabilities for each of the nine disease classes. The number of units in this layer matches the number of unique disease categories in the dataset.
        """
    )

    st.subheader("Training Process")
    st.write(
        """
        The training of the melanoma detection model involves:

        * **Loss Function:** For multiclass classification with one-hot encoded labels, `categorical_crossentropy` is the standard loss function.
        * **Optimizer:** An optimizer (e.g., Adam) is used to iteratively adjust the model's weights to minimize the loss function during training.
        * **Two-Stage Fine-tuning:**
            * **Stage 1 (Feature Extraction):** The pre-trained MobileNetV2 base is initially *frozen* (its weights are not updated). Only the newly added classification layers are trained with a moderate learning rate. This efficiently adapts the model's "head" to the new task without corrupting the learned general features.
            * **Stage 2 (Fine-tuning):** The MobileNetV2 base is then *unfrozen*, and the entire model is trained end-to-end with a *very low learning rate*. This allows the pre-trained features to be subtly adjusted and optimized for the specific nuances of the skin lesion dataset, leading to potentially higher accuracy.
        * **Epochs:** The model is trained for a defined number of epochs, where each epoch represents one full pass through the entire training dataset.
        * **Batch Size:** Data is processed in batches (e.g., 32 images per batch) to make training more efficient and stable.
        * **Validation:** A portion of the dataset is set aside as a validation set. This helps in monitoring the model's performance on unseen data during training and in detecting overfitting.
        * **Callbacks:**
            * `ModelCheckpoint`: Used to save the model's weights (or the entire model) at regular intervals, often saving only the "best" model based on a monitored metric (e.g., validation loss or accuracy). The model from the epoch with the best validation accuracy is automatically saved.
            * `ReduceLROnPlateau`: Dynamically adjusts the learning rate during training. If the monitored metric (e.g., validation loss) stops improving for a certain number of epochs, the learning rate is reduced. This helps the model converge more effectively in the later stages of training.
            * `EarlyStopping`: Automatically stops training if the monitored metric (e.g., validation accuracy) does not improve for a specified number of epochs. This prevents overfitting and saves computational resources. It also `restores_best_weights`, ensuring you get the model from the best performing epoch.
        """
    )

    st.subheader("Inference with Streamlit and Image Upload")
    st.write(
        """
        This Streamlit application provides a user-friendly interface for melanoma detection:

        1.  **Image Upload:** The `st.file_uploader` widget allows users to easily upload a single skin lesion image in common formats (JPG, JPEG, PNG).
        2.  **Display Image:** The uploaded image is immediately displayed in the app for visual confirmation.
        3.  **Preprocessing:** Once the 'Predict' button is clicked, the uploaded image is converted to a NumPy array (RGB) and then preprocessed to match the input requirements of the trained model (resized to `180x180` pixels and normalized for MobileNetV2's internal preprocessing layer).
        4.  **Prediction:** The preprocessed image is fed into the loaded TensorFlow/Keras model.
        5.  **Output:** The application displays the **predicted disease class** (e.g., "Melanoma", "Nevus", "Basal cell carcinoma") and the model's **confidence score** for that prediction. It also presents the probabilities for all other classes, providing a comprehensive overview.
        """
    )