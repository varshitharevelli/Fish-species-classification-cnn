# Fish Species Classification using CNN

## Project Overview
This project classifies fish images into different species using a Convolutional Neural Network built with PyTorch. The system takes an input image and predicts the fish category.

---

## Features
- Image classification using CNN  
- PyTorch trained model (.pth file)  
- Flask-based web application  
- Upload image and get prediction result  

---

## Project Structure

Fish-species-classification-cnn/
│
├── modelTrain.ipynb
├── app.py
├── requirements.txt
├── fish_model_jupyter.pth
├── style.css
├── README.md
│
├── templates/
│ ├── index.html
│ └── result.html


---

## Dataset
The dataset contains images of different fish species used to train the CNN model.
Note: Full dataset is not uploaded due to large size.

---

## Installation

1. Clone the repository:

git clone https://github.com/varshitharevelli/Fish-species-classification-cnn.git

cd Fish-species-classification-cnn


2. Install dependencies:

pip install -r requirements.txt


---

## How to Run

1. Start the Flask application:

python app.py


2. Open your browser:

http://127.0.0.1:5000/


3. Upload a fish image and view prediction result.

---

## Model Details
- Model Type: Convolutional Neural Network (CNN)  
- Framework: PyTorch  
- Model File: fish_model_jupyter.pth  

---

## Output
Add screenshots here:
- Upload page  
- Prediction result  

---

## Future Improvements
- Increase model accuracy  
- Add more fish categories  
- Deploy as a web application  

---
