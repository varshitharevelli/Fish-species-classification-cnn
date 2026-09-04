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

<img width="769" height="551" alt="Screenshot 2026-09-04 225635" src="https://github.com/user-attachments/assets/b6f2c28c-bc3a-4536-b29e-5554b34aa271" />
<img width="836" height="514" alt="Screenshot 2026-09-04 225701" src="https://github.com/user-attachments/assets/52751f2e-bf6e-4148-b498-1cf0612dee77" />
<img width="814" height="381" alt="Screenshot 2026-09-04 225647" src="https://github.com/user-attachments/assets/26392185-c0b5-4ac5-a577-cf28e22d47b8" />

<img width="685" height="431" alt="Screenshot 2026-09-04 225716" src="https://github.com/user-attachments/assets/038750a5-5dc2-48d3-a014-308e6bd71d96" />
![Uploading Screenshot 2026-09-04 225701.png…]()





---

## Future Improvements
- Increase model accuracy  
- Add more fish categories  
- Deploy as a web application  

---
