from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import io
import base64
import os
from werkzeug.security import generate_password_hash, check_password_hash
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ---- Global Variables ----
model = None
classes = []
transform = None

# ---- Simple User Store ----
users = {}


# ---------------------- MODEL LOADING ----------------------
def load_model_once():
    global model, classes, transform
    if model is None:
        try:
            MODEL_PATH = r"C:\fish_classification\fish_model_jupyter.pth"  # ✅ FIX

            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            classes = checkpoint["classes"]
            num_classes = len(classes)

            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes
            )

            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            print("Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")

# ---------------------- AUTH ROUTES ----------------------
@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if email in users:
            flash('Email already registered. Please login.', 'warning')
            return redirect(url_for('login'))

        users[email] = {
            'name': name,
            'password': generate_password_hash(password)
        }
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = users.get(email)
        if user and check_password_hash(user['password'], password):
            session['user'] = user['name']
            flash(f'Welcome, {user["name"]}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ---------------------- CLASSIFIER ROUTES ----------------------
@app.route('/index')
def index():
    if 'user' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized access. Please login.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        load_model_once()
        if model is None:
            return jsonify({'error': 'Model not available. Please train the model first.'})

        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top3_probs, top3_indices = torch.topk(probabilities, 3)

        results = []
        for idx, prob in zip(top3_indices, top3_probs):
            results.append({
                'class': classes[idx],
                'confidence': f'{prob.item():.2%}'
            })

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'predictions': results,
            'image': f'data:image/jpeg;base64,{img_str}'
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})


@app.route('/classes')
def get_classes():
    load_model_once()
    return jsonify({'classes': classes})


if __name__ == '__main__':
    MODEL_PATH = r"C:\fish_classification\fish_model_jupyter.pth"

    if os.path.exists(MODEL_PATH):
        load_model_once()
    else:
        print("⚠️ Warning: Model file not found. Please train the model first.")

    app.run(debug=True)

