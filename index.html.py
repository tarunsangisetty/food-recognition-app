import os
import uuid
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained MobileNetV2 model (free, no API key needed)
print("Loading AI model... This may take a moment on first run.")
model = MobileNetV2(weights='imagenet')

# Load nutrition database (local JSON file)
def load_nutrition_db():
    """Load nutrition information for common foods"""
    return {
        'apple': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2, 'portion': '100g'},
        'banana': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3, 'portion': '100g'},
        'pizza': {'calories': 285, 'protein': 12, 'carbs': 36, 'fat': 10, 'portion': '1 slice'},
        'hamburger': {'calories': 354, 'protein': 20, 'carbs': 30, 'fat': 17, 'portion': '1 sandwich'},
        'salad': {'calories': 120, 'protein': 5, 'carbs': 10, 'fat': 7, 'portion': '1 bowl'},
        'sandwich': {'calories': 250, 'protein': 12, 'carbs': 30, 'fat': 9, 'portion': '1 sandwich'},
        'soup': {'calories': 150, 'protein': 6, 'carbs': 15, 'fat': 7, 'portion': '1 cup'},
        'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'portion': '100g'},
        'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'portion': '100g'},
        'fish': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 12, 'portion': '100g'},
        'eggs': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11, 'portion': '2 eggs'},
        'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1.1, 'portion': '100g'},
        'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2, 'portion': '100g'},
        'coffee': {'calories': 1, 'protein': 0.1, 'carbs': 0, 'fat': 0, 'portion': '1 cup'},
        'tea': {'calories': 1, 'protein': 0, 'carbs': 0.3, 'fat': 0, 'portion': '1 cup'},
        'orange': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1, 'portion': '100g'},
        'strawberry': {'calories': 32, 'protein': 0.7, 'carbs': 7.7, 'fat': 0.3, 'portion': '100g'},
        'ice cream': {'calories': 207, 'protein': 3.5, 'carbs': 24, 'fat': 11, 'portion': '100g'},
        'chocolate': {'calories': 546, 'protein': 4.9, 'carbs': 61, 'fat': 31, 'portion': '100g'},
        'cake': {'calories': 257, 'protein': 3, 'carbs': 45, 'fat': 8, 'portion': '1 slice'},
    }

NUTRITION_DB = load_nutrition_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_food(image_path):
    """Use MobileNet to predict food in image"""
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=3)[0]
        
        # Extract food-related predictions
        foods = []
        for _, label, confidence in decoded:
            # Clean up label (remove prefixes like 'n07753592_')
            food_name = label.split('_')[-1].lower()
            foods.append({
                'name': food_name,
                'confidence': float(confidence)
            })
        
        return foods
    except Exception as e:
        print(f"Prediction error: {e}")
        return []

def get_nutrition_info(food_name):
    """Get nutrition info from database or return default"""
    # Try exact match
    if food_name in NUTRITION_DB:
        return NUTRITION_DB[food_name]
    
    # Try partial match
    for key in NUTRITION_DB:
        if key in food_name or food_name in key:
            return NUTRITION_DB[key]
    
    # Return default if no match
    return {
        'calories': 100,
        'protein': 5,
        'carbs': 15,
        'fat': 4,
        'portion': 'standard serving'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        # Save file
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict food
        predictions = predict_food(filepath)
        
        if not predictions:
            return jsonify({"error": "Could not identify food in image"}), 400
        
        # Get nutrition for top prediction
        top_food = predictions[0]['name']
        nutrition = get_nutrition_info(top_food)
        
        # Build response
        foods = []
        total_calories = 0
        
        for pred in predictions[:3]:  # Show top 3 predictions
            food_nutrition = get_nutrition_info(pred['name'])
            foods.append({
                'name': pred['name'].title(),
                'confidence': f"{pred['confidence']*100:.1f}%",
                'portion': food_nutrition['portion'],
                'calories': food_nutrition['calories'],
                'protein': food_nutrition['protein'],
                'carbs': food_nutrition['carbs'],
                'fat': food_nutrition['fat']
            })
            if pred == predictions[0]:  # Only add top prediction to total
                total_calories = food_nutrition['calories']
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'foods': foods,
            'total_calories': total_calories
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)