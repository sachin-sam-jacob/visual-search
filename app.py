from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import json
import os
import sys
import logging
import traceback
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3008", "http://localhost:8000","https://smartgrocery.onrender.com","https://smart-grocery-h9jw.onrender.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.keras')

try:
    # Load the fruit/vegetable recognition model
    fruit_veg_model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Fruit/Vegetable recognition model loaded successfully")
except Exception as e:
    logger.error(f"Error loading fruit/vegetable model: {str(e)}")
    fruit_veg_model = None

# Class names for the model
FRUIT_VEG_CLASSES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Add to global variables
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'product_dataset')
product_features = None
product_index = None

def load_dataset():
    global product_features, product_index
    try:
        # Load features
        features_file = os.path.join(DATASET_PATH, 'features', 'product_features.npz')
        if os.path.exists(features_file):
            product_features = dict(np.load(features_file))
            logger.info(f"Loaded features for {len(product_features)} products")
        
        # Load index
        index_file = os.path.join(DATASET_PATH, 'product_index.json')
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                product_index = json.load(f)
            logger.info(f"Loaded index for {len(product_index)} products")
            
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")

# Call this at startup
load_dataset()

def find_similar_products(query_features, threshold=0.7):
    similar_products = []
    
    for product_name, features in product_features.items():
        similarity = cosine_similarity(
            query_features.reshape(1, -1),
            features.reshape(1, -1)
        )[0][0]
        
        if similarity > threshold:
            similar_products.append({
                'name': product_index[product_name]['name'],
                'similarity': float(similarity)
            })
    
    return sorted(similar_products, key=lambda x: x['similarity'], reverse=True)

def preprocess_image(image_data):
    """Image preprocessing for fruit/vegetable model"""
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Basic enhancements for better recognition
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)  # Slightly increase contrast
        
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)  # Slightly increase sharpness
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)  # Slightly increase brightness
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Preprocess for fruit/vegetable model (224x224 to match training)
        fruit_img = cv2.resize(img_array, (224, 224))  # Changed from 64x64 to 224x224
        fruit_img = fruit_img.astype(np.float32) / 255.0  # Normalize to [0,1]
        fruit_img = np.expand_dims(fruit_img, axis=0)
        
        return fruit_img
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def get_label_variations(label):
    """Get variations of labels for better matching"""
    label = label.lower().strip()
    
    # Common fruit and vegetable variations
    variations = {
        'apple': ['apple', 'apples', 'fresh apple', 'red apple', 'green apple'],
        'banana': ['banana', 'bananas', 'fresh banana', 'yellow banana'],
        'beetroot': ['beetroot', 'beet', 'beets', 'red beet'],
        'bell pepper': ['bell pepper', 'pepper', 'capsicum', 'sweet pepper'],
        'cabbage': ['cabbage', 'green cabbage', 'fresh cabbage'],
        'capsicum': ['capsicum', 'bell pepper', 'pepper', 'sweet pepper'],
        'carrot': ['carrot', 'carrots', 'fresh carrot', 'orange carrot'],
        'cauliflower': ['cauliflower', 'fresh cauliflower', 'white cauliflower'],
        'chilli pepper': ['chilli', 'chili', 'hot pepper', 'red chilli'],
        'corn': ['corn', 'maize', 'sweet corn', 'fresh corn'],
        'cucumber': ['cucumber', 'cucumbers', 'fresh cucumber', 'green cucumber'],
        'eggplant': ['eggplant', 'aubergine', 'brinjal', 'purple eggplant'],
        'garlic': ['garlic', 'garlic bulb', 'fresh garlic'],
        'ginger': ['ginger', 'ginger root', 'fresh ginger'],
        'grapes': ['grape', 'grapes', 'fresh grapes', 'green grapes', 'purple grapes'],
        'jalepeno': ['jalepeno', 'jalapeño', 'hot pepper', 'green chili'],
        'kiwi': ['kiwi', 'kiwifruit', 'fresh kiwi', 'green kiwi'],
        'lemon': ['lemon', 'lemons', 'fresh lemon', 'yellow lemon'],
        'lettuce': ['lettuce', 'green lettuce', 'fresh lettuce', 'leafy greens'],
        'mango': ['mango', 'mangoes', 'fresh mango', 'yellow mango'],
        'onion': ['onion', 'onions', 'fresh onion', 'red onion', 'white onion'],
        'orange': ['orange', 'oranges', 'fresh orange', 'citrus'],
        'paprika': ['paprika', 'red paprika', 'sweet paprika'],
        'pear': ['pear', 'pears', 'fresh pear', 'green pear'],
        'peas': ['peas', 'green peas', 'fresh peas', 'garden peas'],
        'pineapple': ['pineapple', 'fresh pineapple', 'tropical fruit'],
        'pomegranate': ['pomegranate', 'fresh pomegranate', 'red pomegranate'],
        'potato': ['potato', 'potatoes', 'fresh potato', 'white potato'],
        'raddish': ['radish', 'raddish', 'fresh radish', 'white radish'],
        'soy beans': ['soy', 'soya', 'soybeans', 'soy beans'],
        'spinach': ['spinach', 'fresh spinach', 'green spinach', 'leafy greens'],
        'sweetcorn': ['sweetcorn', 'corn', 'maize', 'fresh corn'],
        'sweetpotato': ['sweet potato', 'sweetpotato', 'yam'],
        'tomato': ['tomato', 'tomatoes', 'fresh tomato', 'red tomato'],
        'turnip': ['turnip', 'turnips', 'fresh turnip', 'white turnip'],
        'watermelon': ['watermelon', 'fresh watermelon', 'green watermelon']
    }
    
    # Return variations if found, otherwise return original label
    return variations.get(label, [label])

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        logger.info("Received analyze request")
        
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided'
            }), 400

        file = request.files['image']
        if not file:
            return jsonify({
                'success': False,
                'message': 'Empty file'
            }), 400

        # Process image
        image_data = file.read()
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to process image'
            }), 400

        # Get predictions from fruit/vegetable model
        predictions = []
        
        if fruit_veg_model is not None:
            # Get raw predictions with verbose=0 to reduce noise
            fruit_preds = fruit_veg_model.predict(processed_image, verbose=0)[0]
            
            # Get top 5 predictions with probability > 0.1
            top_indices = np.argsort(fruit_preds)[-5:][::-1]
            
            for idx in top_indices:
                probability = float(fruit_preds[idx])
                if probability > 0.1:  # Only include predictions with >10% confidence
                    label = FRUIT_VEG_CLASSES[idx]
                    variations = get_label_variations(label)
                    
                    predictions.append({
                        'label': label,
                        'variations': variations,
                        'probability': probability,
                        'source': 'fruit_veg',
                        'confidence_level': get_confidence_level(probability)
                    })
            
            # Log the predictions for debugging
            logger.info(f"Raw predictions: {fruit_preds}")
            logger.info(f"Top predictions: {predictions}")

        # Get unique search terms from all predictions
        search_terms = []
        for pred in predictions:
            if pred['probability'] > 0.2:  # Only use high confidence predictions for search
                search_terms.extend(pred['variations'])
        search_terms = list(set(search_terms))  # Remove duplicates

        return jsonify({
            'success': True,
            'predictions': predictions,
            'search_terms': search_terms,
            'debug_info': {
                'image_size': processed_image.shape,
                'prediction_count': len(predictions),
                'top_prediction': predictions[0] if predictions else None
            }
        })

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

def get_confidence_level(probability):
    """Convert probability to confidence level string"""
    if probability > 0.8:
        return "very high"
    elif probability > 0.6:
        return "high"
    elif probability > 0.4:
        return "medium"
    elif probability > 0.2:
        return "low"
    else:
        return "very low"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'python_version': sys.version,
        'tensorflow_version': tf.__version__,
        'fruit_veg_model_loaded': fruit_veg_model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)