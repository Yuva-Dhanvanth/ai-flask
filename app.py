from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from flask_cors import CORS
import cv2
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def load_model():
    """Load the actual trained model"""
    model_files = [
        "covid_model.keras",  # Primary - new Keras format
        "covid_model.h5",     # Secondary - legacy format
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                logger.info(f"🔍 Loading model from: {model_file}")
                model = tf.keras.models.load_model(model_file)
                logger.info(f"✅ SUCCESS: Model loaded from {model_file}")
                logger.info(f"📊 Model input shape: {model.input_shape}")
                logger.info(f"📊 Model output shape: {model.output_shape}")
                return model, model_file
            except Exception as e:
                logger.error(f"❌ Failed to load {model_file}: {str(e)}")
                continue
    
    # If no model could be loaded
    available_files = [f for f in os.listdir() if f.endswith(('.h5', '.keras'))]
    logger.error(f"Available model files: {available_files}")
    raise Exception("No working model found. Please check if model files are corrupted.")

# Load the actual trained model
try:
    model, model_path = load_model()
    logger.info(f"🚀 AI Model ready! Loaded from: {model_path}")
except Exception as e:
    logger.error(f"💥 CRITICAL: {e}")
    exit(1)

def prepare_image(file_path):
    """Prepare image for model prediction - matches your training preprocessing"""
    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Could not read the image file")
        
        logger.info(f"✅ Image loaded successfully: {img.shape}")
        
        # Resize to match your model's training input (100, 100, 3)
        img = cv2.resize(img, (100, 100))
        
        # Normalize pixel values (0-255 -> 0-1) - same as training
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        logger.info(f"✅ Image processed: {img.shape}")
        return img
        
    except Exception as e:
        logger.error(f"❌ Error preparing image: {str(e)}")
        raise

@app.route('/')
def home():
    return jsonify({
        "message": "COVID-19 X-Ray Detection API",
        "status": "active",
        "model_loaded": True,
        "model_type": "AI Neural Network",
        "model_source": model_path,
        "input_shape": str(model.input_shape),
        "endpoints": [
            "POST /predict - Upload X-ray image for COVID detection",
            "GET /health - Check API status",
            "GET /uploaded_image - View uploaded X-ray"
        ]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_source": model_path,
        "ai_model": "Working - Real AI Model"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Analyze X-ray image for COVID-19 using trained AI model"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Use JPG, JPEG, or PNG"}), 400
        
        # Save uploaded file
        upload_path = "uploaded_image.jpg"
        file.save(upload_path)
        logger.info(f"✅ X-ray image saved: {upload_path}")
        
        # Prepare image for AI model
        processed_img = prepare_image(upload_path)
        
        # Get AI prediction from trained model
        logger.info("🧠 Running AI model prediction...")
        prediction_result = model.predict(processed_img, verbose=0)
        prediction_value = prediction_result[0][0]
        
        logger.info(f"🔮 AI Model raw output: {prediction_value}")
        
        # Interpret AI model results
        if prediction_value > 0.5:
            label = "COVID-19 Detected"
            confidence = float(prediction_value)
            status = "high_risk"
        else:
            label = "Normal (No COVID-19)"
            confidence = float(1 - prediction_value)
            status = "low_risk"
        
        # Prepare response
        response_data = {
            "prediction": label,
            "confidence": round(confidence, 4),
            "probability": float(prediction_value),
            "status": status,
            "model_used": "Trained CNN Model",
            "model_source": model_path,
            "message": "AI analysis completed successfully",
            "medical_note": "Consult healthcare professional for diagnosis"
        }
        
        logger.info(f"📊 AI Result: {label} | Confidence: {confidence:.2%}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"💥 Prediction error: {str(e)}")
        return jsonify({
            "error": f"AI analysis failed: {str(e)}",
            "message": "Please try with a clear X-ray image"
        }), 500

@app.route('/uploaded_image')
def get_uploaded_image():
    """Return the uploaded X-ray image"""
    try:
        return send_file("uploaded_image.jpg", mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": "X-ray image not found"}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting COVID-19 AI Detection API on port {port}")
    logger.info(f"🧠 Loaded trained model from: {model_path}")
    logger.info(f"📐 Model expects input shape: {model.input_shape}")
    app.run(host='0.0.0.0', port=port, debug=False)