from flask import Flask, request, render_template, jsonify, redirect, url_for, send_file
import numpy as np
from PIL import Image
import cv2
import io
import json
import requests
from models.img_scanner import extract_features_from_image
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import base64
import seaborn as sns

app = Flask(__name__)

# Load the trained model and scaler || Replace with path on your computer
model = joblib.load(r'C:\\Users\\USER\\OneDrive\\Desktop\\Cancer Fight\\models\\model.joblib')
scaler = joblib.load(r'C:\\Users\\USER\\OneDrive\\Desktop\\Cancer Fight\\models\\scaler.joblib')

# Load the feature names (if available) || replace with path on your computer
try:
    feature_names = joblib.load(r'C:\\Users\\USER\\OneDrive\\Desktop\\Cancer Fight\\models\\feature_names.joblib')
    print(f"Loaded {len(feature_names)} feature names")
except:
    # Fallback to feature names from scaler if available
    feature_names = scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else None
    print(f"Using feature names from scaler: {feature_names is not None}")

# Define typical feature ranges for normalization based on common cancer datasets to account for errors in image_feature_data
FEATURE_REFERENCE = {
    'radius': {'mean': 14.1, 'std': 3.5},
    'texture': {'mean': 19.3, 'std': 4.3},
    'perimeter': {'mean': 92.0, 'std': 24.3},
    'area': {'mean': 654.9, 'std': 351.9},
    'smoothness': {'mean': 0.096, 'std': 0.014},
    'compactness': {'mean': 0.10, 'std': 0.05},
    'concavity': {'mean': 0.09, 'std': 0.08},
    'concave_points': {'mean': 0.05, 'std': 0.04},
    'symmetry': {'mean': 0.18, 'std': 0.03},
    'fractal_dimension': {'mean': 0.06, 'std': 0.01}
}


# route to index/homepage template of the web app.
@app.route('/')
def index():
    return render_template('index.html')

# Route to for image processing, it sends the image to the image scanner for where features will be extracted from the image. 
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'xray' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['xray']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        original_size = img.size
        
        # Convert PIL Image to numpy array for OpenCV processing
        img_array = np.array(img)
        
        # Extract features using our custom feature extractor
        features_dict = extract_features_from_image(img_array)
        
        # Add image metadata
        features_dict['image_name'] = file.filename
        features_dict['image_size'] = [original_size[0], original_size[1]]
        
        # Normalize features
        normalized_features = normalize_features(features_dict)
        
        # Send the features to the predict endpoint internally
        prediction_result = predict_internal(normalized_features)
        
        # Combine features and prediction into a single result
        result = {
            'features': normalized_features,
            'prediction': prediction_result,
            'original_features': features_dict  # Include original features for reference
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Route for model predictions and helper functions that help the model to make predictions from the extracted image features.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.json
        
        # Normalize features before prediction
        normalized_features = normalize_features(input_data)
        
        # Process the prediction
        prediction_result = predict_internal(normalized_features)
        
        # Return the prediction as JSON
        return jsonify(prediction_result)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

def normalize_features(features_dict):
    """
    Normalize feature values to ensure they're on the correct scale for the model.
    """
    normalized = features_dict.copy()
    
    # Skip non-numeric and metadata fields
    skip_fields = ['id', 'image_name', 'image_size', 'diagnosis']
    
    for key, value in features_dict.items():
        # Skip non-numeric and metadata fields
        if key in skip_fields or not isinstance(value, (int, float)):
            continue
        
        # Extract the base feature type (e.g., 'radius' from 'radius_mean')
        feature_parts = key.split('_')
        if len(feature_parts) < 2:
            continue
            
        base_feature = feature_parts[0]
        category = feature_parts[1] if len(feature_parts) > 1 else 'mean'
        
        # Check if we have reference values for this feature type
        if base_feature in FEATURE_REFERENCE:
            ref_value = FEATURE_REFERENCE[base_feature]['mean']
            
            # If value is significantly larger than reference, scale it down
            if value > ref_value * 10:
                # Calculate appropriate scaling factor (10, 100, 1000, etc.)
                scale_factor = 10 ** int(np.log10(value / ref_value))
                print(f"Scaling {key}: {value} is too large (ref ~{ref_value}). Scaling by {scale_factor}")
                normalized[key] = value / scale_factor
    
    return normalized

def predict_internal(features_dict):
    """Helper function to make predictions from features"""
    try:
        # If we have feature names from training, use those to create the input vector
        if feature_names:
            # Print missing features for debugging
            missing_features = [f for f in feature_names if f not in features_dict]
            if missing_features:
                print(f"Warning: Missing {len(missing_features)} features: {missing_features[:5]}...")
            
            # Create feature vector with exact matching of feature names
            feature_vector = []
            for feature in feature_names:
                if feature in features_dict:
                    feature_vector.append(features_dict[feature])
                else:
                    # Use 0 for missing features
                    feature_vector.append(0.0)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
        else:
            # Fallback to using scaler's feature names if available
            if hasattr(scaler, 'feature_names_in_'):
                feature_values = [features_dict.get(feature, 0.0) for feature in scaler.feature_names_in_]
                feature_vector = np.array(feature_values).reshape(1, -1)
            else:
                # Last resort: try to extract all numeric features, but this is risky
                print("Warning: No feature names available, using all numeric features")
                numeric_features = {k: v for k, v in features_dict.items() 
                                  if isinstance(v, (int, float)) and k not in ['id', 'diagnosis']}
                feature_vector = np.array(list(numeric_features.values())).reshape(1, -1)
        
        # Print feature vector shape for debugging
        print(f"Feature vector shape: {feature_vector.shape}")
        
        # Scale the features
        scaled_features = scaler.transform(feature_vector)
        
        # Replace NaN values with zeros to avoid errors in KNN
        scaled_features = np.nan_to_num(scaled_features)
        
        # Make prediction
        probability = model.predict_proba(scaled_features)[0][1]  # Probability of being malignant
        classification = 'Malignant' if probability > 0.5 else 'Benign'
        confidence = max(probability, 1-probability)
        
        return {
            'probability': float(probability),
            'probability_benign': float(1-probability),
            'probability_malignant': float(probability),
            'classification': classification,
            'confidence': float(confidence)
        }
    except Exception as e:
        import traceback
        print(f"Error in predict_internal: {str(e)}")
        print(traceback.format_exc())
        raise e

# Test endpoint for carrying targeted tests. Use it if you want to do further analysis on the model's behaviour.
@app.route('/analyze', methods=['POST'])
def analyze_json():
    """
    Endpoint to directly analyze JSON features without image upload
    Useful for testing or batch processing
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    try:
        features_dict = request.get_json()
        
        # Normalize features
        normalized_features = normalize_features(features_dict)
        
        # Make prediction using internal function
        prediction_result = predict_internal(normalized_features)
        
        return jsonify({
            'features': normalized_features,
            'prediction': prediction_result,
            'original_features': features_dict
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Visualization of feautures for analysis.
@app.route('/feature_visualization', methods=['POST'])
def feature_visualization():
    try:
        # Get features from request
        data = request.json
        features = data.get('features', {})
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Generate visualization
        viz_type = data.get('type', 'importance')
        img_data = generate_visualization(features, viz_type)
        
        # Return the image as base64 encoded string
        return jsonify({
            'visualization': img_data,
            'type': viz_type
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def generate_visualization(features, viz_type='importance'):
    """Generate feature visualizations based on the provided features"""
    plt.figure(figsize=(10, 6))
    
    if viz_type == 'importance':
        # Extract mean features for importance visualization
        mean_features = {k: v for k, v in features.items() if k.endswith('_mean') and isinstance(v, (int, float))}
        
        # Sort by value ,Handle cases with fewer than 10 features
        sorted_features = dict(sorted(mean_features.items(), key=lambda item: item[1], reverse=True)[:min(10, len(mean_features))])
        
        # Create bar chart
        plt.barh(
            [k.replace('_mean', '').replace('_', ' ').title() for k in sorted_features.keys()],
            list(sorted_features.values()),
            color='skyblue'
        )
        plt.xlabel('Value')
        plt.ylabel('Feature')
        plt.title('Top 10 Important Mean Features')
        plt.tight_layout()
        
    elif viz_type == 'comparison':
        # Create feature comparison between categories (mean, SE, worst)
        selected_features = ['radius', 'texture', 'perimeter', 'area', 'smoothness']
        data = []
        
        for feature in selected_features:
            mean_val = features.get(f'{feature}_mean', 0)
            se_val = features.get(f'{feature}_se', 0)
            worst_val = features.get(f'{feature}_worst', 0)
            data.append([mean_val, se_val, worst_val])
        
        # Create heatmap
        sns.heatmap(
            data, 
            annot=True, 
            fmt='.2f', 
            xticklabels=['Mean', 'SE', 'Worst'],
            yticklabels=selected_features,
            cmap='viridis_r'
        )
        plt.title('Feature Comparison')
        plt.tight_layout()
        
    elif viz_type == 'distribution':
        # Plot distribution of values for specific feature types
        mean_vals = [v for k, v in features.items() if k.endswith('_mean') and isinstance(v, (int, float))]
        se_vals = [v for k, v in features.items() if k.endswith('_se') and isinstance(v, (int, float))]
        worst_vals = [v for k, v in features.items() if k.endswith('_worst') and isinstance(v, (int, float))]
        
        plt.boxplot([mean_vals, se_vals, worst_vals], labels=['Mean', 'SE', 'Worst'])
        plt.ylabel('Value Distribution')
        plt.title('Feature Value Distribution by Category')
        plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_data

# Route for Key Metrics 
@app.route('/key-metrics', methods=['POST'])
def key_metrics():
    try:
        # Get features from request
        data = request.json
        features = data.get('features', {})
        prediction = data.get('prediction', {})
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Calculate key metrics
        metrics = calculate_key_metrics(features, prediction)
        
        return jsonify(metrics)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def calculate_key_metrics(features, prediction):
    """Calculate important metrics based on the features and prediction"""
    metrics = {}
    
    # Confidence level (already available in prediction)
    metrics['confidence'] = prediction.get('confidence', 0) * 100
    
    # Top risk factors (highest value features that correlate with malignancy)
    mean_features = {k: v for k, v in features.items() if k.endswith('_mean') and isinstance(v, (int, float))}
    
    # For each feature, calculate normalized risk score based on typical ranges
    risk_scores = {}
    for k, v in mean_features.items():
        base_feature = k.split('_')[0]
        if base_feature in FEATURE_REFERENCE:
            ref_mean = FEATURE_REFERENCE[base_feature]['mean']
            ref_std = FEATURE_REFERENCE[base_feature]['std']
            # Calculate z-score to represent deviation from typical values
            z_score = (v - ref_mean) / ref_std if ref_std > 0 else 0
            risk_scores[k] = abs(z_score)  # Use absolute deviation for risk score
    
    # Get top 5 risk factors by deviation from normal range
    sorted_risk_features = dict(sorted(risk_scores.items(), key=lambda item: item[1], reverse=True)[:5])
    metrics['top_factors'] = [
        {
            'name': k.replace('_mean', '').replace('_', ' ').title(),
            'value': round(mean_features[k], 4),
            'risk_score': round(v, 2)
        } 
        for k, v in sorted_risk_features.items()
    ]
    
    # Data quality (count of non-zero features)
    feature_count = sum(1 for k, v in features.items() 
                        if isinstance(v, (int, float)) and v != 0 
                        and k not in ['image_name', 'image_size'])
    metrics['data_quality'] = min(100, round((feature_count / 30) * 100))  # Assuming ~30 features is complete
    
    # Risk level classification
    if prediction.get('classification') == 'Malignant':
        if prediction.get('confidence', 0) > 0.9:
            risk_level = "High Risk"
        elif prediction.get('confidence', 0) > 0.7:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low-Moderate Risk"
    else:
        if prediction.get('confidence', 0) > 0.9:
            risk_level = "Very Low Risk"
        elif prediction.get('confidence', 0) > 0.7:
            risk_level = "Low Risk"
        else:
            risk_level = "Low-Moderate Risk"
    
    metrics['risk_level'] = risk_level
    
    return metrics

if __name__ == '__main__':
    app.run(debug=True)