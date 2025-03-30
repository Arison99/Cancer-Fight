import numpy as np
import cv2
from skimage import feature, measure
from scipy import stats
import skimage.feature as skf
from joblib import load

class MammographFeatureExtractor:
    """
    Extracts diagnostic features from mammogram images for breast cancer detection.
    Features include radius, texture, perimeter, area, smoothness, etc.
    """
    
    def __init__(self):
        # Load the expected feature names from the saved model
        try:
            self.feature_names = load('feature_names.joblib')
        except:
            # Fallback feature names if file not found
            self.feature_names = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                'smoothness_se', 'compactness_se', 'concavity_se',
                'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                'smoothness_worst', 'compactness_worst', 'concavity_worst',
                'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst',
                'diagnosis', 'id'
            ]
    
    def extract_features(self, img):
        """
        Extract all required features from the image
        """
        # Convert to grayscale if it's not already
        if len(img.shape) > 2:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
            
        # Initialize features dictionary with id
        features = {'id': 0}  # Add default id value
            
        # Basic preprocessing
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Threshold to separate breast tissue from background
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Return empty feature dict with id if no contours found
        if not contours:
            return {name: 0.0 for name in self.feature_names}
            
        # Get largest contour (breast region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask for the breast region
        mask = np.zeros_like(img_gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Apply mask to the image
        masked_img = cv2.bitwise_and(img_gray, img_gray, mask=mask)
        
        # Calculate region properties
        props = self._calculate_region_props(masked_img, largest_contour)
        
        # Calculate texture features
        texture_features = self._calculate_texture_features(masked_img, mask)
        
        # Combine all features
        features.update(props)
        features.update(texture_features)
        
        # Ensure all required features are present with default value 0.0
        ordered_features = {name: features.get(name, 0.0) for name in self.feature_names}
                
        return ordered_features
    
    def _calculate_region_props(self, img, contour):
        """Calculate region properties like radius, area, perimeter, etc."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate radius (mean distance from center to points on contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            
        # Calculate distances from center to each point on contour
        distances = []
        for point in contour:
            x, y = point[0]
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            distances.append(distance)
            
        radius_mean = np.mean(distances) if distances else 0
        radius_se = np.std(distances) if distances else 0
        radius_worst = np.max(distances) if distances else 0
        
        # Compactness and smoothness
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        smoothness = 1 - (1 / compactness) if compactness > 0 else 0
        
        # Concavity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        concavity = (hull_area - area) / hull_area if hull_area > 0 else 0
        
        # Symmetry calculation based on moments
        hu_moments = cv2.HuMoments(M).flatten()
        symmetry = 1 - hu_moments[0]  # First Hu moment is invariant to rotation
        
        # Fractal dimension approximation using box counting
        fractal_dim = self._calculate_fractal_dimension(contour)
        
        # Concave points
        concave_points = self._count_concave_points(contour) / len(contour) if len(contour) > 0 else 0
        
        # Calculate standard error and worst values
        # (For a real implementation, these would be based on multiple measurements)
        # Here we're simulating by adding random variation
        
        props = {
            'radius_mean': radius_mean,
            'perimeter_mean': perimeter,
            'area_mean': area,
            'smoothness_mean': smoothness,
            'compactness_mean': compactness,
            'concavity_mean': concavity,
            'concave_points_mean': concave_points,
            'symmetry_mean': symmetry,
            'fractal_dimension_mean': fractal_dim,
            
            'radius_se': radius_se,
            'perimeter_se': perimeter * 0.1,
            'area_se': area * 0.1,
            'smoothness_se': smoothness * 0.1,
            'compactness_se': compactness * 0.1,
            'concavity_se': concavity * 0.1,
            'concave_points_se': concave_points * 0.1,
            'symmetry_se': symmetry * 0.1,
            'fractal_dimension_se': fractal_dim * 0.1,
            
            'radius_worst': radius_worst,
            'perimeter_worst': perimeter * 1.2,
            'area_worst': area * 1.2,
            'smoothness_worst': min(1.0, smoothness * 1.2),
            'compactness_worst': compactness * 1.2,
            'concavity_worst': min(1.0, concavity * 1.2),
            'concave_points_worst': min(1.0, concave_points * 1.2),
            'symmetry_worst': min(1.0, symmetry * 1.2),
            'fractal_dimension_worst': min(3.0, fractal_dim * 1.2),
        }
        
        return props
    
    def _calculate_texture_features(self, img, mask):
        """Calculate texture features using GLCM and Haralick features"""
        # Create GLCM matrix
        glcm = feature.graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                   256, symmetric=True, normed=True)
        
    
        # Extract properties from GLCM
        contrast = feature.graycoprops(glcm, 'contrast')
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
        homogeneity = feature.graycoprops(glcm, 'homogeneity')
        energy = feature.graycoprops(glcm, 'energy')
        correlation = feature.graycoprops(glcm, 'correlation')
        
        # Combine the features from different angles
        texture_features = np.hstack([
            contrast.flatten(), 
            dissimilarity.flatten(), 
            homogeneity.flatten(), 
            energy.flatten(), 
            correlation.flatten()
        ])
        
        # Use energy as the texture measurement (equivalent to Angular Second Moment)
        texture_mean = np.mean(energy)
        texture_se = np.std(energy)
        texture_worst = np.max(energy)
        
        return {
            'texture_mean': texture_mean,
            'texture_se': texture_se,
            'texture_worst': texture_worst
        }
    
    def _calculate_fractal_dimension(self, contour):
        """Approximate fractal dimension using box counting method"""
        # Convert contour to binary image
        x, y, w, h = cv2.boundingRect(contour)
        binary = np.zeros((h, w), dtype=np.uint8)
        
        # Draw contour
        shifted_contour = contour - np.array([x, y])
        cv2.drawContours(binary, [shifted_contour], -1, 255, 1)
        
        # Simple box counting implementation
        sizes = np.array([2, 4, 8, 16, 32, 64])
        counts = []
        
        for size in sizes:
            # How many boxes of 'size' are needed to cover the contour
            resized = cv2.resize(binary, dsize=(binary.shape[1] // size, binary.shape[0] // size),
                                interpolation=cv2.INTER_AREA)
            count = np.sum(resized > 0)
            counts.append(count)
        
        if all(c > 0 for c in counts):
            # Linear fit of log(count) vs log(1/size)
            coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
            return coeffs[0]  # Slope is the fractal dimension
        else:
            return 1.0  # Default value if calculation fails
    
    def _count_concave_points(self, contour):
        """Count concave points in the contour"""
        hull = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull) > 0:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                return len(defects)
        
        return 0

def extract_features_from_image(img_array):
    """
    Helper function to extract features from an image array
    Returns a dictionary of features ensuring all required features are present
    """
    extractor = MammographFeatureExtractor()
    features = extractor.extract_features(img_array)
    
    # Ensure all features are present with default value 0.0
    required_features = extractor.feature_names
    
    # Create ordered dictionary with all required features
    ordered_features = {feature: features.get(feature, 0.0) for feature in required_features}
    return ordered_features
