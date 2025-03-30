# Cancer Fight: AI-Powered Breast Cancer Detection

[![Forks](https://img.shields.io/github/forks/Arison99/Cancer-Fight?style=for-the-badge)](https://github.com/Arison99/Cancer-Fight/network/members)
[![Stars](https://img.shields.io/github/stars/Arison99/Cancer-Fight?style=for-the-badge)](https://github.com/Arison99/Cancer-Fight/stargazers)
[![Issues](https://img.shields.io/github/issues/Arison99/Cancer-Fight?style=for-the-badge)](https://github.com/Arison99/Cancer-Fight/issues)
[![MIT License](https://img.shields.io/github/license/Arison99/Cancer-Fight?style=for-the-badge)](https://github.com/Arison99/Cancer-Fight/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors/Arison99/Cancer-Fight?style=for-the-badge)](https://github.com/Arison99/Cancer-Fight/graphs/contributors)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

An advanced AI tool for early detection of breast cancer through image analysis. This project utilizes computer vision and machine learning to extract diagnostic features from mammogram images and provide accurate classifications.

## 📋 Table of Contents

- [About](#-about)
- [Technologies Used](#-technologies-used)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Contributing](#-contributing)
- [Research Context](#-research-context)
- [License](#-license)

## 🔍 About

Cancer Fight is a web-based diagnostic tool that leverages artificial intelligence to analyze mammogram images for signs of breast cancer. The system extracts critical diagnostic features from X-ray images, processes them through a trained machine learning model, and provides a classification (benign or malignant) along with confidence metrics and visualizations.

This tool is designed as a research prototype to demonstrate how AI can assist medical professionals in early cancer detection.

## 💻 Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework for the application server
- **OpenCV & scikit-image**: Image processing and feature extraction
- **scikit-learn**: Machine learning model implementation
- **Pillow**: Additional image processing capabilities
- **Matplotlib & Seaborn**: Data visualization libraries
- **NumPy & Pandas**: Data manipulation and analysis
- **HTML/CSS/JavaScript**: Front-end interface

## ✨ Features

- **Image Upload & Analysis**: Process mammogram X-ray images for cancer detection
- **Feature Extraction**: Extract 30+ diagnostic features from images
- **AI Classification**: Predict benign or malignant diagnosis with confidence score
- **Visual Analytics**: Interactive visualizations of diagnostic features
- **Key Metrics**: Risk assessment, data quality evaluation, and confidence metrics
- **Detailed Reporting**: Comprehensive breakdown of all extracted features

## 🚀 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Arison99/Cancer-Fight.git
    cd Cancer-Fight
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Update model paths in `app.py`:
    Replace the paths with your local paths:
    ```python
    model = joblib.load(r'path\to\your\model.joblib')
    scaler = joblib.load(r'path\to\your\scaler.joblib')
    ```

5. Run the application:
    ```bash
    python app.py
    ```

6. Open your browser and navigate to:
    ```
    http://127.0.0.1:5000
    ```

## 🖥️ Usage

1. **Upload Image**: Select a mammogram X-ray image using the upload button
2. **Process Image**: Click "Analyze" to process the image
3. **View Results**: Examine the prediction, confidence score, and risk level
4. **Explore Visualizations**: Use the visualization tools to understand feature importance
5. **Review Details**: Check the detailed feature extraction data for in-depth analysis

## 🔬 How It Works

The system follows a multi-step process:

1. **Image Preprocessing**: Normalize, denoise, and enhance the mammogram image
2. **Segmentation**: Isolate breast tissue from background
3. **Feature Extraction**: Calculate diagnostic features including:
    - Radius, texture, perimeter, and area measurements
    - Smoothness, compactness, and concavity metrics
    - Symmetry and fractal dimension analysis
4. **Feature Scaling**: Normalize extracted features to match the training data scale
5. **Classification**: Process normalized features through the machine learning model
6. **Result Generation**: Produce probability scores, risk assessment, and visualizations

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Areas where contributions are particularly valuable:
- Improving feature extraction algorithms
- Enhancing UI/UX design
- Optimizing model performance
- Adding support for different types of medical images
- Creating comprehensive documentation

## 🎓 Research Context

This project is developed for research purposes only and is not intended for clinical use. The goal is to explore how AI and computer vision can assist in medical diagnostics, particularly in regions with limited access to specialist healthcare.

The feature extraction methods are based on established radiomics techniques used in medical image analysis, focusing on the quantitative features that have been shown to correlate with cancer diagnosis.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

⚠️ **Disclaimer**: This tool is for research and educational purposes only and should not replace professional medical diagnosis. Always consult healthcare professionals for medical concerns.

---

<p align="center">Made with ❤️ by Arison99 for advancing cancer research and detection</p>