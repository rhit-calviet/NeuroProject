# NeuroScan AI: Advanced Neurological Analysis

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange.svg)

**NeuroScan AI** is a sophisticated web-based platform designed for the preliminary analysis of brain MRI scans. It leverages a suite of deep learning models to detect indicators of various neurological conditions and integrates a Large Language Model (Google's Gemini) to synthesize the findings into a formal, human-readable report.

---

### Key Features

* **Multi-Model Analysis**: Simultaneously runs three distinct classification models to check for:
    * **Brain Tumor Types**: Identifies one of 15 classes of brain tumors.
    * **Alzheimer's Disease**: Classifies scans into four stages of Alzheimer's progression.
    * **Parkinson's Disease Dementia (PDD)**: Differentiates between healthy scans and those showing PDD indicators.
* **Versatile File Support**: Accepts standard image formats (`.jpg`, `.png`) and medical imaging files (`.dcm`), with automatic DICOM-to-JPG conversion.
* **Generative AI Integration**: Uses Google's Gemini API to generate a professional, integrated summary and list of recommendations based on the collective findings of the models.
* **Comprehensive Reporting**:
    * An immediate, user-friendly results page on the web.
    * A detailed, multi-page PDF report for download, featuring the AI-generated summary, patient information, and a model-by-model breakdown of the analysis.
* **Interactive UI**: A clean, responsive user interface that allows users to learn more about the underlying AI models and their architectures.

---

### Technology Stack

* **Backend**: Python, Flask
* **Deep Learning**: TensorFlow, Keras
* **Generative AI**: Google Gemini API (`google-generativeai`)
* **Frontend**: HTML5, CSS3, JavaScript
* **Core Libraries**:
    * `pydicom`: For parsing DICOM files.
    * `Pillow`: For image processing and manipulation.
    * `fpdf2`: For dynamic PDF report generation.
    * `numpy`: For numerical operations.

---

### Models & Datasets

This project would not be possible without open-source medical imaging datasets. The three models used in this application were trained on data from the following sources:

1.  **Brain Tumor MRI Classification**
    * **Description**: A multi-class classification model trained to identify 15 different types of brain tumors, including various grades of astrocytoma, carcinoma, glioblastoma, and normal scans.
    * **Dataset Source**: [Brain Tumor MRI Images 44 Classes on Kaggle](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)

2.  **Alzheimer's Stage Classification**
    * **Description**: A model that classifies brain scans into four stages of Alzheimer's disease: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented.
    * **Dataset Source**: [Augmented Alzheimer MRI Dataset on Kaggle](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset?resource=download)

3.  **Parkinson's Disease Dementia (PDD) Classification**
    * **Description**: A binary classification model trained to distinguish between scans of healthy individuals and those with Parkinson's Disease Dementia.
    * **Dataset Source**: [Parkinson's Disease Dementia (PDD) MRI Dataset on Kaggle](https://www.kaggle.com/datasets/ajithdari/parkinsons-disease-dementia-pdd)

---

### Setup and Installation

To run this project locally, follow these steps:

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/neuroscan-ai.git](https://github.com/your-username/neuroscan-ai.git)
cd neuroscan-ai
```

**2. Create and Activate a Virtual Environment**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
Create a file named `requirements.txt` in the root of your project directory and add the following content:
```
Flask
tensorflow
numpy
Pillow
fpdf2
pydicom
opencv-python-headless
google.generativeai
pandas
scikit-learn
seaborn
```
Then, run the following command to install the packages:
```bash
pip install -r requirements.txt
```

**4. Download AI Models**
Download the three pre-trained model files (`tumor.keras`, `alzhiemer.keras`, `dementia.h5`) and place them inside the `models/` directory.

**5. Configure API Key**
Open the `chat.py` file and replace `"your_api"` with your actual Google AI Studio API key.
```python
# in chat.py
API_KEY = "PASTE_YOUR_GOOGLE_API_KEY_HERE"
```

**6. Run the Application**
```bash
python app.py
```
Open your web browser and navigate to `http://127.0.0.1:5000`.

---

### Usage

1.  **Fill in Patient Information**: Enter the patient's name, age, and sex in the form.
2.  **Upload a Scan**: Click "Select Brain Scan Image" to upload a `.jpg`, `.png`, or `.dcm` file.
3.  **Begin Analysis**: Click the "Begin Analysis" button to submit the scan.
4.  **Review Web Results**: The application will display a summary of the findings from each model on the results page.
5.  **Download PDF Report**: Click the "Download Full PDF Report" button to get a detailed, multi-page report containing the AI-generated summary and a breakdown of each model's analysis.

---

### Project Structure

```
.
├── app.py              # Main Flask application, routes, and logic
├── chat.py             # Handles the call to the Google Gemini API
├── dicom_utils.py      # Utility for converting DICOM files to JPG
├── models/             # Directory to store the .h5 and .keras model files
│   ├── alzhiemer.keras
│   ├── dementia.h5
│   └── tumor.keras
├── static/
│   ├── css/
│   │   └── style.css   # Main stylesheet
│   └── js/
│       └── main.js     # Frontend JavaScript for interactivity
└── templates/
    ├── index.html      # Main landing and upload page
    └── results.html    # Page to display analysis results
```

---

### Disclaimer

**Important**: NeuroScan AI is an experimental tool developed for informational and academic purposes only. It is **not a medical device**, and its output **does not constitute a medical diagnosis**. The analysis provided is preliminary and should not be used as a substitute for consultation with a qualified healthcare professional. Always consult a doctor for any health concerns.

---

### Contact

This project was developed by:
* **Matteo Calviello** - Lead Data Scientist
* **Tommaso Calviello** - Project Manager
