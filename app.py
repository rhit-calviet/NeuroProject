import os
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_file
from PIL import Image
from fpdf import FPDF
from datetime import datetime

# --- Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Define image dimensions for each model type
SEGMENTATION_IMG_SIZE = 128
CLASSIFICATION_IMG_SIZE = 224

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_new_even_more_secret_key'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Dictionaries ---
EXPECTED_MODELS = {
    'Tumor Identification': {'file': 'tumor.h5', 'type': 'segmentation'},
    'Alzheimer\'s Classification': {'file': 'alzhiemer.keras', 'type': 'classification'},
    'FLAIR Abnormality Segmentation': {'file': 'mri.h5', 'type': 'segmentation'}
}
MODELS = {}
ALZHEIMER_CLASSES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# --- Model Loading Function ---
def load_all_models():
    """Loads all available models from the models directory."""
    print("--- Loading all available models, this may take a moment... ---")
    for model_name, model_info in EXPECTED_MODELS.items():
        model_path = os.path.join(MODEL_FOLDER, model_info['file'])
        if os.path.exists(model_path):
            try:
                MODELS[model_name] = tf.keras.models.load_model(model_path, compile=False)
                print(f"✔️  Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"❌ Error loading model {model_name}: {e}")
        else:
            print(f"⚠️  Warning: Model file not found for '{model_name}'. It will be skipped.")
    print("--- Model loading complete. ---")

# --- Preprocessing Functions ---
def preprocess_for_segmentation(image_path):
    """Preprocesses an image for segmentation models (128x128)."""
    img = Image.open(image_path).convert("RGB")
    original_pil_img_for_display = img.copy() # Keep a copy for display
    img = img.resize((SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, original_pil_img_for_display

def preprocess_for_classification(image_path):
    """Preprocesses an image for classification models (224x224)."""
    img = Image.open(image_path).convert("RGB")
    original_pil_img_for_display = img.copy() # Keep a copy for display
    img = img.resize((CLASSIFICATION_IMG_SIZE, CLASSIFICATION_IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, original_pil_img_for_display

def encode_image_for_html(image):
    if isinstance(image, np.ndarray):
        if image.size == 0: image = Image.new('L', (128, 128), color=0)
        else: image = Image.fromarray((image * 255).astype(np.uint8).squeeze())
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Interpretation Functions ---
def get_segmentation_interpretation(model_name, prediction_array):
    mask_array = (prediction_array > 0.5)
    coverage = np.mean(mask_array) * 100
    positive_pixels = prediction_array[mask_array]
    confidence = np.mean(positive_pixels) * 100 if positive_pixels.size > 0 else "N/A"
    confidence_str = f"{confidence:.2f}%" if isinstance(confidence, (float, np.number)) else confidence
    report = {"status": "Analysis Complete", "summary": "", "details": "", "confidence": confidence_str, "type": "segmentation"}
    if model_name == 'FLAIR Abnormality Segmentation':
        report["details"] = "This model analyzes FLAIR MRI sequences to identify and segment hyperintense regions often indicative of lower-grade gliomas."
        if coverage > 0.1:
            report["status"] = "FLAIR Abnormality Detected"
            report["summary"] = f"The model identified a region of FLAIR hyperintensity covering approximately {coverage:.2f}% of the brain area."
        else:
            report["status"] = "No Significant FLAIR Abnormality Detected"
            report["summary"] = "The AI model did not detect significant areas of FLAIR hyperintensity."
    else:
        report["details"] = "This model is trained to segment specific regions of interest."
        if coverage > 0.1:
            report["status"] = "Region of Interest Detected"
            report["summary"] = f"The model has segmented a region covering {coverage:.2f}% of the scan."
        else: report["summary"] = "No significant regions were segmented."
    return report

def get_classification_interpretation(model_name, prediction_array):
    predicted_index = np.argmax(prediction_array)
    confidence = np.max(prediction_array) * 100
    predicted_class = ALZHEIMER_CLASSES[predicted_index]
    report = {"status": f"Classification: {predicted_class}", "summary": "", "details": "", "confidence": f"{confidence:.2f}%", "type": "classification", "predicted_class": predicted_class}
    report["details"] = "This model classifies the input image into one of four categories related to the progression of Alzheimer's disease."
    report["summary"] = f"The model has classified this scan as {predicted_class} with a confidence of {confidence:.2f}%."
    return report

# --- PDF Generation Class ---
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'AI-Powered Brain Scan Analysis Report', 0, 1, 'C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-20)
        self.set_font('Helvetica', 'I', 8)
        self.multi_cell(0, 5, 'Disclaimer: This report is generated by an automated AI system for informational purposes only. It is not a medical diagnosis and should not be used as a substitute for consultation with a qualified healthcare professional.', 0, 'C')
        self.set_y(-10)
        self.cell(0, 5, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
    def chapter_body(self, data, key_color=(0,0,0)):
        self.set_font('Helvetica', '', 11)
        key_width = 45 
        for key, value in data.items():
            start_y = self.get_y()
            self.set_font('Helvetica', 'B')
            self.set_text_color(*key_color)
            self.multi_cell(key_width, 7, f"{key}:")
            self.set_text_color(0,0,0)
            self.set_xy(self.get_x() + key_width, start_y)
            self.set_font('Helvetica', '')
            self.multi_cell(0, 7, str(value).replace('**', ''))
            self.ln(2) 

    def add_analysis_section(self, title, report, original_img_path, mask_img=None):
        self.add_page()
        self.chapter_title(title)
        if "Not Available" in report["status"]:
            self.chapter_body({"Status": report["status"]}, key_color=(108, 117, 125))
            self.chapter_body({"Summary": report["summary"]})
            return
        status_color = (200, 0, 0) if "Detected" in report["status"] or "Demented" in report.get("predicted_class", "") else (0, 100, 0)
        self.chapter_body({"Status": report["status"], "AI Confidence": report["confidence"]}, key_color=status_color)
        self.chapter_body({"Summary": report["summary"], "Model Details": report["details"]})
        
        if report["type"] == 'segmentation':
            original_pil = Image.open(original_img_path).convert("RGB")
            resized_original_pil = original_pil.resize((SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE))
            temp_original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_original.png')
            resized_original_pil.save(temp_original_path)

            mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8).squeeze())
            temp_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_mask.png')
            mask_pil.save(temp_mask_path)
            
            y_before_images = self.get_y()
            if y_before_images > 160: self.add_page(); y_before_images = self.get_y()
            
            img_width = 75; img_gap = 10
            page_content_width = self.w - self.l_margin - self.r_margin
            start_x = self.l_margin + (page_content_width - (img_width * 2 + img_gap)) / 2
            img1_x = start_x; img2_x = start_x + img_width + img_gap
            
            self.image(temp_original_path, x=img1_x, w=img_width, y=y_before_images)
            self.image(temp_mask_path, x=img2_x, w=img_width, y=y_before_images)
            
            self.set_y(y_before_images + img_width + 5)
            self.set_font('Helvetica', 'I', 9)
            self.cell(0, 5, 'Left: Original Scan  |  Right: AI Model Segmentation Mask', 0, 1, 'C')
            
            os.remove(temp_original_path)
            os.remove(temp_mask_path)
            
        elif report["type"] == 'classification':
            self.ln(10)
            self.image(original_img_path, x=self.w / 2 - 45, w=90)


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(request.url)
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    original_pil_img = Image.open(filepath).convert("RGB")
    display_img_resized = original_pil_img.resize((SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE))

    results_data = {}
    for name, model_info in EXPECTED_MODELS.items():
        if name in MODELS:
            try:
                model = MODELS[name]
                
                if model_info['type'] == 'segmentation':
                    processed_image, _ = preprocess_for_segmentation(filepath)
                    prediction = model.predict(processed_image)
                    report = get_segmentation_interpretation(name, prediction[0])
                    mask_for_display = (prediction[0] > 0.5).astype(np.uint8)
                    results_data[name] = {'report': report, 'mask_uri': encode_image_for_html(mask_for_display)}
                
                elif model_info['type'] == 'classification':
                    processed_image, _ = preprocess_for_classification(filepath)
                    prediction = model.predict(processed_image)
                    report = get_classification_interpretation(name, prediction[0])
                    results_data[name] = {'report': report, 'mask_uri': None}

            except Exception as e:
                print(f"[ERROR] Prediction for '{name}' failed: {e}")
                results_data[name] = {'report': {"status": "Analysis Failed", "summary": f"An error occurred: {e}", "type": "error"}, 'mask_uri': None}
        else:
            results_data[name] = {'report': {"status": "Model Not Available", "summary": "Model could not be loaded.", "type": "unavailable"}, 'mask_uri': None}
    
    final_context = {
        'original_img_uri': encode_image_for_html(display_img_resized),
        'analysis': results_data
    }
    
    return render_template('results.html', results=final_context, uploaded_filename=filename)

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return redirect(url_for('index'))

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=25)

    for name, model_info in EXPECTED_MODELS.items():
        if name in MODELS:
            try:
                model = MODELS[name]
                
                if model_info['type'] == 'segmentation':
                    processed_image, _ = preprocess_for_segmentation(filepath)
                    prediction = model.predict(processed_image)
                    report = get_segmentation_interpretation(name, prediction[0])
                    mask_array = (prediction[0] > 0.5).astype(np.uint8)
                    pdf.add_analysis_section(name, report, filepath, mask_array)

                elif model_info['type'] == 'classification':
                    processed_image, _ = preprocess_for_classification(filepath)
                    prediction = model.predict(processed_image)
                    report = get_classification_interpretation(name, prediction[0])
                    pdf.add_analysis_section(name, report, filepath)

            except Exception as e:
                pdf.add_analysis_section(name, {"status": "Analysis Failed", "summary": f"An error occurred: {e}", "type": "error"}, filepath)
        else:
             # --- FIX: Add the missing 'filepath' argument here ---
             report_data = {"status": "Model Not Available", "summary": "Model could not be loaded.", "type": "unavailable"}
             pdf.add_analysis_section(name, report_data, filepath)
    
    pdf_buffer = io.BytesIO(pdf.output())
    pdf_buffer.seek(0)
    
    return send_file(pdf_buffer, as_attachment=True, download_name='AI_Brain_Scan_Report.pdf', mimetype='application/pdf')

load_all_models()

if __name__ == '__main__':
    app.run(debug=True)
