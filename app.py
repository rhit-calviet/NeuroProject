import os
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from PIL import Image
from fpdf import FPDF
from datetime import datetime

# --- Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_new_even_more_secret_key'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Dictionaries ---
EXPECTED_MODELS = {
    'Tumor Identification': 'tumor.h5',
    'Alzheimer\'s Markers': 'alzheimer.h5',
    'FLAIR Abnormality Segmentation': 'mri.h5'
}
MODELS = {}

# --- Model Loading Function ---
def load_all_models():
    """Loads all available .h5 models from the models directory."""
    print("--- Loading all available models, this may take a moment... ---")
    for model_name, file_name in EXPECTED_MODELS.items():
        model_path = os.path.join(MODEL_FOLDER, file_name)
        if os.path.exists(model_path):
            try:
                MODELS[model_name] = tf.keras.models.load_model(model_path, compile=False)
                print(f"✔️  Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"❌ Error loading model {model_name}: {e}")
        else:
            print(f"⚠️  Warning: Model file not found for '{model_name}'. It will be skipped during analysis.")
    print("--- Model loading complete. ---")

# --- Helper Functions ---
def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array /= 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img

def encode_image_for_html(image):
    if isinstance(image, np.ndarray):
        if image.size == 0:
            image = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), color=0)
        else:
            image = Image.fromarray((image * 255).astype(np.uint8).squeeze())
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_text_interpretation(model_name, prediction_array):
    """
    Generates interpretation from the raw prediction array to calculate confidence.
    """
    # Binarize the mask for coverage calculation
    mask_array = (prediction_array > 0.5)
    coverage = np.mean(mask_array) * 100
    
    # Calculate confidence score
    positive_pixels = prediction_array[mask_array]
    confidence = np.mean(positive_pixels) * 100 if positive_pixels.size > 0 else "N/A"
    
    # Use a more robust check for number types (including numpy's)
    confidence_str = f"{confidence:.2f}%" if isinstance(confidence, (float, np.number)) else confidence

    report = {"status": "No Significant Findings", "summary": "", "details": "", "confidence": confidence_str}

    if model_name == 'FLAIR Abnormality Segmentation':
        report["details"] = ("This model is designed to analyze Fluid-Attenuated Inversion Recovery (FLAIR) MRI sequences. "
                             "It is specifically trained to identify and segment hyperintense (bright) regions that are often "
                             "indicative of abnormalities such as those found in lower-grade gliomas (LGG).")
        if coverage > 0.1:
            report["status"] = "FLAIR Abnormality Detected"
            report["summary"] = (f"The analysis has identified and segmented a region of FLAIR hyperintensity covering approximately **{coverage:.2f}%** "
                                 f"of the brain area in this slice. Such findings are often associated with conditions like vasogenic edema, gliosis, "
                                 f"or demyelination, and are a key feature in the assessment of lower-grade gliomas. "
                                 f"The highlighted area on the mask represents the precise location and extent of the detected abnormality as per the AI model.")
        else:
            report["status"] = "No Significant FLAIR Abnormality Detected"
            report["summary"] = ("The AI model did not detect any significant areas of FLAIR hyperintensity within this MRI slice. "
                                 "The scan appears to be within the normal range as per the model's trained parameters for identifying "
                                 "abnormalities associated with lower-grade gliomas.")
    else: # Placeholder for other models
        report["details"] = f"This model is trained to identify markers for {model_name}."
        if coverage > 0.1:
            report["status"] = "Potential Markers Detected"
            report["summary"] = f"The model has highlighted areas of interest covering {coverage:.2f}% of the scan."
        else:
            report["summary"] = "No significant markers were detected by the model."
            
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
        self.multi_cell(0, 5, 'Disclaimer: This report is generated by an automated AI system for informational purposes only. It is not a medical diagnosis and should not be used as a substitute for consultation with a qualified healthcare professional. All findings are preliminary and require review by a certified radiologist.', 0, 'C')
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
            self.multi_cell(0, 7, str(value).replace('**', '')) # Safely convert value to string
            self.ln(2) 
    def add_analysis_section(self, title, report, original_img_path, mask_img):
        self.add_page()
        self.chapter_title(title)
        if "Not Available" in report["status"]:
            self.chapter_body({"Status": report["status"]}, key_color=(108, 117, 125))
            self.chapter_body({"Summary": report["summary"]})
        else:
            status_color = (200, 0, 0) if "Detected" in report["status"] or "Identified" in report["status"] else (0, 100, 0)
            self.chapter_body({"Status": report["status"], "AI Confidence": report["confidence"]}, key_color=status_color)
            self.chapter_body({"Summary": report["summary"], "Model Details": report["details"]})
            mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8).squeeze())
            temp_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_mask.png')
            mask_pil.save(temp_mask_path)
            y_before_images = self.get_y()
            if y_before_images > 160:
                 self.add_page()
                 y_before_images = self.get_y()
            img_width = 75
            img_gap = 10
            page_content_width = self.w - self.l_margin - self.r_margin
            start_x = self.l_margin + (page_content_width - (img_width * 2 + img_gap)) / 2
            img1_x = start_x
            img2_x = start_x + img_width + img_gap
            self.image(original_img_path, x=img1_x, w=img_width, y=y_before_images)
            self.image(temp_mask_path, x=img2_x, w=img_width, y=y_before_images)
            self.set_y(y_before_images + img_width + 5)
            self.set_font('Helvetica', 'I', 9)
            self.cell(0, 5, 'Left: Original Scan  |  Right: AI Model Segmentation Mask', 0, 1, 'C')
            os.remove(temp_mask_path)

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

    try:
        processed_image, original_pil_img = preprocess_image(filepath)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return redirect(url_for('index'))

    results_data = {}
    for name in EXPECTED_MODELS:
        if name in MODELS:
            try:
                model = MODELS[name]
                prediction = model.predict(processed_image)
                # Pass the raw prediction array to the interpretation function
                report = get_text_interpretation(name, prediction[0])
                mask_for_display = (prediction[0] > 0.5).astype(np.uint8)
                results_data[name] = {'report': report, 'mask_uri': encode_image_for_html(mask_for_display)}
            except Exception as e:
                results_data[name] = {'report': {"status": "Analysis Failed", "summary": f"An error occurred: {e}", "details": "", "confidence": "N/A"}, 'mask_uri': encode_image_for_html(np.array([]))}
        else:
            results_data[name] = {'report': {"status": "Model Not Available", "summary": "Model could not be loaded.", "details": "This analysis was skipped.", "confidence": "N/A"}, 'mask_uri': encode_image_for_html(np.array([]))}
    
    final_context = {
        'original_img_uri': encode_image_for_html(original_pil_img),
        'analysis': results_data
    }
    
    return render_template('results.html', results=final_context, uploaded_filename=filename)

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return redirect(url_for('index'))

    try:
        processed_image, _ = preprocess_image(filepath)
    except Exception as e:
        return redirect(url_for('index'))

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=25)

    for name in EXPECTED_MODELS:
        report_data = {}
        mask_array_for_pdf = np.array([])
        if name in MODELS:
            try:
                model = MODELS[name]
                prediction = model.predict(processed_image)
                report_data = get_text_interpretation(name, prediction[0])
                mask_array_for_pdf = (prediction[0] > 0.5).astype(np.uint8)
            except Exception as e:
                report_data = {"status": "Analysis Failed", "summary": f"An error occurred: {e}"}
        else:
             report_data = {"status": "Model Not Available", "summary": "Model could not be loaded."}
        
        pdf.add_analysis_section(name, report_data, filepath, mask_array_for_pdf)

    pdf_buffer = io.BytesIO(pdf.output())
    pdf_buffer.seek(0)
    
    return send_file(pdf_buffer, as_attachment=True, download_name='AI_Brain_Scan_Report.pdf', mimetype='application/pdf')

load_all_models()

if __name__ == '__main__':
    app.run(debug=True)
