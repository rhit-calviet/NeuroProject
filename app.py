import os
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_file
from PIL import Image
from fpdf import FPDF
from datetime import datetime
from dicom_utils import convert_dicom_to_jpg
from chat import chat # <--- MODIFICATION 1: IMPORT THE CHAT FUNCTION

# --- Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SEGMENTATION_IMG_SIZE = 128
ALZHEIMER_CLASSIFICATION_IMG_SIZE = 224
TUMOR_CLASSIFICATION_IMG_SIZE = 256 
PDD_CLASSIFICATION_IMG_SIZE = 224

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_new_even_more_secret_key'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Dictionaries and Class Lists ---
EXPECTED_MODELS = {
    'Parkinson\'s Disease Dementia Classification': {'file': 'dementia.h5', 'type': 'classification_pdd'},
    'Alzheimer\'s Stage Classification': {'file': 'alzhiemer.keras', 'type': 'classification_alzheimer'},
    'Tumor Type Classification': {'file': 'tumor_old.keras', 'type': 'classification_tumor'}
}
MODELS = {}
ALZHEIMER_CLASSES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
TUMOR_CLASSES = [
    'Astrocitoma T1', 'Astrocitoma T2', 'Astrocitoma T1C+', 'Carcinoma T1', 'Carcinoma T2', 'Carcinoma T1C+', 'Ependimoma T1', 'Ependimoma T2', 'Ependimoma T1C+', 'Ganglioglioma T1', 'Ganglioglioma T2', 'Ganglioglioma T1C+', 'Germinoma T1', 'Germinoma T2', 'Germinoma T1C+', 'Glioblastoma T1', 'Glioblastoma T2', 'Glioblastoma T1C+', 'Granuloma T1', 'Granuloma T2', 'Granuloma T1C+', 'Meduloblastoma T1', 'Meduloblastoma T2', 'Meduloblastoma T1C+', 'Meningioma T1', 'Meningioma T2', 'Meningioma T1C+', 'Neurocitoma T1', 'Neurocitoma T2', 'Neurocitoma T1C+', 'Oligodendroglioma T1', 'Oligodendroglioma T2', 'Oligodendroglioma T1C+', 'Papiloma T1', 'Papiloma T2', 'Papiloma T1C+', 'Schwannoma T1', 'Schwannoma T2', 'Schwannoma T1C+', 'Tuberculoma T1', 'Tuberculoma T2', 'Tuberculoma T1C+', '_NORMAL T1', '_NORMAL T2'
]
PDD_CLASSES = ['Non-Demented', 'Very-Mild-Demented', 'Mild-Demented', 'Moderate-Demented', 'Severe-Demented']

# --- Model Loading Function ---
def load_all_models():
    print("--- Loading all available models... ---")
    for model_name, model_info in EXPECTED_MODELS.items():
        model_path = os.path.join(MODEL_FOLDER, model_info['file'])
        if os.path.exists(model_path):
            try:
                MODELS[model_name] = tf.keras.models.load_model(model_path, compile=False)
                print(f"Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
        else:
            print(f"Warning: Model file not found for '{model_name}'.")
    print("--- Model loading complete. ---")

# --- Preprocessing Functions ---
def preprocess_for_segmentation(image_path):
    img = Image.open(image_path).convert("RGB"); original_pil = img.copy()
    img = img.resize((SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE)); img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), original_pil

def preprocess_for_alzheimer(image_path):
    img = Image.open(image_path).convert("RGB"); original_pil = img.copy()
    img = img.resize((ALZHEIMER_CLASSIFICATION_IMG_SIZE, ALZHEIMER_CLASSIFICATION_IMG_SIZE)); img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), original_pil

def preprocess_for_tumor(image_path):
    img = Image.open(image_path).convert("RGB"); original_pil = img.copy()
    img = img.resize((TUMOR_CLASSIFICATION_IMG_SIZE, TUMOR_CLASSIFICATION_IMG_SIZE)); img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), original_pil

def preprocess_for_pdd(image_path):
    img = Image.open(image_path).convert("RGB"); original_pil = img.copy()
    img = img.resize((PDD_CLASSIFICATION_IMG_SIZE, PDD_CLASSIFICATION_IMG_SIZE)); img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), original_pil

# --- FIX: CORRECTED IMAGE ENCODING FUNCTION ---
def encode_image_for_html(image):
    """Encodes a PIL image or NumPy array into a base64 string for HTML display."""
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        if image.size == 0: 
            pil_img = Image.new('L', (128, 128), color=0) # Create a blank image for placeholders
        else:
            # Squeeze to remove single-channel dimension if it exists
            squeezed_array = np.squeeze(image)
            pil_img = Image.fromarray((squeezed_array * 255).astype(np.uint8))
    else:
        # It's already a PIL image
        pil_img = image

    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Interpretation Functions ---
def get_segmentation_interpretation(model_name, prediction_array):
    mask_array = (prediction_array > 0.5)
    coverage = np.mean(mask_array) * 100
    positive_pixels = prediction_array[mask_array]
    confidence = np.mean(positive_pixels) * 100 if positive_pixels.size > 0 else "N/A"
    confidence_str = f"{confidence:.2f}%" if isinstance(confidence, (float, np.number)) else confidence
    
    report = {"type": "segmentation", "confidence": confidence_str}
    if coverage > 0.1:
        report["status"] = "FLAIR Abnormality Detected"
        report["summary"] = f"A potential FLAIR abnormality covering {coverage:.2f}% of the area was detected."
        report["pdf_summary"] = (f"The analysis identified a region of FLAIR hyperintensity covering approximately {coverage:.2f}% of the brain area in this slice. "
                               "Such findings can be associated with conditions like vasogenic edema, gliosis, or demyelination, and are a key feature in the assessment of lower-grade gliomas.")
    else:
        report["status"] = "No Significant Abnormality Detected"
        report["summary"] = "The scan appears to be within the normal range for FLAIR abnormalities."
        report["pdf_summary"] = "The AI model did not detect any significant areas of FLAIR hyperintensity within this MRI slice according to its trained parameters."
    return report

# In app.py

def get_classification_interpretation(model_name, model_type, prediction_array):
    confidence = np.max(prediction_array) * 100
    predicted_index = np.argmax(prediction_array)
    report = {"type": "classification", "confidence": f"{confidence:.2f}%"}

    if model_type == 'classification_alzheimer':
        # ... (this part remains the same)
        predicted_class = ALZHEIMER_CLASSES[predicted_index]
        report["status"] = f"{predicted_class}"
        report["summary"] = f"The model predicts a stage of {predicted_class}."
        report["pdf_summary"] = f"Based on its analysis, the model classified this scan as showing features consistent with the {predicted_class} stage of Alzheimer's disease with a confidence of {confidence:.2f}%. This classification is based on volumetric and textural analysis of the brain scan."
    
    elif model_type == 'classification_pdd':
        # ... (this part remains the same)
        predicted_class = PDD_CLASSES[predicted_index]
        report["status"] = f"{predicted_class}"
        if predicted_class == 'Parkinson\'s Disease Dementia':
            report["summary"] = f"The model predicts features consistent with Parkinson's Disease Dementia (PDD)."
            report["pdf_summary"] = f"The model has classified this scan as potentially showing features consistent with Parkinson's Disease Dementia (PDD) with a confidence of {confidence:.2f}%. This is based on patterns learned from scans of patients with and without the condition."
        else:
            report["status"] = "Healthy Control"
            report["summary"] = "The model classifies this scan as a Healthy Control."
            report["pdf_summary"] = f"The model classified this scan as a Healthy Control with a confidence of {confidence:.2f}%, indicating the absence of features associated with Parkinson's Disease Dementia."
    
    # v-- MODIFY THIS ENTIRE BLOCK v--
    elif model_type == 'classification_tumor':
        # Get the full, detailed class name like "Astrocitoma T1"
        predicted_class_full = TUMOR_CLASSES[predicted_index]

        # Check if the prediction is a "normal" class first
        if "normal" in predicted_class_full.lower():
             report["status"] = "No Tumor Detected"
             report["summary"] = "The model classifies this scan as normal (no tumor detected)."
             report["pdf_summary"] = f"The model has classified this scan as '{predicted_class_full}' with a high confidence of {confidence:.2f}%, indicating the absence of the tumor types it is trained to identify."
             report["predicted_class"] = "Normal" # Set a simple name for the report
        else:
            # --- NEW LOGIC TO SIMPLIFY THE NAME ---
            # This splits "Astrocitoma T1" into ["Astrocitoma", "T1"] and takes the first part.
            base_tumor_name = predicted_class_full.split(' ')[0]

            # Populate the report with the simplified name
            report["status"] = f"Potential {base_tumor_name} Detected"
            report["summary"] = f"The model predicts a potential tumor type of {base_tumor_name}."
            # The PDF summary can retain the full detail for professional review
            report["pdf_summary"] = (f"The model has identified features consistent with a {base_tumor_name} tumor "
                                   f"(detailed classification: '{predicted_class_full}') with a confidence of {confidence:.2f}%. "
                                   "This classification is based on learned features from a diverse dataset of brain tumor MRIs.")
            report["predicted_class"] = base_tumor_name # Store the simplified name
    # ^-- END OF MODIFIED BLOCK --^
    
    # This logic now uses the simplified name set in the block above
    if "predicted_class" not in report:
        report["predicted_class"] = predicted_class

    return report

# --- PDF Generation Class ---
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16); self.cell(0, 10, 'AI-Powered Brain Scan Analysis Report', 0, 1, 'C')
        self.set_font('Helvetica', '', 10); self.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-20); self.set_font('Helvetica', 'I', 8)
        self.multi_cell(0, 5, 'Disclaimer: This report is generated by an automated AI system for informational purposes only. It is not a medical diagnosis and should not be used as a substitute for consultation with a qualified healthcare professional.', 0, 'C')
        self.set_y(-10); self.cell(0, 5, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14); self.cell(0, 10, title, 0, 1, 'L'); self.ln(5)
    def chapter_body(self, data, key_color=(0,0,0)):
        self.set_font('Helvetica', '', 11); key_width = 45 
        for key, value in data.items():
            start_y = self.get_y()
            self.set_font('Helvetica', 'B'); self.set_text_color(*key_color); self.multi_cell(key_width, 7, f"{key}:")
            self.set_text_color(0,0,0); self.set_xy(self.get_x() + key_width, start_y); self.set_font('Helvetica', '')
            self.multi_cell(0, 7, str(value).replace('**', '')); self.ln(2) 
    def add_patient_info(self, patient_info):
        self.add_page(); self.chapter_title("Patient Information")
        self.chapter_body(patient_info)

    # <--- MODIFICATION 2: ADD NEW METHOD FOR THE GENERATIVE SUMMARY ---
    def add_generative_summary(self, summary_text):
        self.ln(5) # Add some space after patient info
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Integrated Summary & Recommendations', 0, 1, 'L')
        self.ln(5)
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 7, summary_text.replace('*', '')) # Remove markdown for PDF
        self.ln(5)

    def add_analysis_section(self, title, report_data, original_img_path, mask_img=None):
        self.add_page(); self.chapter_title(title)
        report = report_data.get('report', {}) # Safely get the report dictionary
        if not report or "Not Available" in report.get("status", ""):
            self.chapter_body({"Status": report.get("status", "Not Available")}, key_color=(108, 117, 125))
            self.chapter_body({"Summary": report.get("summary", "Model could not be loaded.")})
            return
        
        is_positive = "Detected" in report.get("status", "") or "Demented" in report.get("predicted_class", "") or ("T1" in report.get("predicted_class", "") or "T2" in report.get("predicted_class", "")) and 'NORMAL' not in report.get("predicted_class", "")
        status_color = (200, 0, 0) if is_positive else (0, 100, 0)
        self.chapter_body({"Status": report.get("status"), "AI Confidence": report.get("confidence")}, key_color=status_color)
        self.chapter_body({"Summary": report.get("pdf_summary")})
        
        if report.get("type") == 'segmentation':
            original_pil = Image.open(original_img_path).convert("RGB"); resized_original_pil = original_pil.resize((SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE))
            temp_original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_original.png'); resized_original_pil.save(temp_original_path)
            mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8).squeeze()); temp_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_mask.png'); mask_pil.save(temp_mask_path)
            y_before_images = self.get_y()
            if y_before_images > 160: self.add_page(); y_before_images = self.get_y()
            img_width = 75; img_gap = 10; page_content_width = self.w - self.l_margin - self.r_margin
            start_x = self.l_margin + (page_content_width - (img_width * 2 + img_gap)) / 2; img1_x = start_x; img2_x = start_x + img_width + img_gap
            self.image(temp_original_path, x=img1_x, w=img_width, y=y_before_images); self.image(temp_mask_path, x=img2_x, w=img_width, y=y_before_images)
            self.set_y(y_before_images + img_width + 5); self.set_font('Helvetica', 'I', 9); self.cell(0, 5, 'Left: Original Scan  |  Right: AI Model Segmentation Mask', 0, 1, 'C')
            os.remove(temp_original_path); os.remove(temp_mask_path)
        elif report.get("type") == 'classification':
            self.ln(10); self.image(original_img_path, x=self.w / 2 - 45, w=90)


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(request.url)
    
    patient_info = {
        "Name": request.form.get('patient_name'),
        "Age": request.form.get('patient_age'),
        "Sex": request.form.get('patient_sex')
    }

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    analysis_filepath = filepath
    if filename.lower().endswith('.dcm'):
        try:
            analysis_filepath = convert_dicom_to_jpg(filepath, app.config['UPLOAD_FOLDER'])
        except Exception as e:
            error_context = {
                'analysis': {
                    'DICOM Conversion Error': {
                        'report': {
                            "status": "DICOM Processing Failed",
                            "summary": f"The uploaded DICOM file could not be processed. Error: {e}",
                            "type": "error",
                            "confidence": "N/A"
                        },
                        'mask_uri': None
                    }
                },
                'patient_info': patient_info,
                'original_img_uri': None
            }
            return render_template('results.html', results=error_context, uploaded_filename=filename)
    
    analysis_filename = os.path.basename(analysis_filepath)
    original_pil_img = Image.open(analysis_filepath).convert("RGB")
    display_img_resized = original_pil_img.resize((SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE))

    results_data = {}
    for name, model_info in EXPECTED_MODELS.items():
        if name in MODELS:
            try:
                model = MODELS[name]
                model_type = model_info['type']
                if model_type == 'segmentation':
                    processed_image, _ = preprocess_for_segmentation(analysis_filepath)
                    prediction = model.predict(processed_image)
                    report = get_segmentation_interpretation(name, prediction[0])
                    mask_for_display = (prediction[0] > 0.5).astype(np.uint8)
                    results_data[name] = {'report': report, 'mask_uri': encode_image_for_html(mask_for_display)}
                elif model_type == 'classification_alzheimer':
                    processed_image, _ = preprocess_for_alzheimer(analysis_filepath)
                    prediction = model.predict(processed_image)
                    report = get_classification_interpretation(name, model_type, prediction[0])
                    results_data[name] = {'report': report, 'mask_uri': None}
                elif model_type == 'classification_tumor':
                    processed_image, _ = preprocess_for_tumor(analysis_filepath)
                    prediction = model.predict(processed_image)
                    report = get_classification_interpretation(name, model_type, prediction[0])
                    results_data[name] = {'report': report, 'mask_uri': None}
                elif model_type == 'classification_pdd': # <-- ADD THIS BLOCK
                    processed_image, _ = preprocess_for_pdd(analysis_filepath)
                    prediction = model.predict(processed_image)
                    report = get_classification_interpretation(name, model_type, prediction[0])
                    results_data[name] = {'report': report, 'mask_uri': None} # No mask for classification
            except Exception as e:
                results_data[name] = {'report': {"status": "Analysis Failed", "summary": f"An error occurred: {e}", "type": "error"}, 'mask_uri': None}
        else:
            results_data[name] = {'report': {"status": "Model Not Available", "summary": "Model could not be loaded.", "type": "unavailable"}, 'mask_uri': None}
    
    final_context = {'original_img_uri': encode_image_for_html(display_img_resized), 'analysis': results_data, 'patient_info': patient_info}
    return render_template('results.html', results=final_context, uploaded_filename=analysis_filename)


# <--- MODIFICATION 3: HEAVILY UPDATED PDF DOWNLOAD ROUTE ---
@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return redirect(url_for('index'))
    
    patient_info = {
        "Name": request.args.get('name', 'N/A'),
        "Age": request.args.get('age', 'N/A'),
        "Sex": request.args.get('sex', 'N/A')
    }

    # --- Step 1: Rerun analysis to get fresh predictions for the prompt ---
    model_predictions = {}
    for name, model_info in EXPECTED_MODELS.items():
        report_data = {}
        if name in MODELS:
            try:
                model = MODELS[name]
                model_type = model_info['type']
                if model_type == 'segmentation':
                    processed_image, _ = preprocess_for_segmentation(filepath)
                    prediction = model.predict(processed_image)
                    report = get_segmentation_interpretation(name, prediction[0])
                    mask_array = (prediction[0] > 0.5).astype(np.uint8)
                    report_data = {'report': report, 'mask_img': mask_array}
                elif model_type == 'classification_alzheimer':
                    processed_image, _ = preprocess_for_alzheimer(filepath)
                    prediction = model.predict(processed_image)
                    report = get_classification_interpretation(name, model_type, prediction[0])
                    report_data = {'report': report}
                elif model_type == 'classification_tumor':
                    processed_image, _ = preprocess_for_tumor(filepath)
                    prediction = model.predict(processed_image)
                    report = get_classification_interpretation(name, model_type, prediction[0])
                    report_data = {'report': report}
                elif model_type == 'classification_pdd':
                    processed_image, _ = preprocess_for_pdd(filepath)
                    prediction = model.predict(processed_image)
                    report = get_classification_interpretation(name, model_type, prediction[0])
                    report_data = {'report': report}
            except Exception as e:
                report_data = {'report': {"status": "Analysis Failed", "summary": f"An error occurred: {e}", "type": "error"}}
        else:
            report_data = {'report': {"status": "Model Not Available", "summary": "Model could not be loaded.", "type": "unavailable"}}
        model_predictions[name] = report_data

    # --- Step 2: Build the prompt and call the chat function ---
    statuses = [
        model_predictions.get(name, {}).get('report', {}).get('status', 'N/A')
        for name in EXPECTED_MODELS.keys()
    ]
    
    prompt = (f"I have an MRI of a person that has '{statuses[0]}', '{statuses[1]}', and '{statuses[2]}'. "
              "Can you go into detail explaining their conditions and provide possible consequences and "
              "recommended next steps? This will be written in a medical report so please keep it formal, "
              "no need to do any introductions but just give some recomendations and stuff. keep it in 4 sections: "
              " Findings, Discussion, Possible Consequences, and Recomendations")

    try:
        generative_summary = chat(prompt)
    except Exception as e:
        print(f"Error calling the generative AI model: {e}")
        generative_summary = "An AI-generated summary could not be retrieved due to an error. Please refer to the detailed analysis on the following pages."

    # --- Step 3: Generate the PDF with the new summary ---
    pdf = PDF()
    pdf.add_patient_info(patient_info)
    pdf.add_generative_summary(generative_summary)

    # --- Step 4: Add the detailed model-by-model analysis pages ---
    for name, report_data in model_predictions.items():
        mask_img = report_data.get('mask_img', None)
        pdf.add_analysis_section(name, report_data, filepath, mask_img)
    
    pdf_buffer = io.BytesIO(pdf.output())
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name='AI_Brain_Scan_Report.pdf', mimetype='application/pdf')


load_all_models()

if __name__ == '__main__':
    app.run(debug=True)