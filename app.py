import os
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from PIL import Image
from fpdf import FPDF

# --- Configuration ---
# Suppress TensorFlow GPU messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Define image dimensions (must match the model's expected input)
IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- Flask App Initialization ---
app = Flask(__name__)
# A secret key is needed for session management
app.config['SECRET_KEY'] = 'your_super_secret_key'
# Define paths
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Loading ---
# We load models once at startup to avoid slow loading on each request.
MODELS = {}

def load_all_models():
    """Loads all .h5 models from the models directory into a global dictionary."""
    print("--- Loading all models, this may take a moment... ---")
    model_files = {
        'tumor': 'tumor.h5',
        'alzheimer': 'alzheimer.h5',
        'mri_segmentation': 'mri.h5'
    }
    for model_name, file_name in model_files.items():
        model_path = os.path.join(MODEL_FOLDER, file_name)
        if os.path.exists(model_path):
            try:
                MODELS[model_name] = tf.keras.models.load_model(model_path)
                print(f"✔️  Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"❌ Error loading model {model_name}: {e}")
        else:
            print(f"⚠️  Warning: Model file not found at {model_path}")
    print("--- Model loading complete. ---")

# --- Helper Functions ---
def preprocess_image(image_path):
    """Loads and preprocesses an image for model prediction."""
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array /= 255.0  # Normalize to [0, 1]
    img_batch = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_batch, img # Return original PIL image for display

def encode_image_for_html(image):
    """Encodes a PIL image or NumPy array into a base64 string for HTML display."""
    if isinstance(image, np.ndarray):
        # Handle NumPy array (like the mask)
        image = Image.fromarray((image * 255).astype(np.uint8).squeeze())

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_text_interpretation(model_name, mask_array):
    """
    Generates a simple text interpretation based on the presence of segmented pixels.
    *** YOU SHOULD CUSTOMIZE THE TEXT LOGIC AND CONTENT HERE ***
    """
    # Calculate the percentage of the mask that is "active" (white pixels)
    coverage = np.mean(mask_array > 0.5) * 100

    if model_name == 'tumor':
        if coverage > 0.1: # Threshold for detection
            return (
                "The model has identified a region of interest that may correspond to a tumor. "
                "The highlighted area in the segmentation mask indicates the potential location. "
                "This is a preliminary analysis and not a medical diagnosis."
            )
        else:
            return "No significant regions of interest corresponding to a tumor were detected by the model."

    elif model_name == 'alzheimer':
        if coverage > 0.5: # Example threshold
             return (
                "The model has highlighted areas potentially associated with changes seen in Alzheimer's disease, "
                "such as plaque concentration or hippocampal atrophy. "
                "This is an analytical result for research purposes, not a clinical diagnosis."
            )
        else:
            return "The model did not detect significant markers associated with Alzheimer's disease in this scan."

    elif model_name == 'mri_segmentation':
        return (
            "This is a general segmentation model that identifies different brain structures. "
            "The output mask highlights the primary brain tissue region, separating it from the skull and background. "
            "This is often a pre-processing step for further analysis."
        )
    return "No interpretation available for this model."

# --- PDF Generation Class ---
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'AI Brain Scan Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, 'Disclaimer: For informational purposes only. Not a medical diagnosis.', 0, 0, 'R')

    def add_analysis_section(self, title, interpretation, original_img_path, mask_img):
        self.add_page()
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, interpretation)
        self.ln(10)
        
        # Save mask temporarily to be used in PDF
        mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8).squeeze())
        temp_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_mask.png')
        mask_pil.save(temp_mask_path)
        
        # Add images side-by-side
        self.image(original_img_path, x=20, w=80)
        self.image(temp_mask_path, x=110, w=80)
        self.set_font('Helvetica', 'I', 9)
        self.text(50, self.get_y() + 85, 'Original Scan')
        self.text(145, self.get_y() + 85, 'Model Output Mask')
        os.remove(temp_mask_path) # Clean up temp file

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # Clear any previous results from the session
    session.pop('results', None)
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the uploaded image
        processed_image, original_pil_img = preprocess_image(filepath)
        
        results = {}
        # Run inference for each loaded model
        for name, model in MODELS.items():
            prediction = model.predict(processed_image)
            mask = (prediction[0] > 0.5).astype(np.uint8) # Binarize the output mask
            
            results[name] = {
                'mask_uri': encode_image_for_html(mask),
                'interpretation': get_text_interpretation(name, mask),
                'raw_mask': mask.tolist() # Store raw mask for PDF
            }

        # Store results in session for PDF generation
        session['results'] = {
            'original_img_uri': encode_image_for_html(original_pil_img),
            'analysis': results,
            'original_img_path': filepath
        }
        
        return redirect(url_for('show_results'))

@app.route('/results')
def show_results():
    results = session.get('results')
    if not results:
        return redirect(url_for('index'))
    return render_template('results.html', results=results)

@app.route('/download_pdf')
def download_pdf():
    results = session.get('results')
    if not results:
        return redirect(url_for('index'))
    
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # *** CUSTOMIZE THE PDF CONTENT HERE ***
    for model_name, analysis_result in results['analysis'].items():
        title = model_name.replace('_', ' ').title() + " Analysis"
        interpretation = analysis_result['interpretation']
        mask_array = np.array(analysis_result['raw_mask'])
        
        pdf.add_analysis_section(
            title,
            interpretation,
            results['original_img_path'],
            mask_array
        )

    # Generate PDF in memory
    pdf_buffer = io.BytesIO(pdf.output())
    pdf_buffer.seek(0)
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name='brain_scan_report.pdf',
        mimetype='application/pdf'
    )

# --- Main Execution ---
if __name__ == '__main__':
    load_all_models() # Load models before starting the server
    app.run(debug=True)
