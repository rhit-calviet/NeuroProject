document.addEventListener('DOMContentLoaded', () => {
    // --- Form submission UX ---
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        const fileInput = document.getElementById('file-input');
        const fileNameDisplay = document.getElementById('file-name-display');
        const uploadBox = document.getElementById('upload-box');
        const loadingState = document.getElementById('loading-state');

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = fileInput.files[0].name;
            } else {
                fileNameDisplay.textContent = 'No file selected.';
            }
        });

        uploadForm.addEventListener('submit', () => {
            if (uploadBox && loadingState) {
                uploadBox.classList.add('hidden');
                loadingState.classList.remove('hidden');
            }
        });
    }

    // --- Smooth scrolling for nav links ---
    document.querySelectorAll('a.nav-link[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // --- Interactive Modal Logic ---
    const modalOverlay = document.getElementById('model-modal');
    const modalBody = document.getElementById('modal-body');
    const closeModalBtn = document.getElementById('modal-close-btn');
    const modelCards = document.querySelectorAll('.model-card');

    const modalData = {
        tumor: {
            title: "Tumor Type Classification Model",
            description: "This model is a custom-built Convolutional Neural Network (CNN). The architecture consists of three convolutional layers with increasing filter sizes (32, 64, 128) followed by MaxPooling layers to extract key features from the images. A Dropout layer is included to prevent overfitting before the final Dense layers classify the scan into one of 15 consolidated tumor types, including a 'Normal' category.",
            performance: "The model was trained on the 'Brain Tumor MRI Images' dataset, which was split into training, validation, and test sets. To improve robustness, the training data was enhanced with data augmentation techniques, including random rotations, shifts, shears, and zooms. Performance was tracked by monitoring accuracy and loss on the validation set during training.",
            img1: "../static/images/training_plot_tumor.png",
            img2: "../static/images/confusion_matrix_tumor.png"
        },
        alzheimer: {
            title: "Alzheimer's Stage Classification Model",
            description: "This model utilizes transfer learning based on the VGG16 architecture, pre-trained on the ImageNet dataset. The original convolutional base is used as a feature extractor, with its initial layers frozen. A custom classification head—consisting of Dense, Batch Normalization, and Dropout layers—was added and trained on the Alzheimer's MRI dataset. For fine-tuning, the final three convolutional layers of the VGG16 base were unfrozen to adapt more closely to the specific features of MRI scans.",
            performance: "The model was trained on the 'Augmented Alzheimer MRI Dataset', which categorizes scans into four stages of the disease. The model's learning rate was adjusted dynamically using a ReduceLROnPlateau callback, and EarlyStopping was used to prevent overfitting. Performance is evaluated by its accuracy in classifying the stage of dementia and is visualized with accuracy/loss graphs and a detailed classification report.",
            img1: "../static/images/Training_History_Alz.png",
            img2: "../static/images/Confusion_Matrix_Alz.png"
        },
        pdd: {
            title: "Parkinson's Disease Dementia (PDD) Model",
            description: "This model is a custom-designed Convolutional Neural Network (CNN) built to perform multi-class classification. The architecture includes a series of three Conv2D and MaxPooling layers to downsample the image and learn hierarchical features. A Flatten layer prepares the data for a final classification block, which uses a Dense layer and a Dropout layer for regularization before outputting a prediction for one of five stages.",
            performance: "The model was trained on the 'Parkinson's Disease Dementia (PDD)' dataset to distinguish between 'Healthy Control' and 'PDD' scans. It was compiled with the Adam optimizer and categorical cross-entropy loss function. The model's performance on the validation set was monitored throughout training to ensure it learned to generalize effectively from the training data.",
            img1: "../static/images/training_plot_dem.png",
            img2: "../static/images/confusion_matrix_dem.png"
        }
    };

    function openModal(modelKey) {
        const data = modalData[modelKey];
        if (!data) return;

        modalBody.innerHTML = `
            <h2>${data.title}</h2>
            <h3>Model Architecture and Training</h3>
            <p>${data.description}</p>
            <h3>Performance and Evaluation</h3>
            <p>${data.performance}</p>
            <div class="modal-image-grid">
                <img src="${data.img1}" alt="Performance Graph">
                <img src="${data.img2}" alt="Evaluation Matrix">
            </div>
        `;
        modalOverlay.classList.remove('hidden');
    }

    function closeModal() {
        modalOverlay.classList.add('hidden');
    }

    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            openModal(card.dataset.model);
        });
    });

    closeModalBtn.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            closeModal();
        }
    });
});
