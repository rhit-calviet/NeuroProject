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

    // --- NEW: Interactive Modal Logic ---
    const modalOverlay = document.getElementById('model-modal');
    const modalBody = document.getElementById('modal-body');
    const closeModalBtn = document.getElementById('modal-close-btn');
    const modelCards = document.querySelectorAll('.model-card');

    const modalData = {
        tumor: {
            title: "Tumor Type Classification Model",
            description: "This model is a deep convolutional neural network (CNN), likely based on a proven architecture like VGG16, ResNet, or a custom-designed structure. It was trained on the 'Brain Tumor MRI Images 44 Classes' dataset from Kaggle. The training process involved feeding the model thousands of labeled MRI images, allowing it to learn the distinct visual features—such as texture, shape, and contrast patterns—that differentiate between 44 types of brain tumors, including astrocytoma, glioblastoma, and normal tissue.",
            performance: "The model's performance is evaluated based on its accuracy in correctly classifying new, unseen MRI images. Key metrics include training and validation accuracy/loss curves, which show how well the model learned and generalized, and a confusion matrix, which provides a detailed breakdown of correct and incorrect predictions for each tumor type.",
            img1: "https://placehold.co/400x300/e9ecef/495057?text=Accuracy/Loss+Graph",
            img2: "https://placehold.co/400x300/e9ecef/495057?text=Confusion+Matrix"
        },
        alzheimer: {
            title: "Alzheimer's Stage Classification Model",
            description: "This classification model utilizes a deep learning architecture trained on the 'Augmented Alzheimer MRI Dataset'. It has learned to identify subtle morphological changes in the brain that are characteristic of Alzheimer's disease progression, such as hippocampal atrophy and changes in cortical thickness. By analyzing these patterns, it can categorize a scan into one of four stages.",
            performance: "Performance is measured by its ability to correctly classify the stage of dementia. The accuracy/loss graphs illustrate the learning process, while the confusion matrix shows the model's performance in distinguishing between the 'Non Demented', 'Very Mild', 'Mild', and 'Moderate' stages. High accuracy in this task is crucial for its utility.",
            img1: "https://placehold.co/400x300/e9ecef/495057?text=Training+Curves",
            img2: "https://placehold.co/400x300/e9ecef/495057?text=Classification+Matrix"
        },
        flair: {
            title: "FLAIR Abnormality Segmentation Model",
            description: "This model is based on the U-Net architecture, which is the gold standard for biomedical image segmentation. It features an encoder-decoder structure with skip connections that allow it to capture both fine-grained detail and high-level context. It was trained on the 'LGG MRI Segmentation' dataset, learning to recognize and outline hyperintense (bright) pixels in FLAIR sequences, which are indicative of potential tumors.",
            performance: "The model's success is measured by its ability to accurately match the ground-truth masks provided by human experts. Key performance metrics include the Dice Coefficient and Intersection over Union (IoU), which quantify the overlap between the predicted mask and the actual abnormality. The training history shows the model's convergence towards accurate segmentation.",
            img1: "https://placehold.co/400x300/e9ecef/495057?text=IoU+Metric+Graph",
            img2: "https://placehold.co/400x300/e9ecef/495057?text=Sample+Segmentation"
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
