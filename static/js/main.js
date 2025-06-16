document.addEventListener('DOMContentLoaded', () => {
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

    // Smooth scrolling for nav links
    document.querySelectorAll('a.nav-link[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});
