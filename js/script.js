document.addEventListener('DOMContentLoaded', function() {
    // Add search functionality to symptoms
    const searchBox = document.createElement('div');
    searchBox.className = 'mb-3';
    searchBox.innerHTML = `
        <input type="text" id="symptomSearch" class="form-control" placeholder="Search symptoms...">
    `;
    document.querySelector('#symptomsContainer').parentNode.prepend(searchBox);
    
    document.getElementById('symptomSearch').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        document.querySelectorAll('.form-check').forEach(item => {
            const symptomText = item.textContent.toLowerCase();
            item.style.display = symptomText.includes(searchTerm) ? 'block' : 'none';
        });
    });

    // Form submission handler
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        const checkedSymptoms = document.querySelectorAll('input[name="symptoms"]:checked');
        if (checkedSymptoms.length === 0) {
            e.preventDefault();
            alert('Please select at least one symptom');
        }
    });
});