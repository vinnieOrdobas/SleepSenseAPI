document.addEventListener('DOMContentLoaded', function() {
    // Initialize Materialize select elements
    var selectElems = document.querySelectorAll('select');
    M.FormSelect.init(selectElems);

    // Initialize Materialize modal
    var modalElems = document.querySelectorAll('.modal');
    M.Modal.init(modalElems);

    // Attach the event listener to the submit button
    document.getElementById('healthForm').addEventListener('submit', showDisclaimerModal);

    // Initialize height fields based on the selected unit
    handleHeightUnitChange();
});

function showDisclaimerModal(event) {
    event.preventDefault();
    if (validateForm()) {
        var modal = document.getElementById('disclaimerModal');
        var instance = M.Modal.getInstance(modal);
        instance.open();
    }
}

function submitForm() {
    var heightUnit = document.getElementById('heightUnit').value;
    if (heightUnit === 'feet') {
        convertHeight();
    }
    var weightUnit = document.getElementById('weightUnit').value;
    if (weightUnit === 'lbs') {
        convertWeight();
    }
    document.getElementById('healthForm').submit();
}

function isInteger(value) {
    return Number.isInteger(parseFloat(value));
}

function validateField(field, isValid) {
    if (isValid) {
        field.classList.remove('invalid');
        field.classList.add('valid');
    } else {
        field.classList.remove('valid');
        field.classList.add('invalid');
    }
}

function validateForm() {
    var heartRateField = document.getElementById('heart_rate');
    var systolicField = document.getElementById('systolic');
    var diastolicField = document.getElementById('diastolic');

    var isHeartRateValid = isInteger(heartRateField.value);
    var isSystolicValid = isInteger(systolicField.value);
    var isDiastolicValid = isInteger(diastolicField.value);

    validateField(heartRateField, isHeartRateValid);
    validateField(systolicField, isSystolicValid);
    validateField(diastolicField, isDiastolicValid);

    // Validate height based on selected unit
    var heightUnit = document.getElementById('heightUnit').value;
    var heightField = document.getElementById('height');
    var isHeightValid = true;

    if (heightUnit === 'cm') {
        isHeightValid = !isNaN(parseFloat(heightField.value));
    } else if (heightUnit === 'feet') {
        // No specific validation needed for feet; it's converted to cm before submission
    }

    validateField(heightField, isHeightValid);
    return isHeartRateValid && isSystolicValid && isDiastolicValid && isHeightValid;
}

function convertWeight() {
    var weight = document.getElementById('weight').value;
    var convertedWeight = weight * 0.453592; // lbs to kg conversion
    document.getElementById('weight').value = convertedWeight.toFixed(2);
}

function convertHeight() {
    var height = parseFloat(document.getElementById('height').value) || 0;
    var heightCm = height * 30.48; // feet to cm conversion
    document.getElementById('height').value = heightCm.toFixed(2);
}

function handleHeightUnitChange() {
    var heightUnit = document.getElementById('heightUnit').value;
    if (heightUnit === 'feet') {
        convertHeight();
    }
}
