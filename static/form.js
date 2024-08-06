document.addEventListener('DOMContentLoaded', function() {
    // Initialize Materialize select elements
    var selectElems = document.querySelectorAll('select');
    M.FormSelect.init(selectElems);

    // Initialize Materialize modal
    var modalElems = document.querySelectorAll('.modal');
    M.Modal.init(modalElems);
});

function showDisclaimerModal(event) {
    event.preventDefault();
    var modal = document.getElementById('disclaimerModal');
    var instance = M.Modal.getInstance(modal);
    instance.open();
}

function submitForm() {
    document.getElementById('healthForm').submit();
}

// Attach the event listener to the submit button
document.getElementById('healthForm').addEventListener('submit', showDisclaimerModal);