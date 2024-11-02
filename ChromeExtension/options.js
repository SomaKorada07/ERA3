document.addEventListener('DOMContentLoaded', function() {
  const reminderFrequencyInput = document.getElementById('reminderFrequency');
  const saveButton = document.getElementById('save');

  chrome.storage.sync.get({reminderFrequency: 30}, function(data) {
    reminderFrequencyInput.value = data.reminderFrequency;
  });

  saveButton.addEventListener('click', function() {
    const reminderFrequency = parseInt(reminderFrequencyInput.value);
    if (reminderFrequency > 0) {
      chrome.storage.sync.set({reminderFrequency: reminderFrequency}, function() {
        alert('Options saved');
      });
    } else {
      alert('Please enter a valid number greater than 0');
    }
  });
});
