// Function to check tasks and send notification
function checkTasks() {
  console.log('checkTasks function called');
  chrome.storage.sync.get({tasks: [], reminderFrequency: 30}, function(data) {
    console.log('Retrieved data:', data);
    const tasks = data.tasks;
    const uncompletedTasks = tasks.filter(task => !task.completed);
    console.log('Uncompleted tasks:', uncompletedTasks);
    
    if (uncompletedTasks.length > 0) {
      chrome.notifications.create({
        type: 'basic',
        title: 'Task Reminder',
        message: `You have ${uncompletedTasks.length} uncompleted tasks. Don't forget to finish them!`,
        iconUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=='
      }, function(notificationId) {
        if (chrome.runtime.lastError) {
          console.error('Notification error:', chrome.runtime.lastError);
        } else {
          console.log('Notification created with ID:', notificationId);
        }
      });
    } else {
      console.log('No uncompleted tasks to remind about');
    }
  });
}

// Set up alarm when extension is installed or updated
chrome.runtime.onInstalled.addListener(function() {
  console.log('Extension installed or updated');
  chrome.storage.sync.get({reminderFrequency: 30}, function(data) {
    console.log('Setting up alarm with frequency:', data.reminderFrequency);
    chrome.alarms.create('checkTasks', {periodInMinutes: data.reminderFrequency});
  });
  // Check tasks immediately after installation
  checkTasks();
});

// Listen for alarm
chrome.alarms.onAlarm.addListener(function(alarm) {
  console.log('Alarm fired:', alarm);
  if (alarm.name === 'checkTasks') {
    checkTasks();
  }
});

// Update alarm when reminder frequency changes
chrome.storage.onChanged.addListener(function(changes, namespace) {
  console.log('Storage changed:', changes);
  if (changes.reminderFrequency) {
    console.log('Reminder frequency changed to:', changes.reminderFrequency.newValue);
    chrome.alarms.clear('checkTasks', function() {
      chrome.alarms.create('checkTasks', {periodInMinutes: changes.reminderFrequency.newValue});
    });
  }
});

// For debugging: log when alarm is created or cleared
chrome.alarms.onAlarmCreated.addListener(function(alarm) {
  console.log('Alarm created:', alarm);
});

chrome.alarms.onAlarmCleared.addListener(function(alarmName) {
  console.log('Alarm cleared:', alarmName);
});

// Immediately check alarms
chrome.alarms.getAll(function(alarms) {
  console.log('Current alarms:', alarms);
});
