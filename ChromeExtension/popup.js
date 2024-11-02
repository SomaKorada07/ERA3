document.addEventListener('DOMContentLoaded', function() {
  const addTaskButton = document.getElementById('addTask');
  const taskInput = document.getElementById('taskInput');
  const timeInput = document.getElementById('timeInput');
  const taskList = document.getElementById('taskList');

  loadTasks();

  addTaskButton.addEventListener('click', function() {
    const task = taskInput.value;
    const time = timeInput.value;
    if (task && time) {
      addTask(task, time);
      taskInput.value = '';
      timeInput.value = '';
    }
  });

  function addTask(task, time) {
    chrome.storage.sync.get({tasks: []}, function(data) {
      const tasks = data.tasks;
      tasks.push({task, time, completed: false});
      chrome.storage.sync.set({tasks: tasks}, function() {
        loadTasks();
      });
    });
  }

  function loadTasks() {
    chrome.storage.sync.get({tasks: []}, function(data) {
      const tasks = data.tasks;
      taskList.innerHTML = '';
      tasks.forEach(function(taskItem, index) {
        const li = document.createElement('li');
        li.innerHTML = `
          <input type="checkbox" ${taskItem.completed ? 'checked' : ''}>
          <span>${taskItem.task} - ${taskItem.time}</span>
          <button class="delete">Delete</button>
        `;
        li.querySelector('input[type="checkbox"]').addEventListener('change', function() {
          toggleTaskCompletion(index);
        });
        li.querySelector('.delete').addEventListener('click', function() {
          deleteTask(index);
        });
        taskList.appendChild(li);
      });
    });
  }

  function toggleTaskCompletion(index) {
    chrome.storage.sync.get({tasks: []}, function(data) {
      const tasks = data.tasks;
      tasks[index].completed = !tasks[index].completed;
      chrome.storage.sync.set({tasks: tasks}, function() {
        loadTasks();
      });
    });
  }

  function deleteTask(index) {
    chrome.storage.sync.get({tasks: []}, function(data) {
      const tasks = data.tasks;
      tasks.splice(index, 1);
      chrome.storage.sync.set({tasks: tasks}, function() {
        loadTasks();
      });
    });
  }
});
