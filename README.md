# ECG-MONITOR - Arrythmia Categorizator
## Frontend - Running on Oracle Instance
Server running on Oracle instance with free tier, powered by Astro. Shows ecg signal an relevant info.
### What can do?
#### Front <hr/>
With run button can run predictive model and show ecg received signals from backend but processed by predictive system, this shows singals assigned to possible categorization, **class 0**, **class 1**, **class 2**, **class 3**, **class 4**.

#### Scripts <hr/>
This repository contains just two bash scripts which keeps active with pm2 commands:
- *pullGitRepo.sh* keeps  frontend with same backend url. ngrok uses a temporary tunnel, so in case of change this script make a pull from repository wich automatically upload the new url from backend.
#### How to run
```bash
pm2 start ./pullGitRepo.sh
```
- *star_front.sh* this just keep running the frontend, making it more monitoreable and easy to excute.

#### How to run
```bash
pm2 start ./startFront.sh
```

## TODOS
### MONITOR <hr/>
- [x] Show all data from back to front.

- [x] Show ECG signals on monitor.

- [x] Make run button functional, needs to run prediction and retrieve responses.

- [x] Manage upload and download model from remote desk to backend.

- [x] Manage prediction results from remote desk with model loaded.
  
- [x] Define data flow, scripts created with pm2.

- [ ] Make popups to fullscreen ECG signals.

- [ ] Add aditional health info.

- [ ] Add functions to measure ECG with a stopwatch.

- [ ] Make functions that show electrodes connections at real time.

- [ ] ~~Define data flow, create scripts on cron.~~

- [ ] Generate Charts and relevant info about prediction. --> *Maybe use Gemini API.*

### MODELS <hr/>
- [ ] Models shows low performance on certains classes, maybe needs to remove class 3.

- [ ] Class 4 need to be removed, those signals can't be identified and uses resources.

- [ ] Get more info about signals with P,QRS and T values.

- [ ] Redo models after changes.

* CNN model with images discarded, was to heavy.
  
* MLP model has the best perfomance.
  
* AutoEncoder has strange results.
