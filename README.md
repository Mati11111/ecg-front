# ECG-MONITOR - Arrythmia Categorizator 
## Backend - Server RaspberryPI
Server running locally on RaspberryPI with fastapi on Python. Tunneled with ngrok. Current URL [here](https://github.com/Mati11111/ecg-front/blob/master/src/assets/ngrok_link.txt).
### Endpoints
- bpm: Shows bpm with a timestamp.
- ecg: Shows ecg data received.
- health: Shows backend status.
- ws 
- test_signal
- activar_escritura

## Frontend - Running on Oracle Instance
Server running on Oracle instance with free tier, powered by Astro. Shows ecg signal an relevant info.

## TODOS
### MONITOR <hr/>
- [ ] Show all data from back to front.

- [ ] Show ECG signals on monitor.

- [ ] Make popups to fullscreen ECG signals.

- [ ] Add aditional health info.

- [ ] Add functions to measure ECG with a stopwatch.

- [ ] Make run button functional, needs to run prediction and retrieve responses.

- [ ] Make functions that show electrodes connections at real time.

- [ ] Manage upload and download model from remote desk to backend.

- [ ] Manage prediction results from remote desk with model loaded.

- [ ] Define data flow, create scripts on cron.

- [ ] Generate Charts and relevant info about prediction. --> *Maybe use Gemini API.*

### MODELS <hr/>
- [ ] Models shows low performance on certains classes, maybe needs to remove class 3.

- [ ] Class 4 need to be removed, those signals can't be identified and uses resources.

- [ ] Get more info about signals with P,QRS and T values.

- [ ] Redo models after changes.

* CNN model with images discarded, was to heavy.
  
* MLP model has the best perfomance.
  
* AutoEncoder has strange results.
