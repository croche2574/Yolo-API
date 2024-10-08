# Yolo-API
Item-FindAR computer vision backend. 

## Install and Run
 1. Requires Python 3.8 or later.
 2. Run `pip install -r requirements.txt`
 3. Place trained model in `./models/`. Certificates are included in `./certs/` for demo purposes. For future research or deployment you should regenerate them.
 4. Run `python.exe main.py`

## Forward-proxy with NGROK
Running the server on a local device with high-end graphics hardware and then connecting over a forward-proxy proved to be the easiest and cost-effective method for running the computer vision backend. Localtunnel is a free alternative to NGROK that allows for static domains, but uptime was patchy when we were testing. Ngrok requires a premium plan to register a domain, if using the free plan, remember to change the url to the one returned by the command below in `\src\features\augment_system\workers\sendImg.js` for the front-end before compiling with webpack: [Front-end Repo](https://github.com/croche2574/Item-FindAR).
 1. Install [NGROK](https://ngrok.com/download).
 2. Run the following command to start the proxy: `ngrok http https://localhost:8000 --domain ai.itemfindar.net`

## Troubleshooting
To check what frames the server is receiving, uncomment the line `img.save("./saved-images/file-" + time.strftime("%Y%m%d-%H%M%S") + ".png")` in `main.py`. Frames are stored in `./saved-images`.

If "Queue Full" is displayed, restart the server. The error message when you stop the server may provide some information to explain the cause.