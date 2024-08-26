from fastapi import FastAPI, Request, WebSocket
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageOps
from asyncio import Queue, QueueFull, create_task
import uvicorn, json, os, time
from config.definitions import ROOT_DIR

model = YOLO('./models/mosaicXL-detect.pt')
keyfile = os.path.join(ROOT_DIR, 'certificates', 'private.key')
certfile = os.path.join(ROOT_DIR, 'certificates', 'certificate.crt')

app = FastAPI(ssl_keyfile=keyfile, ssl_certfile=certfile)


inc: int = 0

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # can alter with time
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def classLookup(c):
    try:
        return list(model.names.keys())[list(model.names.values()).index(c)]
    except:
        print("class not in list")

class Detector:
    def __init__(self, socket) -> None:
        self.queue: Queue = Queue(maxsize=10)
        self.width = 0
        self.height = 0
        self.socket: WebSocket = socket
        self.searchClasses = []

    async def receive(self):
        data = await self.socket.receive_bytes()
        try:
            d = json.loads(data.decode('utf-8'))
            print('d:', d)
            if "classes" in d:
                self.searchClasses = list(filter(lambda item: item is not None, list(map(classLookup, d["classes"]))))
                print("classes set: ", self.searchClasses)
            elif "height" in d:
                print('set height', d['height'])
                self.height = d["height"]
                self.width = d["width"]
        except Exception as e:
            print(e)
            try:
                self.queue.put_nowait(data)
            except QueueFull:
                print("Queue full.")
    
    async def start(self):
        try:
            while True:
                await self.receive()

        except Exception as e:
            print(e)
            # detect_task.cancel()
            # video_out.cancel()
            #await websocket.close()

    async def prediction(self):
        while True:
            bytes = await self.queue.get()
            print("detecting")
            try:
                img = Image.frombytes('RGBA', (self.width,self.height), bytes, 'raw')
            except:
                continue
            img = ImageOps.flip(img)
            # Save a copy of the frame for debug purposes
            # img.save("./saved-images/file-" + time.strftime("%Y%m%d-%H%M%S") + ".png")
                
            results = model.track(img, persist=True, classes=self.searchClasses, show=False)

            for r in results:
                namedict = r.names
                result_list = []
                for idx, x in enumerate(r.boxes.data):
                    #print(r.boxes.xywhn)
                    try:
                        result = {
                            'id': int(r.boxes.id[idx].item()),
                            'cls': int(r.boxes.cls[idx].item()),
                            'clsname': namedict[int(r.boxes.cls[idx])],
                            'imgw': r.boxes.orig_shape[1],
                            'imgh': r.boxes.orig_shape[0],
                            'x': r.boxes.xywhn[idx][0].item(),
                            'y': r.boxes.xywhn[idx][1].item(),
                            'w': r.boxes.xywhn[idx][2].item(),
                            'h': r.boxes.xywhn[idx][3].item()
                        }
                        result_list.append(result)
                    except:
                        print("empty")
                result_json = json.dumps(result_list)
                await self.socket.send_json(result_json)

@app.websocket("/detect")
async def detect(websocket: WebSocket):
    await websocket.accept()
    logger.debug("connected")

    
    detector = Detector(websocket)
    start = create_task(detector.start())
    predict = create_task(detector.prediction()) 
    
    await start
    await predict

@app.get("/")
async def get_home(request: Request):
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, log_level="debug", reload=True, 
                ssl_keyfile=keyfile, ssl_certfile=certfile, ws="websockets", ws_ping_interval=None)