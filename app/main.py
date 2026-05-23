from __future__ import annotations
import time
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np

from .state import STATE
from .esp32_udp import Esp32Udp, start_receiver
from .vision import Vision, bgr_to_jpeg_bytes

templates = Jinja2Templates(directory="templates")
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# =======================
# IP設定
# =======================
LOCAL_UDP_IP = "0.0.0.0"
SHARED_UDP_PORT = 50000
ESP32_UDP_IP = "192.168.1.1"
ESP32_UDP_PORT = 55555

CENTER_THRESHOLD = 60
SEND_INTERVAL = 0.15
FRAME_CENTER_X = 240 // 2

# フレーム比較間隔
FRAME_COMPARE_INTERVAL = 2


droidcam = None
esp = None

vision = Vision(model_path="models/stools6-11s.pt", priority_class_ids=[0,1])

# =======================
# 画像一致度
# =======================
prev_frame = None
prev_frame_time = 0
commone_rate = -1

# スタックカウント
stuck_count = 0

# =======================
# 初期化
# =======================
@app.on_event("startup")
async def on_startup():
    global esp, cmd_sink

    esp = Esp32Udp("0.0.0.0", 50000, "192.168.1.1", 55555)
    start_receiver(esp)
    cmd_sink = esp

    vision.start_infer_loop(fps_limit=15.0)

    asyncio.create_task(auto_control_loop())


@app.on_event("shutdown")
async def on_shutdown():
    STATE.running = False
    vision.stop()
    if droidcam is not None:
        droidcam.stop()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/status")
def status():
    return JSONResponse(
        {
            "auto_enabled": STATE.auto_enabled,
            "reverse_flag": STATE.reverse_flag,
            "last_cmd": STATE.last_cmd,
            "target_detected": STATE.target_detected,
            "target_center_x": STATE.target_center_x,
            "target_classID":STATE.target_class_id,
            "telemetry": STATE.telemetry,
            "image_match": commone_rate,
            "size":STATE.size,
        }
    )


@app.post("/reset")
def reset():
    STATE.auto_enabled = False
    STATE.last_cmd = ""
    return JSONResponse({"ok": True})


# ============================
# 画像一致度計算
# ============================
def calc_image_similarity(img1, img2):

    img1 = cv2.resize(img1, (160, 160))
    img2 = cv2.resize(img2, (160, 160))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    score = diff.mean()

    similarity = 1 - (score / 255)

    return float(similarity)


# ============================
# MJPEG映像
# ============================
def mjpeg_generator():

    global prev_frame, prev_frame_time, commone_rate

    while STATE.running:

        with STATE.image_lock:
            frame = STATE.image.copy()

        if STATE.reverse_flag:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        vision.set_latest_frame(frame)

        annotated = vision.get_latest_annotated()
        if annotated is None:
            annotated = frame

        now = time.time()

        # フレーム一致度更新
        if prev_frame is None:

            prev_frame = annotated.copy()
            prev_frame_time = now

        elif now - prev_frame_time > FRAME_COMPARE_INTERVAL:

            commone_rate = calc_image_similarity(prev_frame, annotated)

            prev_frame = annotated.copy()
            prev_frame_time = now

        jpg = bgr_to_jpeg_bytes(annotated, 500, 500)

        if not jpg:
            time.sleep(0.01)
            continue

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

        time.sleep(0.03)


@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================
# WebSocket
# ============================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):

    await ws.accept()

    async def status_sender():
        while True:

            await asyncio.sleep(0.1)
            await ws.send_json(
                {
                    "auto_enabled": STATE.auto_enabled,
                    "reverse_flag": STATE.reverse_flag,
                    "last_cmd": STATE.last_cmd,
                    "target_detected": STATE.target_detected,
                    "target_center_x": STATE.target_center_x,
                    "target_classID":STATE.target_class_id,
                    "telemetry": STATE.telemetry,
                    "image_match": commone_rate,
                    "size":STATE.size,
                }
            )

    sender_task = asyncio.create_task(status_sender())

    try:
        while True:

            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "manual":

                cmd = str(data.get("cmd", "")).upper()

                if not STATE.auto_enabled:
                    cmd_sink.send_command(cmd)

                await ws.send_json({"ok": True})

            elif msg_type == "auto":

                STATE.auto_enabled = bool(data.get("enabled", False))

                await ws.send_json({"ok": True, "auto_enabled": STATE.auto_enabled})

            elif msg_type == "reverse":

                STATE.reverse_flag = bool(data.get("enabled", True))

                await ws.send_json({"ok": True, "reverse_flag": STATE.reverse_flag})
            
            else:

                await ws.send_json({"ok": False, "error": "unknown type"})

    except WebSocketDisconnect:
        sender_task.cancel()
        return



# スタック検出
STUCK_THRESHOLD = 0.95
STUCK_COUNT_LIMIT = 50

# 回避動作(要調整)
BACK_TIME = 1000
ROTATE_TIME = 2000
FORWARD_TIME=500
ROTATE_TIME2 = 2000
FORWARD_TIME2=1000


#クリップ定数
CLIP_CENTER_THRESHOLD= 50
CLIP_FORWARD_TIME=0.5

# ============================
# 自動運転
# ============================
async def auto_control_loop():

    global stuck_count

    while STATE.running:

        await asyncio.sleep(0.01)

        if not STATE.auto_enabled:
            continue


        #衝突関連処理
        if commone_rate > STUCK_THRESHOLD:
            stuck_count += 1
        else:
            stuck_count = 0

        print(stuck_count,"-----------------------")
        if stuck_count > STUCK_COUNT_LIMIT:
            stuck_count = 0

            print("STUCK DETECTED")

            c = 0
            # ① 後退
            while c < BACK_TIME:
                c += 1 
                cmd_sink.send_command("S")
                print("BACK")

            c = 0
            while c < ROTATE_TIME:
                c += 1
                cmd_sink.send_command("D")
                print("ROTATE1")

            c =0
            while c < FORWARD_TIME:
                c += 1 
                cmd_sink.send_command("W")
                print("FORWARD")


            c = 0
            while c < ROTATE_TIME2:
                c += 1
                cmd_sink.send_command("A")
                print("ROTATE2")
            
            
            c =0
            while c < FORWARD_TIME2:
                c += 1 
                cmd_sink.send_command("W")
                print("FORWARD")


            continue

        
        clip_detected = STATE.target_class_id==0
        clip_center_x = STATE.target_center_x

         # 2) clip優先
        if clip_detected and clip_center_x is not None:
            clip_diff = clip_center_x - FRAME_CENTER_X

            if abs(clip_diff) <= CLIP_CENTER_THRESHOLD:
                cmd_sink.send_command("W")
                await asyncio.sleep(CLIP_FORWARD_TIME)
            else:
                if clip_diff > 0:
                    cmd_sink.send_command("D")
                else:
                    cmd_sink.send_command("A")
            continue

        # =========================
        # 椅子追従
        # ========================

        if STATE.target_detected and STATE.target_center_x is not None:

            diff = STATE.target_center_x - FRAME_CENTER_X

            if abs(diff) <= CENTER_THRESHOLD:

                cmd_sink.send_command("W")

            else:

                if diff > 0:
                    cmd_sink.send_command("D")
                else:
                    cmd_sink.send_command("A")

        else:

            cmd_sink.send_command("D")

