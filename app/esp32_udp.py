from __future__ import annotations
import socket
import time
import struct
import threading
import numpy as np

from .state import STATE

class Esp32Udp:
    def __init__(self, local_ip: str, shared_port: int, esp_ip: str, esp_port: int):
        self.local_ip = local_ip
        self.shared_port = shared_port
        self.esp_ip = esp_ip
        self.esp_port = esp_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.shared_port))
        self.sock.settimeout(0.5)

    def send_command(self, command: str) -> None:
        self.sock.sendto(command.encode(), (self.esp_ip, self.esp_port))
        STATE.last_cmd = command

    def receive_telemetry(self):
        try:
            data, _ = self.sock.recvfrom(2048)
        except socket.timeout:
            return None, None, None
        packet = np.frombuffer(data, dtype=np.uint8)
        return int(packet[0]), packet[1:-1], int(packet[-1])

    @staticmethod
    def telemetry_reader(data: np.ndarray) -> None:
        # data[0] はすでに除かれている想定（呼び出し側で合わせる）
        if len(data) != 13:
            return
        pressure, temperature, humidity = struct.unpack('<fff', data[1:13])
        STATE.telemetry = {
            "pressure": float(pressure),
            "temperature": float(temperature),
            "humidity": float(humidity),
            "ts": time.time(),
        }

    @staticmethod
    def image_decode(packet_number: int, data: np.ndarray, img: np.ndarray) -> None:
        index = (packet_number - 1) // 3 * 12
        rgb = 2 - (packet_number - 1) % 3
        img[index:index+12, :, rgb] = (
            np.reshape(
                np.ravel(np.array([data // 16, data % 16]).T),
                (12, 240)
            ) * 16
        )

def start_receiver(esp: Esp32Udp) -> threading.Thread:
    def loop():
        while STATE.running:
            packet_number, data, _ = esp.receive_telemetry()
            if packet_number is None:
                continue
            if packet_number == 0x5C:
                Esp32Udp.telemetry_reader(data)
            elif packet_number == 0xFF:
                pass
            else:
                with STATE.image_lock:
                    Esp32Udp.image_decode(packet_number, data, STATE.image)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th
