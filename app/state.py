from __future__ import annotations
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class SharedState:
    # --- image buffer ---
    image: np.ndarray = field(default_factory=lambda: np.zeros((240, 240, 3), dtype=np.uint8))
    image_lock: threading.Lock = field(default_factory=threading.Lock)

    # --- runtime flags ---
    running: bool = True
    reverse_flag: bool = True

    # --- target detection ---
    target_detected: bool = False
    target_center_x: Optional[int] = None
    size: Optional[int] = None
    target_class_id: Optional[int] = None   # 追加

    # --- control ---
    auto_enabled: bool = False
    last_cmd: str = ""
    telemetry: Dict[str, Any] = field(default_factory=dict)


STATE = SharedState()
