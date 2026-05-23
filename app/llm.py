from __future__ import annotations
import json
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

ALLOWED_ACTIONS = {
    "auto_on", "auto_off",
    "send",          # 単発コマンド送信
    "set_reverse_on", "set_reverse_off",
}

ALLOWED_CMDS = {"W","A","S","D","M","R","G","T","B","STOP"}

SYSTEM = """あなたはロボット制御の指示を「安全なJSON」に変換する。
出力は必ず次のどれか:
1) {"action":"auto_on"}
2) {"action":"auto_off"}
3) {"action":"send","cmd":"W|A|S|D|M|R|G|T|B|STOP"}
4) {"action":"set_reverse_on"}
5) {"action":"set_reverse_off"}

追加の文章は禁止。JSONのみ。"""

def build_llm() -> Llama:
    repo = "SakanaAI/TinySwallow-1.5B-Instruct-GGUF"
    filename = "tinyswallow-1.5b-instruct-q5_k_m.gguf"
    path = hf_hub_download(repo_id=repo, filename=filename)

    return Llama(
        model_path=path,
        chat_format="qwen",
        n_ctx=2048,
        n_threads=8,
        n_batch=512,
        n_gpu_layers=0,  # Metal有効なら -1 を検討
        f16_kv=True,
        verbose=False,
    )

def parse_json_only(text: str) -> dict:
    # JSON以外が混ざっても最初の {...} を抜く
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {"action": "send", "cmd": "STOP"}  # 迷ったら安全側
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"action": "send", "cmd": "STOP"}

def nl_to_action(llm: Llama, user_jp: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_jp},
    ]
    res = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
    )
    out = res["choices"][0]["message"]["content"].strip()
    data = parse_json_only(out)

    action = data.get("action")
    if action not in ALLOWED_ACTIONS:
        return {"action": "send", "cmd": "STOP"}

    if action == "send":
        cmd = str(data.get("cmd", "")).upper()
        if cmd not in ALLOWED_CMDS:
            return {"action": "send", "cmd": "STOP"}
        return {"action": "send", "cmd": cmd}

    return {"action": action}
