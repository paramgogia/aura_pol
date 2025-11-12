# server/server.py
# Flask server to accept webcam image + wallet signature + nonce, run ezkl.prove, and return proof artifacts.
import os
import json
import subprocess
from flask import Flask, request, jsonify
from utils_preprocess import preprocess_base64_image_to_nhwc
from eth_account.messages import encode_defunct
from eth_account.account import Account
import numpy as np

app = Flask(__name__)

# --- CONFIG: adjust these paths if you placed files elsewhere ---
EZKL_BIN = "ezkl"  # full path if not in PATH
MODEL_ONNX = os.path.join("..", "model", "model.onnx")
SETTINGS_JSON = os.path.join("..", "zk", "settings.json")
PK_KEY = os.path.join("..", "zk", "pk.key")
COMPILED_JSON = os.path.join("..", "zk", "compiled.json")
# outputs produced in server working dir: input_runtime.json, proof.json, calldata.txt

SCALE = 1000000  # 1e6
# LIVE_THRESHOLD = 0.5 * SCALE -> 500000 will be checked on-chain
LIVE_THRESHOLD = 500000

def verify_personal_sign(message_text, signature, expected_address):
    """
    Verifies personal_sign signature.
    message_text: plain text (string) that was signed, e.g., "Aura nonce:123456"
    signature: hex string from client personal_sign
    expected_address: 0x... address string that should be signer
    """
    message = encode_defunct(text=message_text)
    try:
        recovered = Account.recover_message(message, signature=signature)
        return recovered.lower() == expected_address.lower()
    except Exception as e:
        print("Signature verify error:", e)
        return False

@app.route("/prove", methods=["POST"])
def prove():
    """
    Expected JSON:
    {
      "image": "<base64 data uri or raw base64>",
      "wallet": "0xabc...",
      "signature": "<personal_sign signature hex>",
      "nonce": 1234567890  # unix timestamp
    }
    """
    req = request.get_json()
    if not req:
        return jsonify({"error": "expected json body"}), 400
    image_b64 = req.get("image")
    wallet = req.get("wallet")
    signature = req.get("signature")
    nonce = req.get("nonce")

    if not image_b64 or not wallet or not signature or not nonce:
        return jsonify({"error": "missing required fields (image, wallet, signature, nonce)"}), 400

    # Verify signature server-side: message convention must match client
    message_text = f"Aura nonce:{nonce}"
    if not verify_personal_sign(message_text, signature, wallet):
        return jsonify({"error": "invalid signature"}), 400

    # Preprocess image to NHWC (1,224,224,3)
    tensor = preprocess_base64_image_to_nhwc(image_b64)  # numpy array
    # The ezkl input.json structure must match what we used in setup (NHWC).
    # Write input_runtime.json (exact same format as model/input.json)
    input_payload = {"input": tensor.tolist()}
    with open("input_runtime.json", "w") as f:
        json.dump(input_payload, f)

    # Call ezkl prove:
    # Example CLI (ezkl must be installed and in PATH)
    # You may have to adapt flags to your ezkl version
    cmd = [
        EZKL_BIN, "prove",
        "--model", MODEL_ONNX,
        "--input", "input_runtime.json",
        "--pk", PK_KEY,
        "--settings", SETTINGS_JSON,
        "--out", "proof.json"
    ]
    print("Running ezkl prove:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ezkl prove failed stderr:", proc.stderr)
        return jsonify({"error": "ezkl prove failed", "stderr": proc.stderr}), 500

    # produce calldata (ezkl 'calldata' command) if supported by your ezkl
    cmd2 = [EZKL_BIN, "calldata", "--proof", "proof.json", "--settings", SETTINGS_JSON, "--out", "calldata.txt"]
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    calldata = None
    if proc2.returncode == 0 and os.path.exists("calldata.txt"):
        with open("calldata.txt", "r") as f:
            calldata = f.read().strip()

    # Read proof.json and (if present) public outputs
    proof_json = None
    try:
        with open("proof.json", "r") as f:
            proof_json = json.load(f)
    except Exception as e:
        print("Could not read proof.json:", e)

    # Optionally compute local model probability (if you want to return p_scaled)
    # (We do NOT require local inference; trust the circuit's public output.)
    # But for user convenience return the expected publicInputs array we will use:
    # publicInputs convention: [p_scaled, nonce, subject_uint]
    # subject_uint is integer form of address (lower 160 bits)
    subject_uint = int(wallet, 16)
    # We cannot compute p_scaled here without running ONNX inference on server.
    # So the server will attempt to read public outputs from ezkl if they're produced (implementation depends on ezkl).
    public_inputs = None
    # some ezkl versions output public inputs in proof.json["public_inputs"] — check and adapt
    if proof_json and "public_inputs" in proof_json:
        public_inputs = proof_json["public_inputs"]
    else:
        # fallback: return minimal contract public inputs skeleton (client must fill p_scaled)
        public_inputs = {"notice": "public inputs not extracted from proof.json automatically; use ezkl output or calculate locally."}

    return jsonify({
        "ok": True,
        "calldata": calldata,
        "proof_json": proof_json,
        "public_inputs": public_inputs,
        "subject_uint": str(subject_uint)
    })


if __name__ == "__main__":
    # Run server on 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
