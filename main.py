"""
EyeSpy FastAPI Server — v3

Endpoints:
  POST /api/auth/register     — Create account
  POST /api/auth/login        — Login, returns JWT
  GET  /api/auth/me           — Current user info (includes motivation)
  PUT  /api/auth/profile      — Update motivation & mode preferences
  WS   /ws/detect             — Real-time video processing WebSocket
  POST /api/sessions          — Save a detection session
  GET  /api/sessions          — List user sessions
  GET  /api/sessions/{id}     — Session detail with EAR/MAR history
  GET  /api/analytics         — Aggregate user analytics
"""

import asyncio
import base64
import json
import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional

from cv_engine import DrowsinessEngine
from database import (
    create_user, get_user_by_username, get_user_by_email,
    save_session, get_user_sessions, get_session_detail, get_user_analytics,
    update_user_profile,
)
from auth import hash_password, verify_password, create_token, decode_token


# ── App Setup ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("EyeSpy v3 server starting...")
    yield
    print("EyeSpy v3 server shutting down.")


app = FastAPI(
    title="EyeSpy API",
    version="3.0.0",
    description="Multi-signal drowsiness detection — personal guardian system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ── Auth Dependency ──────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    if not credentials:
        raise HTTPException(401, "Not authenticated")
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    user = get_user_by_username(payload["username"])
    if not user:
        raise HTTPException(401, "User not found")
    return user


# ── Pydantic Models ──────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ProfileUpdateRequest(BaseModel):
    preferred_mode: Optional[str] = None
    motivation_reason: Optional[str] = None
    motivation_who: Optional[str] = None
    motivation_stakes: Optional[str] = None

class SessionSaveRequest(BaseModel):
    session_data: dict


# ── Auth Endpoints ───────────────────────────────────────────────────

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    if get_user_by_username(req.username):
        raise HTTPException(400, "Username already exists")
    if get_user_by_email(req.email):
        raise HTTPException(400, "Email already registered")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    pw_hash = hash_password(req.password)
    user_id = create_user(req.username, req.email, pw_hash)
    if not user_id:
        raise HTTPException(500, "Failed to create user")

    token = create_token(user_id, req.username)
    return {"token": token, "username": req.username, "user_id": user_id}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")

    token = create_token(user["id"], user["username"])
    return {
        "token": token,
        "username": user["username"],
        "user_id": user["id"],
        "profile": {
            "preferred_mode": user.get("preferred_mode", "study"),
            "motivation_reason": user.get("motivation_reason", ""),
            "motivation_who": user.get("motivation_who", ""),
            "motivation_stakes": user.get("motivation_stakes", ""),
        },
    }


@app.get("/api/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "profile": {
            "preferred_mode": user.get("preferred_mode", "study"),
            "motivation_reason": user.get("motivation_reason", ""),
            "motivation_who": user.get("motivation_who", ""),
            "motivation_stakes": user.get("motivation_stakes", ""),
        },
    }


@app.put("/api/auth/profile")
async def update_profile(req: ProfileUpdateRequest, user: dict = Depends(get_current_user)):
    data = req.dict(exclude_none=True)
    if not data:
        raise HTTPException(400, "No fields to update")
    update_user_profile(user["id"], data)
    return {"message": "Profile updated", "profile": data}


# ── Session Endpoints ────────────────────────────────────────────────

@app.post("/api/sessions")
async def create_session(req: SessionSaveRequest, user: dict = Depends(get_current_user)):
    session_id = save_session(user["id"], req.session_data)
    return {"session_id": session_id, "message": "Session saved"}


@app.get("/api/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    sessions = get_user_sessions(user["id"])
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def session_detail(session_id: int, user: dict = Depends(get_current_user)):
    detail = get_session_detail(session_id)
    if not detail or detail["user_id"] != user["id"]:
        raise HTTPException(404, "Session not found")
    return detail


@app.get("/api/analytics")
async def analytics(user: dict = Depends(get_current_user)):
    data = get_user_analytics(user["id"])
    return data


# ── WebSocket: Real-Time Detection ───────────────────────────────────

# Per-connection CV engine instances
engines: dict[str, DrowsinessEngine] = {}


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time drowsiness detection.

    Client sends: base64-encoded JPEG frames
    Server returns: JSON with all metrics + base64 annotated frame
    """
    await websocket.accept()
    conn_id = str(id(websocket))
    print(f"New detection connection accepted: {conn_id}")
    engine = DrowsinessEngine()
    engines[conn_id] = engine

    try:
        while True:
            data = await websocket.receive_text()

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                # Treat raw string as base64 frame
                msg = {"type": "frame", "data": data}

            msg_type = msg.get("type", "frame")

            if msg_type == "frame":
                frame_b64 = msg.get("data", "")
                if not frame_b64:
                    continue

                # Decode base64 → numpy array
                try:
                    img_bytes = base64.b64decode(frame_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                except Exception:
                    continue

                # Process through CV engine
                try:
                    annotated, result = engine.process_frame(frame)
                except Exception as e:
                    print(f"Engine processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # Encode annotated frame back to base64 JPEG
                try:
                    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    frame_b64_out = base64.b64encode(buffer).decode("utf-8")

                    response = {
                        "type": "detection",
                        "metrics": result.to_dict(),
                        "frame": frame_b64_out,
                    }
                    await websocket.send_text(json.dumps(response))
                except Exception as e:
                    print(f"Response encoding/sending error: {e}")
                    continue

            elif msg_type == "reset":
                engine.reset()
                await websocket.send_text(json.dumps({"type": "reset_ack"}))

            elif msg_type == "get_summary":
                summary = engine.get_session_summary()
                await websocket.send_text(json.dumps({
                    "type": "summary",
                    "data": summary,
                }))

    except WebSocketDisconnect:
        pass
    finally:
        engines.pop(conn_id, None)


# ── Health Check ─────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {
        "service": "EyeSpy API",
        "version": "3.0.0",
        "status": "running",
        "features": [
            "EAR (Eye Aspect Ratio)",
            "MAR (Mouth Aspect Ratio / Yawn Detection)",
            "PERCLOS (Percentage of Eye Closure)",
            "Blink Rate (blinks/min)",
            "Head Pose Estimation (pitch/yaw/roll)",
            "Composite Alertness Score",
            "Personal Motivation Engine",
            "Mode-Specific Detection (Study/Drive/Work/Night/Custom)",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
