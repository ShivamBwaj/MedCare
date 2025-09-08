from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List

router = APIRouter(prefix="/ws", tags=["chat"])
rooms: Dict[str, List[WebSocket]] = {}

@router.websocket("/chat/{room_id}")
async def chat(ws: WebSocket, room_id: str):
    await ws.accept()
    rooms.setdefault(room_id, []).append(ws)
    try:
        while True:
            msg = await ws.receive_text()
            for c in rooms[room_id]:
                if c != ws:
                    await c.send_text(msg)
    except WebSocketDisconnect:
        rooms[room_id].remove(ws)
