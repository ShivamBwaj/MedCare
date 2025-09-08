import os
import uuid
import hashlib
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.document import Document
from app.models.user import Role, User
from app.utils.security import require_roles, get_current_user
from app.services.chain import log_hash_on_chain

router = APIRouter(prefix="/files", tags=["files"])

STORAGE = os.path.abspath("backend/storage")
os.makedirs(STORAGE, exist_ok=True)

@router.post("", dependencies=[Depends(require_roles(Role.manager))])
async def upload_file(title: str = Form(...), f: UploadFile = File(...),
                      db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)):
    data = await f.read()
    digest = hashlib.sha256(data).hexdigest()
    name = f"{uuid.uuid4()}_{f.filename}"
    path = os.path.join(STORAGE, name)
    
    with open(path, "wb") as out:
        out.write(data)
    
    doc = Document(title=title, filename=name, mime=f.content_type, 
                   sha256=digest, owner_id=user.id)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"id": str(doc.id), "sha256": digest}

@router.post("/{doc_id}/approve", dependencies=[Depends(require_roles(Role.manager))])
def approve(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).get(doc_id)
    if not doc:
        raise HTTPException(404, "Not found")
    if doc.tx_hash:
        return {"tx_hash": doc.tx_hash, "status": doc.status}
    
    txh = log_hash_on_chain(doc.sha256)
    doc.tx_hash = txh
    doc.status = "approved"
    db.commit()
    return {"tx_hash": txh, "status": doc.status}

@router.get("", dependencies=[Depends(require_roles(Role.manager))])
def list_docs(db: Session = Depends(get_db)):
    return db.query(Document).order_by(Document.created_at.desc()).all()
