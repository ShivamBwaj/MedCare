from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.adherence import AdherenceLog
from app.models.user import Role, User
from app.utils.security import require_roles, get_current_user
import datetime as dt

router = APIRouter(prefix="/adherence", tags=["adherence"])

class AdLogIn(BaseModel):
    medication: str
    due_time: str  # ISO format
    taken: bool

@router.post("", dependencies=[Depends(require_roles(Role.patient))])
def log_adherence(body: AdLogIn, db: Session = Depends(get_db), u: User = Depends(get_current_user)):
    due_dt = dt.datetime.fromisoformat(body.due_time.replace('Z', '+00:00'))
    log = AdherenceLog(patient_id=u.id, medication=body.medication, due_time=due_dt, taken=body.taken)
    db.add(log)
    db.commit()
    db.refresh(log)
    return {"id": str(log.id)}

@router.get("/{patient_id}", dependencies=[Depends(require_roles(Role.doctor, Role.manager))])
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    q = db.query(AdherenceLog).filter(AdherenceLog.patient_id == patient_id).order_by(AdherenceLog.due_time.desc()).all()
    # Simple adherence score (last 30 logs)
    last = q[:30]
    score = (sum(1 for r in last if r.taken) / len(last)) * 100 if last else None
    return {"logs": q, "adherence_score": score}
