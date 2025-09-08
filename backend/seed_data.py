from sqlalchemy.orm import Session
from database import SessionLocal, User
from auth import get_password_hash

def create_demo_users():
    db = SessionLocal()
    
    # Check if users already exist
    existing_users = db.query(User).count()
    if existing_users > 0:
        print("Demo users already exist. Skipping seed data creation.")
        db.close()
        return
    
    # Create demo users
    demo_users = [
        {
            "username": "manager1",
            "email": "manager@medcare.com",
            "password": "manager123",
            "role": "manager",
            "full_name": "Dr. Sarah Johnson",
            "employee_id": "MGR001"
        },
        {
            "username": "doctor1",
            "email": "doctor1@medcare.com", 
            "password": "doctor123",
            "role": "doctor",
            "full_name": "Dr. Michael Chen",
            "employee_id": "DOC001"
        },
        {
            "username": "doctor2",
            "email": "doctor2@medcare.com",
            "password": "doctor123", 
            "role": "doctor",
            "full_name": "Dr. Emily Rodriguez",
            "employee_id": "DOC002"
        },
        {
            "username": "patient1",
            "email": "patient1@medcare.com",
            "password": "patient123",
            "role": "patient",
            "full_name": "John Smith",
            "patient_id": "PAT001"
        },
        {
            "username": "patient2", 
            "email": "patient2@medcare.com",
            "password": "patient123",
            "role": "patient", 
            "full_name": "Maria Garcia",
            "patient_id": "PAT002"
        },
        {
            "username": "patient3",
            "email": "patient3@medcare.com",
            "password": "patient123",
            "role": "patient",
            "full_name": "Robert Wilson", 
            "patient_id": "PAT003"
        }
    ]
    
    for user_data in demo_users:
        hashed_password = get_password_hash(user_data["password"])
        
        db_user = User(
            username=user_data["username"],
            email=user_data["email"],
            hashed_password=hashed_password,
            role=user_data["role"],
            full_name=user_data["full_name"],
            employee_id=user_data.get("employee_id"),
            patient_id=user_data.get("patient_id")
        )
        
        db.add(db_user)
    
    db.commit()
    print("✅ Demo users created successfully!")
    print("\nDemo Login Credentials:")
    print("Manager: manager1 / manager123")
    print("Doctor 1: doctor1 / doctor123") 
    print("Doctor 2: doctor2 / doctor123")
    print("Patient 1: patient1 / patient123")
    print("Patient 2: patient2 / patient123")
    print("Patient 3: patient3 / patient123")
    
    db.close()

if __name__ == "__main__":
    create_demo_users()
