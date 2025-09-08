#!/usr/bin/env python3
"""
Create initial admin user for MedCare system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.base import SessionLocal, engine, Base
from app.models.user import User, Role
from app.utils.security import hash_pw

# Create tables
Base.metadata.create_all(bind=engine)

# Create admin user
db = SessionLocal()
try:
    # Check if admin already exists
    admin = db.query(User).filter(User.email == "admin@medcare.com").first()
    if admin:
        print("Admin user already exists!")
    else:
        admin = User(
            email="admin@medcare.com",
            password_hash=hash_pw("admin123"),
            role=Role.admin
        )
        db.add(admin)
        db.commit()
        print("✅ Admin user created!")
        print("Email: admin@medcare.com")
        print("Password: admin123")
        
    # Create sample users for testing
    users_to_create = [
        ("manager@medcare.com", "manager123", Role.manager),
        ("doctor@medcare.com", "doctor123", Role.doctor),
        ("patient@medcare.com", "patient123", Role.patient),
    ]
    
    for email, password, role in users_to_create:
        existing = db.query(User).filter(User.email == email).first()
        if not existing:
            user = User(
                email=email,
                password_hash=hash_pw(password),
                role=role
            )
            db.add(user)
            print(f"✅ Created {role.value}: {email} / {password}")
    
    db.commit()
    print("\n🎉 Database setup complete!")
    print("\nTest accounts:")
    print("Admin: admin@medcare.com / admin123")
    print("Manager: manager@medcare.com / manager123") 
    print("Doctor: doctor@medcare.com / doctor123")
    print("Patient: patient@medcare.com / patient123")
    
finally:
    db.close()
