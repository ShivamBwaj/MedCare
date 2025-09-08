import sqlite3
import hashlib

print("Creating database...")

# Create database connection
conn = sqlite3.connect('medcare.db')
cursor = conn.cursor()

# Create users table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'patient',
        is_active BOOLEAN DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
''')

# Create documents table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        filename TEXT NOT NULL,
        mime TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        tx_hash TEXT,
        status TEXT DEFAULT 'pending',
        owner_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
''')

# Create adherence_logs table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS adherence_logs (
        id TEXT PRIMARY KEY,
        patient_id TEXT,
        medication TEXT NOT NULL,
        due_time TEXT NOT NULL,
        taken BOOLEAN DEFAULT 0,
        logged_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
''')

# Hash password function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Create demo users
users = [
    ('admin-001', 'admin@medcare.com', hash_password('admin123'), 'admin'),
    ('manager-001', 'manager@medcare.com', hash_password('manager123'), 'manager'),
    ('doctor-001', 'doctor@medcare.com', hash_password('doctor123'), 'doctor'),
    ('patient-001', 'patient@medcare.com', hash_password('patient123'), 'patient'),
]

for user_id, email, password_hash, role in users:
    cursor.execute('''
        INSERT OR REPLACE INTO users (id, email, password_hash, role)
        VALUES (?, ?, ?, ?)
    ''', (user_id, email, password_hash, role))
    print(f"✅ Created {role}: {email}")

# Commit and close
conn.commit()
conn.close()

print("\n🎉 Database setup complete!")
print("\nDemo accounts:")
print("Admin: admin@medcare.com / admin123")
print("Manager: manager@medcare.com / manager123")
print("Doctor: doctor@medcare.com / doctor123")
print("Patient: patient@medcare.com / patient123")
