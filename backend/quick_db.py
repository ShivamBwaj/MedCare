import sqlite3
import hashlib

# Create database
conn = sqlite3.connect("medcare.db")
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

# Hash passwords
def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Insert users
users = [
    ("admin-001", "admin@medcare.com", hash_pw("admin123"), "admin"),
    ("manager-001", "manager@medcare.com", hash_pw("manager123"), "manager"),
    ("doctor-001", "doctor@medcare.com", hash_pw("doctor123"), "doctor"),
    ("patient-001", "patient@medcare.com", hash_pw("patient123"), "patient"),
]

for user_id, email, password_hash, role in users:
    cursor.execute('''
        INSERT OR REPLACE INTO users (id, email, password_hash, role)
        VALUES (?, ?, ?, ?)
    ''', (user_id, email, password_hash, role))
    print(f"Created {role}: {email}")

conn.commit()
conn.close()
print("Database created successfully!")
