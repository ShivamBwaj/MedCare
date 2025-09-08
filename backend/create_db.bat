@echo off
echo Creating database...
python -c "import sqlite3, hashlib; conn = sqlite3.connect('medcare.db'); cursor = conn.cursor(); cursor.execute('CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT \"patient\", is_active BOOLEAN DEFAULT 1, created_at TEXT DEFAULT CURRENT_TIMESTAMP)'); cursor.execute('CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL, mime TEXT NOT NULL, sha256 TEXT NOT NULL, tx_hash TEXT, status TEXT DEFAULT \"pending\", owner_id TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (owner_id) REFERENCES users (id))'); cursor.execute('CREATE TABLE IF NOT EXISTS adherence_logs (id TEXT PRIMARY KEY, patient_id TEXT, medication TEXT NOT NULL, due_time TEXT NOT NULL, taken BOOLEAN DEFAULT 0, logged_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (patient_id) REFERENCES users (id))'); users = [('admin-001', 'admin@medcare.com', hashlib.sha256('admin123'.encode()).hexdigest(), 'admin'), ('manager-001', 'manager@medcare.com', hashlib.sha256('manager123'.encode()).hexdigest(), 'manager'), ('doctor-001', 'doctor@medcare.com', hashlib.sha256('doctor123'.encode()).hexdigest(), 'doctor'), ('patient-001', 'patient@medcare.com', hashlib.sha256('patient123'.encode()).hexdigest(), 'patient')]; [cursor.execute('INSERT OR REPLACE INTO users (id, email, password_hash, role) VALUES (?, ?, ?, ?)', user) for user in users]; conn.commit(); conn.close(); print('✅ Database created with demo users!')"
echo.
echo Demo accounts:
echo Admin: admin@medcare.com / admin123
echo Manager: manager@medcare.com / manager123
echo Doctor: doctor@medcare.com / doctor123
echo Patient: patient@medcare.com / patient123
pause
