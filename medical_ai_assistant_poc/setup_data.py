import json
import sqlite3
import os

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_FILE = os.path.join(DATA_DIR, "patient_reports.json")
DB_FILE = os.path.join(DATA_DIR, "patient_data.db")

def initialize_database():
    """Initializes the SQLite database and creates the patient_reports table."""
    print(f"Initializing database at: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Drop table if it exists to ensure a clean start
    cursor.execute("DROP TABLE IF EXISTS patient_reports")

    # Create the table
    # We store the full report as a JSON string for easy retrieval by the tool
    cursor.execute("""
        CREATE TABLE patient_reports (
            patient_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            report_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database table 'patient_reports' created successfully.")

def populate_database():
    """Reads patient data from JSON and populates the SQLite database."""
    if not os.path.exists(JSON_FILE):
        print(f"Error: JSON file not found at {JSON_FILE}")
        return

    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    insert_count = 0
    for report in data:
        try:
            patient_id = report['patient_id']
            name = report['name']
            # Store the entire report as a JSON string in the database
            report_json = json.dumps(report)
            
            cursor.execute("""
                INSERT INTO patient_reports (patient_id, name, report_json)
                VALUES (?, ?, ?)
            """, (patient_id, name, report_json))
            insert_count += 1
        except KeyError as e:
            print(f"Skipping record due to missing key: {e} in report: {report}")
        except Exception as e:
            print(f"An error occurred while inserting record: {e}")

    conn.commit()
    conn.close()
    print(f"Successfully populated database with {insert_count} patient records.")

def main():
    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Initialize the database structure
    initialize_database()
    
    # 2. Populate the database with patient data
    populate_database()
    
    print("\nData setup complete.")
    print(f"Database file created at: {DB_FILE}")

if __name__ == "__main__":
    main()
