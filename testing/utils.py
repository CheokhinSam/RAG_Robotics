import os
import re
from datetime import datetime
import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_file(file, file_path=None):
    if file_path is None:
        safe_name = re.sub(r'[<>:"/\\|?*]', '', file.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(WORKING_DIR, f"{timestamp}_{safe_name}")
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def get_available_collections():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM langchain_pg_collection;")
            collections = [row[0] for row in cur.fetchall()]
        conn.close()
        return collections if collections else ["No collections available"]
    except Exception as e:
        logger.error(f"Error fetching collections: {str(e)}")
        raise