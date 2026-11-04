"""每次操作独立连接；音频文件留在 outputs，数据库保存元数据。"""
import sqlite3
from contextlib import contextmanager


class Database:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY, prompt TEXT NOT NULL,
                duration INTEGER NOT NULL, guidance_scale REAL NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL,
                finished_at TEXT, actual_duration REAL, elapsed_seconds REAL,
                filename TEXT, error TEXT
            )''')
            conn.execute("UPDATE generations SET status='failed', error=? WHERE status IN ('queued','running')",
                         ('上次服务停止，任务未完成；请重新提交。',))

    def insert(self, item):
        with self.connect() as conn:
            conn.execute('''INSERT INTO generations
                (id,prompt,duration,guidance_scale,status,created_at)
                VALUES (:id,:prompt,:duration,:guidance_scale,:status,:created_at)''', item)

    def update(self, task_id, **fields):
        allowed = {'status','finished_at','actual_duration','elapsed_seconds','filename','error'}
        if not fields or not fields.keys() <= allowed:
            raise ValueError('Invalid fields')
        with self.connect() as conn:
            conn.execute('UPDATE generations SET ' + ','.join(f'{k}=?' for k in fields) + ' WHERE id=?',
                         [*fields.values(), task_id])

    def get(self, task_id):
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM generations WHERE id=?', (task_id,)).fetchone()
            return dict(row) if row else None

    def history(self, limit, offset):
        with self.connect() as conn:
            total = conn.execute('SELECT COUNT(*) FROM generations').fetchone()[0]
            rows = conn.execute('SELECT * FROM generations ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?',
                                (limit, offset)).fetchall()
            return total, [dict(row) for row in rows]
