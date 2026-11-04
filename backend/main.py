"""启动：在项目根目录执行 python -m uvicorn backend.main:app。"""
import logging
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .database import Database
from .music_service import MusicService

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger('uvicorn.error')


def now():
    return datetime.now(timezone.utc).isoformat()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    duration: int = Field(default=20, ge=5, le=30)
    guidance_scale: float = Field(default=3.0, ge=1.0, le=10.0)

    @field_validator('prompt')
    @classmethod
    def clean_prompt(cls, value):
        if not value.strip():
            raise ValueError('音乐描述不能为空')
        return value.strip()


def create_app(root=ROOT, service=None):
    root = Path(root)
    db = Database(root / 'data' / 'musicgen.db')
    music = service if service is not None else MusicService(root)
    pending = queue.Queue()
    stopping = threading.Event()
    submit_lock = threading.Lock()

    def worker():
        while True:
            task = pending.get()
            try:
                if task is None:
                    return
                if stopping.is_set():
                    db.update(task['id'], status='failed', error='服务停止，任务未执行。', finished_at=now())
                    continue
                db.update(task['id'], status='running')
                try:
                    result = music.generate(task)
                    db.update(task['id'], status='completed', finished_at=now(), **result)
                except Exception as exc:
                    logger.exception('MusicGen task failed: %s', task['id'])
                    db.update(task['id'], status='failed', error=str(exc), finished_at=now())
            finally:
                pending.task_done()

    @asynccontextmanager
    async def lifespan(app):
        (root / 'outputs').mkdir(parents=True, exist_ok=True)
        db.init()
        thread = threading.Thread(target=worker, name='musicgen-worker', daemon=True)
        thread.start()
        yield
        stopping.set()
        pending.put(None)
        # 正常退出等待当前生成完成；强制退出的任务在下次启动时标记失败。
        import asyncio
        await asyncio.to_thread(thread.join)

    app = FastAPI(title='MusicGen 本地音乐生成', version='1.0.0', lifespan=lifespan)

    def present(item):
        item = dict(item)
        done = item['status'] == 'completed'
        item['audio_url'] = f"/api/generations/{item['id']}/audio" if done else None
        item['download_url'] = f"/api/generations/{item['id']}/download" if done else None
        return item

    @app.get('/')
    def home():
        return {'message': 'MusicGen 后端已启动', 'docs': '/docs'}

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'database': 'sqlite', 'model_load': '首次生成时加载'}

    @app.post('/api/generations', status_code=202)
    def generate(body: GenerateRequest):
        with submit_lock:
            if stopping.is_set() or pending.qsize() >= 10:
                raise HTTPException(503, '任务队列已满或服务正在停止，请稍后重试。')
            item = dict(id=uuid.uuid4().hex, **body.model_dump(), status='queued', created_at=now())
            db.insert(item)
            pending.put(item)
        return present(db.get(item['id']))

    @app.get('/api/history')
    def history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
        total, items = db.history(limit, offset)
        return {'total': total, 'items': [present(item) for item in items]}

    @app.get('/api/generations/{task_id}')
    def detail(task_id: str):
        item = db.get(task_id)
        if item is None:
            raise HTTPException(404, '生成记录不存在')
        return present(item)

    def audio_response(task_id, download):
        item = db.get(task_id)
        if item is None:
            raise HTTPException(404, '生成记录不存在')
        if item['status'] != 'completed' or not item['filename']:
            raise HTTPException(409, '音频尚未生成成功')
        path = root / 'outputs' / item['filename']
        if not path.is_file():
            raise HTTPException(404, '音频文件已丢失')
        return FileResponse(path, media_type='audio/wav',
                            filename=item['filename'] if download else None)

    @app.get('/api/generations/{task_id}/audio')
    def audio(task_id: str):
        return audio_response(task_id, False)

    @app.get('/api/generations/{task_id}/download')
    def download(task_id: str):
        return audio_response(task_id, True)

    return app


app = create_app()
