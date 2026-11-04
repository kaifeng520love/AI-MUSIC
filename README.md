# MusicGen 后端第一版

解压后，将 backend 文件夹和 start_backend.bat 放进 C:\AI\MusicGen。
正确位置：C:\AI\MusicGen\backend\main.py。
保留原来的 generate.py、.venv、model_cache 和 outputs。

双击 start_backend.bat，或在 PowerShell 运行：

```powershell
Set-Location C:\AI\MusicGen
& ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

不要使用 --reload 或多个 workers，避免重复加载模型或产生多个数据库任务消费者。
浏览器打开 http://127.0.0.1:8000/docs 。当前交付为后端，正式前端下一步接入。

## 验证流程
1. GET /api/health，点击 Try it out → Execute，应该返回 status: ok。
2. POST /api/generations，填下面示例后 Execute：
```json
{"prompt":"Gentle relaxing piano music with soft ambient pads, instrumental","duration":5,"guidance_scale":3.0}
```
3. 返回 202 和 id 表示任务已接收，不代表已完成。复制 id。
4. GET /api/generations/{task_id} 填 id 并查询；状态从 queued 到 running 再到 completed。
首次任务需加载模型，可能等待较久；实际加载或生成失败时会记录 error，终端保留完整报错。
5. 完成后用返回的 audio_url 试听、download_url 下载（在本机地址后拼接该路径）。
6. GET /api/history 查询记录。Ctrl+C 停止，重新启动后再次查询，记录和音频应仍存在。

## 保存位置与行为
- 数据库：data/musicgen.db，启动时自动建库建表。
- 音频：outputs/{唯一任务编号}.wav；文件名不会互相覆盖。
- 时间使用 UTC ISO 格式，正式前端将转换为本地时间。
- 同时只执行一个任务，最多额外等待 10 个任务。
- 正常退出等待当前任务完成；待执行任务标记失败。
- 强制退出后，下次启动将遗留 queued/running 记录标记失败，可重新提交。
- 模型只从现有 model_cache 加载；缓存缺失会报错，请提供终端报错排查。
- 生成时长按每秒约 50 token 换算，actual_duration 记录真实音频时长。
- elapsed_seconds 在首个任务中包含加载模型耗时。
- 过去用独立 generate.py 生成的 WAV 未自动入库；如需保留在历史页，后续加入导入功能，不虚构当时参数。
- prompt 建议先用英文描述；duration 5–30 秒；guidance_scale 1–10。

## 本次验证范围
使用替代生成服务验证了接口、队列、成功与失败状态、音频响应、参数校验和重启持久化。
真实 GPU 推理需要在你的 Windows / RTX 4060 环境中完成验证。
