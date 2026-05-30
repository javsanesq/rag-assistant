# Worker Notes

The API only enqueues ingestion and evaluation jobs. The worker process claims durable jobs from the SQL database, leases them, executes them, and writes progress/results back to the jobs table.

Local Docker starts the worker with:

```bash
python -m rag_assistant_api.worker
```

The queue intentionally uses the same SQLAlchemy database as the API instead of Redis/Celery so this project remains portable across local Docker, VPS, and simple platform-as-a-service deployments.
