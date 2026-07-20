# Deployment

## Docker Compose

```bash
cd docker
docker compose up --build
```

Services:

| Service | Port |
|---------|------|
| web (nginx) | 5173 → 80 |
| api | 8000 |
| postgres | 5432 |
| redis | 6379 |

`ml-service` is reserved (commented / profile) for a future dedicated inference worker.

## Environment

Copy `.env.example` → `.env` and rotate `SECRET_KEY` before any public deploy.

## Production checklist

- [ ] Alembic migrations instead of `create_all`
- [ ] Managed Postgres
- [ ] HTTPS + hardened CORS
- [ ] Object storage for screenshots
- [ ] Celery workers for async scans
- [ ] Model artifact versioning
