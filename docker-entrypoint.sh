#!/bin/sh
set -e
python -m prisma db push --skip-generate --schema /app/prisma/schema.prisma
exec python run_web.py
