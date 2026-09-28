"""Web login, sessions, and role permissions."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

COOKIE_NAME = "car_scan_session"
SESSION_DAYS = 7
PBKDF2_ROUNDS = 180_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
ROLES = ("admin", "operator", "viewer")
PERMISSIONS = {
    "admin": frozenset({"scan.image", "scan.video", "scan.camera", "results.save", "users.manage"}),
    "operator": frozenset({"scan.image", "scan.video", "scan.camera", "results.save"}),
    "viewer": frozenset(),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL
);
"""


def _auth_secret() -> str:
    secret = os.getenv("CAR_SCAN_AUTH_SECRET", "").strip()
    if secret:
        return secret
    from .config import Settings

    path = Settings.from_env().output_dir / ".auth_secret"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        stored = path.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    generated = secrets.token_hex(32)
    path.write_text(generated, encoding="utf-8")
    return generated


def _auth_db_path() -> Path:
    raw = os.getenv("CAR_SCAN_AUTH_DB", "").strip()
    if raw:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    from .config import Settings

    path = Settings.from_env().output_dir / "auth.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return f"pbkdf2$sha256${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, algo, rounds_text, salt_hex, digest_hex = stored.split("$", 4)
    except ValueError:
        return False
    if scheme != "pbkdf2" or algo != "sha256":
        return False
    try:
        rounds = int(rounds_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(actual, expected)


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    role = str(row.get("role") or "viewer")
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "display_name": str(row.get("display_name") or row["username"]),
        "role": role,
        "active": bool(row.get("active", 1)),
        "permissions": sorted(PERMISSIONS.get(role, ())),
    }


@dataclass
class AuthStore:
    path: Path
    secret: str

    @classmethod
    def from_env(cls) -> "AuthStore | PrismaAuthStore":
        if os.getenv("CAR_SCAN_AUTH_DB", "").strip():
            store: AuthStore | PrismaAuthStore = cls(path=_auth_db_path(), secret=_auth_secret())
            store.initialize()
            return store
        from .config import Settings

        database_url = Settings.from_env().database_url
        if database_url:
            prisma_store = PrismaAuthStore(database_url=database_url, secret=_auth_secret())
            prisma_store.initialize()
            return prisma_store
        fallback = cls(path=_auth_db_path(), secret=_auth_secret())
        fallback.initialize()
        return fallback

    @contextmanager
    def _connect(self) -> Any:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        self.ensure_admin()

    def ensure_admin(self) -> None:
        username = os.getenv("CAR_SCAN_ADMIN_USERNAME", "admin").strip() or "admin"
        password = os.getenv("CAR_SCAN_ADMIN_PASSWORD", "changeme").strip() or "changeme"
        existing = self.get_by_username(username)
        if existing is None:
            self.create_user(username, "ผู้ดูแลระบบ", "admin", password)
            return
        if not existing["active"]:
            self.update_user(int(existing["id"]), active=True, role="admin")

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, username, display_name, role, active, created_at FROM users ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, display_name, role, password_hash, active FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, display_name, role, password_hash, active FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        user = self.get_by_username(username)
        if user is None or not user["active"] or not verify_password(password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
        return user

    def create_user(self, username: str, display_name: str, role: str, password: str) -> dict[str, Any]:
        username = username.strip()
        display_name = display_name.strip() or username
        role = role.strip().lower()
        if not USERNAME_RE.match(username):
            raise HTTPException(status_code=400, detail="ชื่อผู้ใช้ต้องเป็น a-z, 0-9, จุด, _ หรือ - ความยาว 3–32 ตัว")
        if role not in ROLES:
            raise HTTPException(status_code=400, detail="บทบาทต้องเป็น admin, operator หรือ viewer")
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (username, display_name, role, password_hash, active, created_at)
                    VALUES (?, ?, ?, ?, 1, ?)
                    """,
                    (username, display_name, role, hash_password(password), int(time.time())),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            raise HTTPException(status_code=409, detail="ชื่อผู้ใช้นี้มีอยู่แล้ว") from error
        created = self.get_by_id(user_id)
        assert created is not None
        return created

    def update_user(
        self,
        user_id: int,
        *,
        display_name: str | None = None,
        role: str | None = None,
        password: str | None = None,
        active: bool | None = None,
    ) -> dict[str, Any]:
        user = self.get_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้")
        next_role = (role or user["role"]).strip().lower()
        if next_role not in ROLES:
            raise HTTPException(status_code=400, detail="บทบาทต้องเป็น admin, operator หรือ viewer")
        next_active = user["active"] if active is None else (1 if active else 0)
        if user["role"] == "admin" and (next_role != "admin" or not next_active):
            if self._active_admin_count() <= 1:
                raise HTTPException(status_code=400, detail="ต้องเหลือผู้ดูแลระบบที่ใช้งานได้อย่างน้อย 1 คน")
        if password is not None and len(password) < 6:
            raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
        fields = ["display_name = ?", "role = ?", "active = ?"]
        values: list[Any] = [display_name.strip() if display_name else user["display_name"], next_role, next_active]
        if password:
            fields.append("password_hash = ?")
            values.append(hash_password(password))
        values.append(user_id)
        with self._connect() as connection:
            connection.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", values)
        updated = self.get_by_id(user_id)
        assert updated is not None
        return updated

    def delete_user(self, user_id: int) -> None:
        user = self.get_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้")
        if user["role"] == "admin" and user["active"] and self._active_admin_count() <= 1:
            raise HTTPException(status_code=400, detail="ลบผู้ดูแลระบบคนสุดท้ายไม่ได้")
        with self._connect() as connection:
            connection.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def _active_admin_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND active = 1"
            ).fetchone()
        return int(row["n"] if row else 0)

    def issue_session(self, user: dict[str, Any]) -> str:
        payload = {
            "uid": int(user["id"]),
            "exp": int(time.time()) + SESSION_DAYS * 86400,
        }
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        encoded = _b64(body.encode("utf-8"))
        signature = hmac.new(self.secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def read_session(self, token: str | None) -> dict[str, Any] | None:
        if not token or "." not in token:
            return None
        encoded, signature = token.rsplit(".", 1)
        expected = hmac.new(self.secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            payload = json.loads(_unb64(encoded))
        except (ValueError, json.JSONDecodeError):
            return None
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        user = self.get_by_id(int(payload["uid"]))
        if user is None or not user["active"]:
            return None
        return user


def _b64(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> str:
    import base64

    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding).decode("utf-8")


def _user_from_prisma(user: Any) -> dict[str, Any]:
    created = getattr(user, "createdAt", None)
    created_at = int(created.timestamp()) if created is not None else 0
    return {
        "id": int(user.id),
        "username": str(user.username),
        "display_name": str(user.displayName),
        "role": str(user.role),
        "password_hash": str(user.passwordHash),
        "active": 1 if user.active else 0,
        "created_at": created_at,
    }


@dataclass
class PrismaAuthStore:
    """User accounts stored in PostgreSQL through Prisma Client Python."""

    database_url: str
    secret: str

    def _client(self) -> Any:
        from .prisma_db import get_client

        return get_client(self.database_url)

    def initialize(self) -> None:
        from .prisma_db import ensure_schema

        ensure_schema(self.database_url)
        self._import_sqlite_users_if_empty()
        self.ensure_admin()

    def _import_sqlite_users_if_empty(self) -> None:
        if self._client().user.count() > 0:
            return
        sqlite_path = _auth_db_path()
        if not sqlite_path.is_file():
            return
        connection = sqlite3.connect(sqlite_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT username, display_name, role, password_hash, active, created_at FROM users"
            ).fetchall()
        except sqlite3.Error:
            return
        finally:
            connection.close()
        for row in rows:
            self._client().user.create(
                data={
                    "username": row["username"],
                    "displayName": row["display_name"],
                    "role": row["role"],
                    "passwordHash": row["password_hash"],
                    "active": bool(row["active"]),
                }
            )

    def ensure_admin(self) -> None:
        username = os.getenv("CAR_SCAN_ADMIN_USERNAME", "admin").strip() or "admin"
        password = os.getenv("CAR_SCAN_ADMIN_PASSWORD", "changeme").strip() or "changeme"
        existing = self.get_by_username(username)
        if existing is None:
            self.create_user(username, "ผู้ดูแลระบบ", "admin", password)
            return
        if not existing["active"]:
            self.update_user(int(existing["id"]), active=True, role="admin")

    def list_users(self) -> list[dict[str, Any]]:
        rows = self._client().user.find_many(order={"id": "asc"})
        return [_user_from_prisma(row) for row in rows]

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        row = self._client().user.find_unique(where={"id": int(user_id)})
        return _user_from_prisma(row) if row else None

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        row = self._client().user.find_unique(where={"username": username.strip()})
        return _user_from_prisma(row) if row else None

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        user = self.get_by_username(username)
        if user is None or not user["active"] or not verify_password(password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
        return user

    def create_user(self, username: str, display_name: str, role: str, password: str) -> dict[str, Any]:
        username = username.strip()
        display_name = display_name.strip() or username
        role = role.strip().lower()
        if not USERNAME_RE.match(username):
            raise HTTPException(status_code=400, detail="ชื่อผู้ใช้ต้องเป็น a-z, 0-9, จุด, _ หรือ - ความยาว 3–32 ตัว")
        if role not in ROLES:
            raise HTTPException(status_code=400, detail="บทบาทต้องเป็น admin, operator หรือ viewer")
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
        try:
            created = self._client().user.create(
                data={
                    "username": username,
                    "displayName": display_name,
                    "role": role,
                    "passwordHash": hash_password(password),
                    "active": True,
                }
            )
        except Exception as error:
            if error.__class__.__name__ == "UniqueViolationError":
                raise HTTPException(status_code=409, detail="ชื่อผู้ใช้นี้มีอยู่แล้ว") from error
            raise
        return _user_from_prisma(created)

    def update_user(
        self,
        user_id: int,
        *,
        display_name: str | None = None,
        role: str | None = None,
        password: str | None = None,
        active: bool | None = None,
    ) -> dict[str, Any]:
        user = self.get_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้")
        next_role = (role or user["role"]).strip().lower()
        if next_role not in ROLES:
            raise HTTPException(status_code=400, detail="บทบาทต้องเป็น admin, operator หรือ viewer")
        next_active = bool(user["active"] if active is None else active)
        if user["role"] == "admin" and (next_role != "admin" or not next_active):
            if self._active_admin_count() <= 1:
                raise HTTPException(status_code=400, detail="ต้องเหลือผู้ดูแลระบบที่ใช้งานได้อย่างน้อย 1 คน")
        if password is not None and len(password) < 6:
            raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
        data: dict[str, Any] = {
            "displayName": display_name.strip() if display_name else user["display_name"],
            "role": next_role,
            "active": next_active,
        }
        if password:
            data["passwordHash"] = hash_password(password)
        updated = self._client().user.update(where={"id": user_id}, data=data)
        assert updated is not None
        return _user_from_prisma(updated)

    def delete_user(self, user_id: int) -> None:
        user = self.get_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้")
        if user["role"] == "admin" and user["active"] and self._active_admin_count() <= 1:
            raise HTTPException(status_code=400, detail="ลบผู้ดูแลระบบคนสุดท้ายไม่ได้")
        self._client().user.delete(where={"id": user_id})

    def _active_admin_count(self) -> int:
        return int(self._client().user.count(where={"role": "admin", "active": True}))

    def issue_session(self, user: dict[str, Any]) -> str:
        payload = {
            "uid": int(user["id"]),
            "exp": int(time.time()) + SESSION_DAYS * 86400,
        }
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        encoded = _b64(body.encode("utf-8"))
        signature = hmac.new(self.secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def read_session(self, token: str | None) -> dict[str, Any] | None:
        if not token or "." not in token:
            return None
        encoded, signature = token.rsplit(".", 1)
        expected = hmac.new(self.secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            payload = json.loads(_unb64(encoded))
        except (ValueError, json.JSONDecodeError):
            return None
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        user = self.get_by_id(int(payload["uid"]))
        if user is None or not user["active"]:
            return None
        return user


AuthBackend = AuthStore | PrismaAuthStore


def current_user(request: Request) -> dict[str, Any] | None:
    store: AuthBackend = request.app.state.auth
    return store.read_session(request.cookies.get(COOKIE_NAME))


def require_user(request: Request) -> dict[str, Any]:
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="กรุณาเข้าสู่ระบบ")
    return user


def has_permission(user: dict[str, Any], permission: str) -> bool:
    return permission in PERMISSIONS.get(str(user.get("role") or ""), ())


def require_permission(request: Request, permission: str) -> dict[str, Any]:
    user = require_user(request)
    if not has_permission(user, permission):
        raise HTTPException(status_code=403, detail="บัญชีนี้ไม่มีสิทธิ์ทำรายการนี้")
    return user


def scan_permission(media_type: str) -> str:
    if media_type == "video":
        return "scan.video"
    if media_type == "camera":
        return "scan.camera"
    return "scan.image"


def set_session_cookie(response: JSONResponse, token: str) -> JSONResponse:
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_DAYS * 86400,
        path="/",
    )
    return response


def clear_session_cookie(response: JSONResponse | RedirectResponse) -> JSONResponse | RedirectResponse:
    response.delete_cookie(COOKIE_NAME, path="/")
    return response
