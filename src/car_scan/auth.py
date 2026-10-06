"""Web login, sessions, and role permissions."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

LOGGER = logging.getLogger(__name__)

COOKIE_NAME = "car_scan_session"
SESSION_DAYS = 7
PBKDF2_ROUNDS = 180_000
AUTH_DB_RETRY_SECONDS = 5.0
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
ROLES = ("admin", "operator", "viewer", "superuser")
ROLE_CATALOG = (
    {"code": "admin", "name": "Administrator", "permissions": sorted({"scan.image", "scan.video", "scan.camera", "results.save", "users.manage"})},
    {"code": "operator", "name": "Operator", "permissions": sorted({"scan.image", "scan.video", "scan.camera", "results.save"})},
    {"code": "viewer", "name": "Viewer", "permissions": []},
    {"code": "superuser", "name": "Super User", "permissions": ["*"]},
)
STATUS_CATALOG = (
    {"code": "ACTIVE", "name": "Active"},
    {"code": "INACTIVE", "name": "Inactive"},
    {"code": "SUSPENDED", "name": "Suspended"},
)
PERMISSIONS = {
    "admin": frozenset({"scan.image", "scan.video", "scan.camera", "results.save", "users.manage"}),
    "operator": frozenset({"scan.image", "scan.video", "scan.camera", "results.save"}),
    "viewer": frozenset(),
    "superuser": frozenset({"scan.image", "scan.video", "scan.camera", "results.save", "users.manage"}),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    session_hash TEXT NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS user_roles (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    permissions TEXT NOT NULL DEFAULT '[]',
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS user_statuses (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
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


def _session_hash(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _encode_session(secret: str, user_id: int, session_id: str, expires_at: int) -> str:
    payload = {"uid": int(user_id), "sid": session_id, "exp": int(expires_at)}
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    encoded = _b64(body.encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _decode_session(secret: str, token: str | None) -> dict[str, Any] | None:
    if not token or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    expected = hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_unb64(encoded))
        user_id = int(payload["uid"])
        session_id = str(payload["sid"])
        expires_at = int(payload["exp"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not session_id or expires_at < int(time.time()):
        return None
    return {"uid": user_id, "sid": session_id, "exp": expires_at}


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    role = canonical_role(row.get("role"))
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "display_name": str(row.get("display_name") or row["username"]),
        "role": role,
        "active": bool(row.get("active", 1)),
        "status": str(row.get("status") or ("ACTIVE" if row.get("active", 1) else "INACTIVE")),
        "permissions": sorted(PERMISSIONS.get(role, ())),
    }


def canonical_role(value: Any) -> str:
    role = str(value or "viewer").strip().lower()
    return {
        "admin": "admin",
        "operator": "operator",
        "editor": "operator",
        "viewer": "viewer",
        "user": "viewer",
        "superuser": "superuser",
    }.get(role, role)


def database_role(value: Any) -> str:
    return {
        "admin": "ADMIN",
        "operator": "EDITOR",
        "viewer": "USER",
        "superuser": "SUPERUSER",
    }.get(canonical_role(value), str(value or "USER").strip().upper())


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
            try:
                prisma_store.initialize()
                return prisma_store
            except Exception as error:
                LOGGER.warning(
                    "PostgreSQL authentication unavailable; PostgreSQL remains the only auth source: %s",
                    error,
                )
                return ResilientAuthStore(primary=prisma_store)
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
            try:
                connection.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'")
            except sqlite3.OperationalError:
                pass
            now = int(time.time())
            connection.executemany(
                """
                INSERT INTO user_roles (code, name, permissions, active, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(code) DO UPDATE SET name = excluded.name, permissions = excluded.permissions,
                    active = 1, updated_at = excluded.updated_at
                """,
                [(str(item["code"]).upper(), item["name"], json.dumps(item["permissions"]), now, now) for item in ROLE_CATALOG],
            )
            connection.executemany(
                """
                INSERT INTO user_statuses (code, name, active, created_at, updated_at)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(code) DO UPDATE SET name = excluded.name, active = 1, updated_at = excluded.updated_at
                """,
                [(item["code"], item["name"], now, now) for item in STATUS_CATALOG],
            )
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
                "SELECT id, username, display_name, role, active, status, created_at FROM users ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_catalog(self) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as connection:
            role_rows = connection.execute("SELECT code, name, permissions, active FROM user_roles WHERE active = 1 ORDER BY code").fetchall()
            status_rows = connection.execute("SELECT code, name, active FROM user_statuses WHERE active = 1 ORDER BY code").fetchall()
        roles = []
        for row in role_rows:
            try:
                permissions = json.loads(row["permissions"] or "[]")
            except json.JSONDecodeError:
                permissions = []
            roles.append({"code": canonical_role(row["code"]), "name": row["name"], "permissions": permissions if isinstance(permissions, list) else []})
        return {
            "roles": roles or [dict(item) for item in ROLE_CATALOG],
            "statuses": [dict(row) for row in status_rows] or [dict(item) for item in STATUS_CATALOG],
        }

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, display_name, role, password_hash, active, status FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, display_name, role, password_hash, active, status FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        user = self.get_by_username(username)
        if user is None or not user["active"] or user.get("status") != "ACTIVE" or not verify_password(password, user["password_hash"]):
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
                    INSERT INTO users (username, display_name, role, password_hash, active, status, created_at)
                    VALUES (?, ?, ?, ?, 1, 'ACTIVE', ?)
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
        fields = ["display_name = ?", "role = ?", "active = ?", "status = ?"]
        values: list[Any] = [display_name.strip() if display_name else user["display_name"], next_role, next_active, "ACTIVE" if next_active else "INACTIVE"]
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
        user_id = int(user["id"])
        session_id = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + SESSION_DAYS * 86400
        with self._connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (int(time.time()),))
            connection.execute(
                """
                INSERT INTO auth_sessions (user_id, session_hash, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    session_hash = excluded.session_hash,
                    expires_at = excluded.expires_at
                """,
                (user_id, _session_hash(session_id), expires_at),
            )
        return _encode_session(self.secret, user_id, session_id, expires_at)

    def read_session(self, token: str | None) -> dict[str, Any] | None:
        payload = _decode_session(self.secret, token)
        if payload is None:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_hash, expires_at FROM auth_sessions WHERE user_id = ?",
                (payload["uid"],),
            ).fetchone()
        if (
            row is None
            or int(row["expires_at"]) < int(time.time())
            or not hmac.compare_digest(str(row["session_hash"]), _session_hash(payload["sid"]))
        ):
            return None
        user = self.get_by_id(payload["uid"])
        if user is None or not user["active"] or user.get("status") != "ACTIVE":
            return None
        return user

    def session_id(self, token: str | None) -> str | None:
        payload = _decode_session(self.secret, token)
        return str(payload["sid"]) if payload is not None else None

    def revoke_session(self, token: str | None) -> None:
        payload = _decode_session(self.secret, token)
        if payload is None:
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE user_id = ? AND session_hash = ?",
                (payload["uid"], _session_hash(payload["sid"])),
            )

    def revoke_user_sessions(self, user_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE user_id = ?", (int(user_id),))


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
        "role": canonical_role(user.role),
        "password_hash": str(user.passwordHash),
        "active": 1 if user.active else 0,
        "status": str(getattr(user, "status", "ACTIVE") or ("ACTIVE" if user.active else "INACTIVE")),
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
        self._ensure_session_storage()
        self._ensure_user_catalog()
        self._import_sqlite_users_if_empty()
        self.ensure_admin()

    def _ensure_session_storage(self) -> None:
        """Keep one revocable session per user without changing existing user data."""

        self._client().execute_raw(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                session_hash TEXT NOT NULL,
                expires_at BIGINT NOT NULL
            )
            """
        )

    def _ensure_user_catalog(self) -> None:
        client = self._client()
        client.execute_raw("ALTER TABLE users ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ACTIVE'")
        client.execute_raw(
            """
            UPDATE users SET role = CASE LOWER(role)
              WHEN 'admin' THEN 'ADMIN'
              WHEN 'operator' THEN 'EDITOR'
              WHEN 'viewer' THEN 'USER'
              WHEN 'superuser' THEN 'SUPERUSER'
              ELSE UPPER(role)
            END,
            status = CASE WHEN active THEN 'ACTIVE' ELSE 'INACTIVE' END
            """
        )
        client.execute_raw(
            """
            CREATE TABLE IF NOT EXISTS user_roles (
              code TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              permissions JSONB,
              active BOOLEAN NOT NULL DEFAULT TRUE,
              created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        client.execute_raw(
            """
            CREATE TABLE IF NOT EXISTS user_statuses (
              code TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              active BOOLEAN NOT NULL DEFAULT TRUE,
              created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        roles = {
            "ADMIN": ("Administrator", '["scan.image","scan.video","scan.camera","results.save","users.manage"]'),
            "EDITOR": ("Editor", '["scan.image","scan.video","scan.camera","results.save"]'),
            "USER": ("User", "[]"),
            "SUPERUSER": ("Super User", '["*"]'),
        }
        for code, (name, permissions) in roles.items():
            client.execute_raw(
                """
                INSERT INTO user_roles (code, name, permissions, active, updated_at)
                VALUES ($1, $2, $3::jsonb, TRUE, CURRENT_TIMESTAMP)
                ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name, permissions = EXCLUDED.permissions,
                  active = TRUE, updated_at = CURRENT_TIMESTAMP
                """,
                code,
                name,
                permissions,
            )
        for code, name in (("ACTIVE", "Active"), ("INACTIVE", "Inactive"), ("SUSPENDED", "Suspended")):
            client.execute_raw(
                """
                INSERT INTO user_statuses (code, name, active, updated_at)
                VALUES ($1, $2, TRUE, CURRENT_TIMESTAMP)
                ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name, active = TRUE,
                  updated_at = CURRENT_TIMESTAMP
                """,
                code,
                name,
            )

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
                "SELECT username, display_name, role, password_hash, active, status, created_at FROM users"
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
                    "status": row["status"] or ("ACTIVE" if row["active"] else "INACTIVE"),
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

    def list_catalog(self) -> dict[str, list[dict[str, Any]]]:
        role_rows = self._client().query_raw(
            "SELECT code, name, permissions, active FROM user_roles WHERE active = TRUE ORDER BY code"
        )
        status_rows = self._client().query_raw(
            "SELECT code, name, active FROM user_statuses WHERE active = TRUE ORDER BY code"
        )
        roles_by_code: dict[str, dict[str, Any]] = {}
        for row in role_rows:
            permissions = row.get("permissions") or []
            if isinstance(permissions, str):
                try:
                    permissions = json.loads(permissions)
                except json.JSONDecodeError:
                    permissions = []
            code = canonical_role(row.get("code"))
            if code in ROLES:
                roles_by_code[code] = {"code": code, "name": str(row.get("name") or ""), "permissions": list(permissions) if isinstance(permissions, list) else []}
        return {
            "roles": list(roles_by_code.values()) or [dict(item) for item in ROLE_CATALOG],
            "statuses": [
                {"code": str(row.get("code") or ""), "name": str(row.get("name") or ""), "active": bool(row.get("active", True))}
                for row in status_rows
            ] or [dict(item) for item in STATUS_CATALOG],
        }

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        row = self._client().user.find_unique(where={"id": int(user_id)})
        return _user_from_prisma(row) if row else None

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        row = self._client().user.find_unique(where={"username": username.strip()})
        return _user_from_prisma(row) if row else None

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        user = self.get_by_username(username)
        if user is None or not user["active"] or user.get("status") != "ACTIVE" or not verify_password(password, user["password_hash"]):
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
                    "role": database_role(role),
                    "passwordHash": hash_password(password),
                    "active": True,
                    "status": "ACTIVE",
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
        next_role = canonical_role(role or user["role"])
        if next_role not in ROLES:
            raise HTTPException(status_code=400, detail="บทบาทต้องเป็น admin, operator หรือ viewer")
        next_active = bool(user["active"] if active is None else active)
        if canonical_role(user["role"]) in {"admin", "superuser"} and (next_role not in {"admin", "superuser"} or not next_active):
            if self._active_admin_count() <= 1:
                raise HTTPException(status_code=400, detail="ต้องเหลือผู้ดูแลระบบที่ใช้งานได้อย่างน้อย 1 คน")
        if password is not None and len(password) < 6:
            raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
        data: dict[str, Any] = {
            "displayName": display_name.strip() if display_name else user["display_name"],
            "role": database_role(next_role),
            "active": next_active,
            "status": "ACTIVE" if next_active else "INACTIVE",
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
        if canonical_role(user["role"]) in {"admin", "superuser"} and user["active"] and self._active_admin_count() <= 1:
            raise HTTPException(status_code=400, detail="ลบผู้ดูแลระบบคนสุดท้ายไม่ได้")
        self._client().user.delete(where={"id": user_id})

    def _active_admin_count(self) -> int:
        rows = self._client().query_raw("SELECT COUNT(*) AS n FROM users WHERE UPPER(role) IN ('ADMIN', 'SUPERUSER') AND active = TRUE")
        return int(rows[0]["n"] if rows else 0)

    def issue_session(self, user: dict[str, Any]) -> str:
        user_id = int(user["id"])
        session_id = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + SESSION_DAYS * 86400
        self._ensure_session_storage()
        self._client().execute_raw("DELETE FROM auth_sessions WHERE expires_at < $1", int(time.time()))
        self._client().execute_raw(
            """
            INSERT INTO auth_sessions (user_id, session_hash, expires_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id) DO UPDATE SET
                session_hash = EXCLUDED.session_hash,
                expires_at = EXCLUDED.expires_at
            """,
            user_id,
            _session_hash(session_id),
            expires_at,
        )
        return _encode_session(self.secret, user_id, session_id, expires_at)

    def read_session(self, token: str | None) -> dict[str, Any] | None:
        payload = _decode_session(self.secret, token)
        if payload is None:
            return None
        self._ensure_session_storage()
        rows = self._client().query_raw(
            "SELECT session_hash, expires_at FROM auth_sessions WHERE user_id = $1",
            payload["uid"],
        )
        row = rows[0] if rows else None
        if (
            row is None
            or int(row.get("expires_at") or 0) < int(time.time())
            or not hmac.compare_digest(str(row.get("session_hash") or ""), _session_hash(payload["sid"]))
        ):
            return None
        user = self.get_by_id(payload["uid"])
        if user is None or not user["active"] or user.get("status") != "ACTIVE":
            return None
        return user

    def session_id(self, token: str | None) -> str | None:
        payload = _decode_session(self.secret, token)
        return str(payload["sid"]) if payload is not None else None

    def revoke_session(self, token: str | None) -> None:
        payload = _decode_session(self.secret, token)
        if payload is None:
            return
        self._ensure_session_storage()
        self._client().execute_raw(
            "DELETE FROM auth_sessions WHERE user_id = $1 AND session_hash = $2",
            payload["uid"],
            _session_hash(payload["sid"]),
        )

    def revoke_user_sessions(self, user_id: int) -> None:
        self._ensure_session_storage()
        self._client().execute_raw("DELETE FROM auth_sessions WHERE user_id = $1", int(user_id))


class ResilientAuthStore:
    """Retry the configured PostgreSQL auth source without a restart."""

    def __init__(self, *, primary: PrismaAuthStore) -> None:
        self.primary = primary
        self._active: PrismaAuthStore | None = None
        self._retry_until = time.monotonic() + AUTH_DB_RETRY_SECONDS
        self._lock = threading.Lock()

    def _restore_primary(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if self._active is self.primary:
                return True
            if self._retry_until > now:
                return False
            self._retry_until = now + AUTH_DB_RETRY_SECONDS
            try:
                self.primary.initialize()
            except Exception as error:
                LOGGER.warning("PostgreSQL auth is still unavailable; retrying later: %s", error)
                return False
            self._active = self.primary
            self._retry_until = 0.0
            LOGGER.info("PostgreSQL auth connection restored")
            return True

    def _require_primary(self) -> PrismaAuthStore:
        if not self._restore_primary():
            raise HTTPException(
                status_code=503,
                detail="ฐานข้อมูล PostgreSQL ยังเชื่อมต่อไม่ได้ จึงยังตรวจสอบผู้ใช้นี้ไม่ได้",
            )
        return self.primary

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        return self._require_primary().authenticate(username, password)

    def read_session(self, token: str | None) -> dict[str, Any] | None:
        if not self._restore_primary():
            return None
        return self.primary.read_session(token)

    def session_id(self, token: str | None) -> str | None:
        if not self._restore_primary():
            return None
        return self.primary.session_id(token)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._require_primary(), name)


AuthBackend = AuthStore | PrismaAuthStore | ResilientAuthStore


def current_user(request: Request) -> dict[str, Any] | None:
    store: AuthBackend = request.app.state.auth
    return store.read_session(request.cookies.get(COOKIE_NAME))


def require_user(request: Request) -> dict[str, Any]:
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="กรุณาเข้าสู่ระบบ")
    return user


def has_permission(user: dict[str, Any], permission: str) -> bool:
    return permission in PERMISSIONS.get(canonical_role(user.get("role")), ())


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
