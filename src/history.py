"""Lịch sử và mục yêu thích của giao diện (lưu cục bộ bằng SQLite, không gửi đi đâu).

Mỗi lần đoán tác giả / kiểm tra bài viết / máy viết thử được ghi một dòng. Mục không được đánh dấu yêu thích tự xóa
sau `keep_days` ngày (người dùng chọn 7 / 30 / 90 / 365 ngày hoặc không bao giờ); mục yêu thích giữ mãi.
Không phụ thuộc Streamlit để kiểm thử được (tests/test_history.py).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

KINDS = {"dinh_danh": "Đoán tác giả", "kiem_tra": "Kiểm tra bài viết", "viet": "Máy viết thử", "so_sanh": "So sánh"}
KEEP_CHOICES = {7: "7 ngày", 30: "30 ngày", 90: "3 tháng", 365: "1 năm", 0: "Không tự xóa"}
DEFAULT_KEEP_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,           -- ISO, giờ máy
    kind TEXT NOT NULL,
    title TEXT NOT NULL,                -- một dòng tóm tắt (tác giả dự đoán, đề bài...)
    text TEXT NOT NULL,                 -- văn bản đầu vào hoặc bài máy viết
    detail TEXT NOT NULL DEFAULT '{}',  -- JSON kết quả
    favorite INTEGER NOT NULL DEFAULT 0,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_created ON entries(created_at);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class History:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    # ---------------------------------------------------------------- cài đặt
    @property
    def keep_days(self) -> int:
        with closing(self._conn()) as c:
            r = c.execute("SELECT value FROM settings WHERE key='keep_days'").fetchone()
        return int(r["value"]) if r else DEFAULT_KEEP_DAYS

    @keep_days.setter
    def keep_days(self, days: int) -> None:
        with closing(self._conn()) as c, c:
            c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES('keep_days', ?)", (str(int(days)),))

    # ---------------------------------------------------------------- ghi / đọc
    def add(self, kind: str, title: str, text: str, detail: Optional[dict] = None,
            now: Optional[datetime] = None) -> int:
        """Ghi một mục. Nếu mục gần nhất cùng loại và cùng văn bản thì cập nhật nó (bấm lại không tạo bản trùng)."""
        now = (now or datetime.now()).isoformat(timespec="seconds")
        det = json.dumps(detail or {}, ensure_ascii=False, default=float)
        with closing(self._conn()) as c, c:
            last = c.execute("SELECT id, kind, text FROM entries ORDER BY id DESC LIMIT 1").fetchone()
            if last and last["kind"] == kind and last["text"] == text:
                c.execute("UPDATE entries SET created_at=?, title=?, detail=? WHERE id=?", (now, title, det, last["id"]))
                return int(last["id"])
            cur = c.execute("INSERT INTO entries(created_at, kind, title, text, detail) VALUES(?,?,?,?,?)",
                            (now, kind, title, text, det))
            return int(cur.lastrowid)

    def get(self, entry_id: int) -> Optional[dict]:
        with closing(self._conn()) as c:
            r = c.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return _row(r) if r else None

    def list(self, period: str = "all", kind: Optional[str] = None, favorites_only: bool = False,
             search: str = "", now: Optional[datetime] = None, limit: int = 500) -> list[dict]:
        """period: 'week' (7 ngày qua), 'month' (30 ngày), 'year' (365 ngày), 'all'. Mới nhất trước."""
        q, args = "SELECT * FROM entries WHERE 1=1", []
        days = {"week": 7, "month": 30, "year": 365}.get(period)
        if days:
            q += " AND created_at >= ?"
            args.append(((now or datetime.now()) - timedelta(days=days)).isoformat(timespec="seconds"))
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        if favorites_only:
            q += " AND favorite = 1"
        if search.strip():
            q += " AND (title LIKE ? OR text LIKE ? OR note LIKE ?)"
            args += [f"%{search.strip()}%"] * 3
        q += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(limit)
        with closing(self._conn()) as c:
            return [_row(r) for r in c.execute(q, args).fetchall()]

    def set_favorite(self, entry_id: int, on: bool) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE entries SET favorite=? WHERE id=?", (int(bool(on)), entry_id))

    def set_note(self, entry_id: int, note: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE entries SET note=? WHERE id=?", (note, entry_id))

    def delete(self, entry_id: int) -> None:
        with closing(self._conn()) as c, c:
            c.execute("DELETE FROM entries WHERE id=?", (entry_id,))

    def clear(self, keep_favorites: bool = True) -> int:
        with closing(self._conn()) as c, c:
            cur = c.execute("DELETE FROM entries" + (" WHERE favorite=0" if keep_favorites else ""))
            return cur.rowcount

    def cleanup(self, now: Optional[datetime] = None) -> int:
        """Xóa mục KHÔNG yêu thích cũ hơn keep_days ngày. Trả về số mục đã xóa."""
        days = self.keep_days
        if days <= 0:
            return 0
        cutoff = ((now or datetime.now()) - timedelta(days=days)).isoformat(timespec="seconds")
        with closing(self._conn()) as c, c:
            return c.execute("DELETE FROM entries WHERE favorite=0 AND created_at < ?", (cutoff,)).rowcount

    def stats(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now()
        with closing(self._conn()) as c:
            total = c.execute("SELECT COUNT(*) n FROM entries").fetchone()["n"]
            fav = c.execute("SELECT COUNT(*) n FROM entries WHERE favorite=1").fetchone()["n"]
            oldest = c.execute("SELECT MIN(created_at) t FROM entries WHERE favorite=0").fetchone()["t"]
        expire = None
        if oldest and self.keep_days > 0:
            expire = datetime.fromisoformat(oldest) + timedelta(days=self.keep_days)
        return {"total": total, "favorites": fav, "next_expire": expire}


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["detail"] = json.loads(d.get("detail") or "{}")
    d["favorite"] = bool(d["favorite"])
    d["created"] = datetime.fromisoformat(d["created_at"])
    return d


def group_label(dt: datetime, by: str) -> str:
    """Nhãn nhóm hiển thị: theo ngày / tuần / tháng / năm."""
    if by == "week":
        y, w, _ = dt.isocalendar()
        start = dt - timedelta(days=dt.weekday())
        return f"Tuần {w}/{y} (từ {start:%d/%m})"
    if by == "month":
        return f"Tháng {dt.month}/{dt.year}"
    if by == "year":
        return f"Năm {dt.year}"
    return f"{dt:%d/%m/%Y}"
