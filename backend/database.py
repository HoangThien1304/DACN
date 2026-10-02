"""Lớp lưu trữ dữ liệu cho MedAI Diagnostic Portal.

File này chịu trách nhiệm:
- mở kết nối SQLite
- tạo schema cho bảng bệnh nhân và bảng phiên chụp
- thêm/sửa/xem thông tin bệnh nhân và giai đoạn chẩn đoán
- lưu labels dự đoán dưới dạng JSON để frontend có thể render dễ dàng

Giao diện backend sẽ gọi các hàm trong file này thay vì thao tác DB trực tiếp.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "hospital_ehr.db"


def connect() -> sqlite3.Connection:
    """Mở một kết nối SQLite mới với row_factory để truy cập dữ liệu như dict."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    """Khởi tạo schema database và_seed dữ liệu demo nếu bảng rỗng."""
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medical_record_number TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                gender TEXT NOT NULL,
                phone TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                frontal_path TEXT NOT NULL,
                lateral_path TEXT NOT NULL,
                labels_json TEXT NOT NULL,
                findings TEXT NOT NULL,
                impression TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                approved_at TEXT
            );
            """
        )
        count = connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        if count == 0:
            connection.executemany(
                """INSERT INTO patients
                   (medical_record_number, full_name, date_of_birth, gender, phone)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    ("MR-10428", "Nguyen Minh Anh", "1984-03-18", "Nữ", "090 321 4810"),
                    ("MR-10431", "Tran Quoc Bao", "1972-11-02", "Nam", "091 772 6405"),
                    ("MR-10435", "Le Thu Ha", "1991-07-26", "Nữ", "098 204 1196"),
                ],
            )


def list_patients() -> list[dict[str, Any]]:
    """Lấy danh sách bệnh nhân kèm số phiên chụp và thời gian khám gần nhất."""
    with connect() as connection:
        rows = connection.execute(
            """SELECT p.*, COUNT(e.id) AS exam_count,
                      MAX(e.created_at) AS last_exam
               FROM patients p LEFT JOIN exams e ON e.patient_id = p.id
               GROUP BY p.id ORDER BY p.full_name"""
        ).fetchall()
    return [dict(row) for row in rows]


def create_patient(
    medical_record_number: str,
    full_name: str,
    date_of_birth: str,
    gender: str,
    phone: str,
) -> dict[str, Any]:
    """Thêm bệnh nhân mới và trả về thông tin vừa tạo."""
    with connect() as connection:
        cursor = connection.execute(
            """INSERT INTO patients
               (medical_record_number, full_name, date_of_birth, gender, phone)
               VALUES (?, ?, ?, ?, ?)""",
            (medical_record_number, full_name, date_of_birth, gender, phone),
        )
        row = connection.execute(
            "SELECT * FROM patients WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return dict(row)


def get_patient(patient_id: int) -> dict[str, Any] | None:
    """Lấy thông tin 1 bệnh nhân theo ID."""
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
    return dict(row) if row else None


def create_exam(
    patient_id: int,
    frontal_path: str,
    lateral_path: str,
    labels: dict[str, float],
    findings: str,
    impression: str,
) -> dict[str, Any]:
    """Lưu một phiên chụp mới với kết quả dự đoán từ mô hình AI."""
    with connect() as connection:
        cursor = connection.execute(
            """INSERT INTO exams
               (patient_id, frontal_path, lateral_path, labels_json, findings, impression)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                patient_id,
                frontal_path,
                lateral_path,
                json.dumps(labels),
                findings,
                impression,
            ),
        )
        exam_id = cursor.lastrowid
    return get_exam(exam_id)  # type: ignore[arg-type]


def list_exams(patient_id: int) -> list[dict[str, Any]]:
    """Lấy lịch sử các phiên chụp của bệnh nhân theo thời gian mới nhất đầu tiên."""
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM exams WHERE patient_id = ? ORDER BY created_at DESC, id DESC",
            (patient_id,),
        ).fetchall()
    return [_exam_dict(row) for row in rows]


def get_exam(exam_id: int) -> dict[str, Any] | None:
    """Lấy thông tin một phiên chụp cụ thể theo ID."""
    with connect() as connection:
        row = connection.execute("SELECT * FROM exams WHERE id = ?", (exam_id,)).fetchone()
    return _exam_dict(row) if row else None


def update_exam(
    exam_id: int, findings: str, impression: str, approved: bool
) -> dict[str, Any] | None:
    """Cập nhật findings/impression và trạng thái duyệt của exam."""
    status = "approved" if approved else "draft"
    with connect() as connection:
        connection.execute(
            """UPDATE exams SET findings = ?, impression = ?, status = ?,
                      approved_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END
               WHERE id = ?""",
            (findings, impression, status, approved, exam_id),
        )
    return get_exam(exam_id)


def _exam_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Chuyển row SQLite thành dict Python và giải mã labels_json."""
    result = dict(row)
    result["labels"] = json.loads(result.pop("labels_json"))
    return result
