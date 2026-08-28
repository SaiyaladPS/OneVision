"""Modern PyQt6 desktop interface for Thai/Lao licence-plate scanning."""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QLocale, QThread, QTimer, QUrl, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .config import Settings
from .database import DatabaseRepository
from .service import ScanService


class ROIImageLabel(QLabel):
    """Image viewer with an always-editable, normalised ROI overlay."""

    roi_dragged = pyqtSignal(float, float, float, float)
    roi_changed = pyqtSignal(float, float, float, float)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._roi = (0.05, 0.10, 0.90, 0.80)
        self._drag_start: tuple[float, float] | None = None
        self._drag_roi: tuple[float, float, float, float] | None = None
        self._drag_mode = ""
        self._resize_handle = ""
        self._roi_enabled = True
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_roi(self, x: float, y: float, width: float, height: float) -> None:
        """Set the displayed ROI without emitting a change signal."""

        self._roi = self._clamp_roi(x, y, width, height)
        self.update()

    def set_roi_enabled(self, enabled: bool) -> None:
        self._roi_enabled = bool(enabled)
        self.update()

    @staticmethod
    def _clamp_roi(x: float, y: float, width: float, height: float) -> tuple[float, float, float, float]:
        width = max(0.02, min(1.0, float(width)))
        height = max(0.02, min(1.0, float(height)))
        x = max(0.0, min(1.0 - width, float(x)))
        y = max(0.0, min(1.0 - height, float(y)))
        return x, y, width, height

    def _image_geometry(self) -> tuple[float, float, float, float] | None:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull() or pixmap.width() <= 0 or pixmap.height() <= 0:
            return None
        scale = min(self.width() / pixmap.width(), self.height() / pixmap.height())
        display_width = pixmap.width() * scale
        display_height = pixmap.height() * scale
        return (
            (self.width() - display_width) / 2.0,
            (self.height() - display_height) / 2.0,
            display_width,
            display_height,
        )

    def _normalised_position(self, event: Any) -> tuple[float, float] | None:
        geometry = self._image_geometry()
        if geometry is None:
            return None
        offset_x, offset_y, display_width, display_height = geometry
        x = (float(event.position().x()) - offset_x) / display_width
        y = (float(event.position().y()) - offset_y) / display_height
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        return x, y

    def _handle_at(self, position: tuple[float, float]) -> str:
        x, y, width, height = self._roi
        px, py = position
        threshold = max(0.018, 10.0 / max(1.0, min(self.width(), self.height())))
        handles = {
            "top_left": (x, y),
            "top_right": (x + width, y),
            "bottom_left": (x, y + height),
            "bottom_right": (x + width, y + height),
        }
        for name, (hx, hy) in handles.items():
            if abs(px - hx) <= threshold and abs(py - hy) <= threshold:
                return name
        return ""

    def _emit_roi(self) -> None:
        self.roi_changed.emit(*self._roi)
        self.update()

    def _update_drag(self, position: tuple[float, float]) -> None:
        if self._drag_start is None or self._drag_roi is None:
            return
        sx, sy = self._drag_start
        x, y, width, height = self._drag_roi
        px, py = position
        if self._drag_mode == "move":
            x += px - sx
            y += py - sy
        elif self._drag_mode == "new":
            x, y = min(sx, px), min(sy, py)
            width, height = abs(px - sx), abs(py - sy)
        else:
            right, bottom = x + width, y + height
            if "left" in self._resize_handle:
                x = min(px, right - 0.02)
                width = right - x
            elif "right" in self._resize_handle:
                width = max(0.02, px - x)
            if "top" in self._resize_handle:
                y = min(py, bottom - 0.02)
                height = bottom - y
            elif "bottom" in self._resize_handle:
                height = max(0.02, py - y)
        self._roi = self._clamp_roi(x, y, width, height)
        self._emit_roi()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = self._normalised_position(event)
            if position is None:
                event.ignore()
                return
            x, y, width, height = self._roi
            self._drag_start = position
            self._drag_roi = self._roi
            self._resize_handle = self._handle_at(position)
            if self._resize_handle:
                self._drag_mode = "resize"
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif x <= position[0] <= x + width and y <= position[1] <= y + height:
                self._drag_mode = "move"
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self._drag_mode = "new"
                self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag_start is not None:
            position = self._normalised_position(event)
            if position is not None:
                self._update_drag(position)
            event.accept()
            return
        position = self._normalised_position(event)
        if position is not None and self._handle_at(position):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif position is not None:
            x, y, width, height = self._roi
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if x <= position[0] <= x + width and y <= position[1] <= y + height
                else Qt.CursorShape.CrossCursor
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            position = self._normalised_position(event)
            if position is not None:
                self._update_drag(position)
            roi = self._roi
            self._drag_start = None
            self._drag_roi = None
            self._drag_mode = ""
            self._resize_handle = ""
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.roi_dragged.emit(*roi)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if not self.isEnabled() or not self._roi_enabled or self.pixmap() is None or self.pixmap().isNull():
            return
        geometry = self._image_geometry()
        if geometry is None:
            return
        from PyQt6.QtGui import QBrush

        offset_x, offset_y, display_width, display_height = geometry
        x, y, width, height = self._roi
        rect_x = offset_x + x * display_width
        rect_y = offset_y + y * display_height
        rect_w = width * display_width
        rect_h = height * display_height
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#19d3ae"), 3))
        painter.setBrush(QBrush(QColor(25, 211, 174, 28)))
        painter.drawRect(int(rect_x), int(rect_y), int(rect_w), int(rect_h))
        painter.setBrush(QBrush(QColor("#19d3ae")))
        for hx, hy in ((rect_x, rect_y), (rect_x + rect_w, rect_y), (rect_x, rect_y + rect_h), (rect_x + rect_w, rect_y + rect_h)):
            painter.drawEllipse(int(hx - 5), int(hy - 5), 10, 10)
        painter.end()


class ScanWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, settings: Settings, image_path: Path) -> None:
        super().__init__()
        self.settings = settings
        self.image_path = image_path

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = ScanService(self.settings).scan_file(self.image_path)
            if self.settings.database_url:
                try:
                    repository = DatabaseRepository(self.settings.database_url)
                    repository.initialize_schema()
                    result["database_id"] = repository.save_scan(result)
                    result["database_status"] = "บันทึก PostgreSQL แล้ว"
                except Exception as error:
                    result["database_status"] = f"PostgreSQL บันทึกไม่สำเร็จ: {error}"
            else:
                result["database_status"] = "ยังไม่ได้ตั้งค่า PostgreSQL"
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class VideoScanWorker(QObject):
    """Run video scanning in a worker thread and stream preview frames to Qt."""

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    frame_ready = pyqtSignal(object, int, int)
    plate_ready = pyqtSignal(dict)

    def __init__(self, settings: Settings, video_path: Path) -> None:
        super().__init__()
        self.settings = settings
        self.video_path = video_path
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = ScanService(self.settings).scan_video(
                self.video_path,
                frame_callback=lambda frame, index, total: self.frame_ready.emit(frame, index, total),
                plate_callback=self.plate_ready.emit,
                stop_requested=self._stop_requested.is_set,
            )
            if self.settings.database_url:
                try:
                    repository = DatabaseRepository(self.settings.database_url)
                    repository.initialize_schema()
                    result["database_id"] = repository.save_scan(result)
                    result["database_status"] = "บันทึก PostgreSQL แล้ว"
                except Exception as error:
                    result["database_status"] = f"PostgreSQL บันทึกไม่สำเร็จ: {error}"
            else:
                result["database_status"] = "ยังไม่ได้ตั้งค่า PostgreSQL"
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class CameraScanWorker(QObject):
    """Own a camera capture and emit only temporally confirmed registrations."""

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    frame_ready = pyqtSignal(object, int)
    plate_ready = pyqtSignal(dict)

    def __init__(self, settings: Settings, camera_index: int, stop_after_first: bool) -> None:
        super().__init__()
        self.settings = settings
        self.camera_index = camera_index
        self.stop_after_first = stop_after_first
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = ScanService(self.settings).scan_camera(
                camera_index=self.camera_index,
                frame_callback=lambda frame, index: self.frame_ready.emit(frame, index),
                plate_callback=self.plate_ready.emit,
                stop_requested=self._stop_requested.is_set,
                stop_after_first=self.stop_after_first,
            )
            if self.settings.database_url and result.get("plates"):
                try:
                    repository = DatabaseRepository(self.settings.database_url)
                    repository.initialize_schema()
                    result["database_id"] = repository.save_scan(result)
                    result["database_status"] = "บันทึก PostgreSQL แล้ว"
                except Exception as error:
                    result["database_status"] = f"PostgreSQL บันทึกไม่สำเร็จ: {error}"
            elif self.settings.database_url:
                result["database_status"] = "ไม่มีทะเบียนที่ยืนยันสำหรับบันทึก"
            else:
                result["database_status"] = "ยังไม่ได้ตั้งค่า PostgreSQL"
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class PlateCard(QFrame):
    """Clickable crop card used in the horizontal plate gallery."""

    clicked = pyqtSignal(int)

    def __init__(self, plate: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plate = plate
        self.setObjectName("plateCard")
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(7)

        crop = QLabel("ไม่มีภาพ crop")
        crop.setObjectName("cropPreview")
        crop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop.setMinimumHeight(108)
        crop_path = Path(
            str(
                plate.get("character_annotated_image")
                or plate.get("ocr_ready_image")
                or plate.get("crop_image", "")
            )
        )
        if crop_path.is_file():
            pixmap = QPixmap(str(crop_path))
            crop.setPixmap(pixmap.scaled(198, 108, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(crop)

        country = _country_label(str(plate.get("country", "unknown")))
        title = QLabel(f"ป้ายที่ {plate.get('id', '-')}  •  {country}")
        title.setObjectName("cardTitle")
        number = QLabel(_registration_text(plate) or "อ่านทะเบียนไม่ได้")
        number.setObjectName("cardNumber")
        province = QLabel(str(plate.get("province") or "ไม่ทราบจังหวัด/แขวง"))
        province.setObjectName("cardProvince")
        vehicle_type = QLabel(f"ประเภทรถ: {plate.get('vehicle_type') or 'ไม่ทราบ'}")
        vehicle_type.setObjectName("cardProvince")
        layout.addWidget(title)
        layout.addWidget(number)
        layout.addWidget(province)
        layout.addWidget(vehicle_type)

    def mousePressEvent(self, event: Any) -> None:
        self.clicked.emit(int(self.plate.get("id", 0)) - 1)
        super().mousePressEvent(event)


class PlateLayoutWidget(QFrame):
    """Draw a compact plate-shaped summary resembling Thai/Lao plates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plate: dict[str, Any] = {}
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_plate(self, plate: dict[str, Any] | None) -> None:
        self.plate = plate or {}
        self.update()

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(18, 18, -18, -18)
        country = self.plate.get("country", "")
        prefix = str(self.plate.get("plate_prefix") or "")
        number = str(self.plate.get("plate_number") or "")
        province = str(self.plate.get("province") or "")
        if country == "thai":
            background, border, text = QColor("#f6d94c"), QColor("#172033"), QColor("#101827")
            heading, registration = "THAILAND  01", _registration_text(self.plate)
            bottom = province or "กรุงเทพมหานคร"
        elif country == "lao":
            background, border, text = QColor("#f5f7fa"), QColor("#19734b"), QColor("#12202b")
            heading, registration = "LAO P.D.R.", " ".join(part for part in (prefix, number) if part)
            bottom = province or "Lao licence plate"
        else:
            background, border, text = QColor("#edf1f5"), QColor("#8793a1"), QColor("#17202b")
            heading, registration, bottom = "LICENCE PLATE", "ยังไม่มีผลลัพธ์", ""
        painter.setBrush(background)
        painter.setPen(QPen(border, 4))
        painter.drawRoundedRect(area, 12, 12)
        painter.setPen(text)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(area.adjusted(0, 12, 0, 0), Qt.AlignmentFlag.AlignHCenter, heading)
        painter.setFont(QFont("Arial", 31, QFont.Weight.Bold))
        painter.drawText(area.adjusted(8, 35, -8, -34), Qt.AlignmentFlag.AlignCenter, registration)
        painter.setFont(QFont("Arial", 10))
        painter.drawText(area.adjusted(8, 0, -8, -12), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, bottom)


def _registration_text(plate: dict[str, Any]) -> str:
    country = plate.get("country", "")
    prefix = str(plate.get("plate_prefix") or "")
    number = str(plate.get("plate_number") or "")
    ocr = plate.get("ocr", {})
    if country == "thai" and prefix and number:
        return f"{prefix}-{number}"
    if country == "lao" and (prefix or number):
        return " ".join(part for part in (prefix, number) if part)
    return number or str(ocr.get("text") or "-")


def _display_with_code(value: object, code: object) -> str:
    """Show readable plate text together with its stable model identifier."""

    text = str(value or "-")
    identifier = str(code or "")
    return f"{text} ({identifier})" if identifier else text


def _plate_codes(plate: dict[str, Any]) -> tuple[str, str]:
    country = str(plate.get("country") or "")
    readings = plate.get("country_readings", {})
    reading = readings.get(country, {}) if isinstance(readings, dict) else {}
    if not isinstance(reading, dict):
        reading = {}
    return (
        str(plate.get("plate_prefix_code") or reading.get("plate_prefix_code") or ""),
        str(plate.get("province_code") or reading.get("province_code") or ""),
    )


def _country_label(country: str, detailed: bool = False) -> str:
    labels = {
        "thai": ("ไทย", "ไทย (Thailand)"),
        "lao": ("ลาว", "ลาว (Laos)"),
        "unknown": ("ไม่ทราบ", "ไม่ทราบประเทศ (Unknown)"),
    }
    short, long = labels.get(country, (country or "ไม่ทราบ", country or "Unknown"))
    return long if detailed else short


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.selected_image: Path | None = None
        self.media_type = "image"
        self.display_path: Path | None = None
        self.preview_frame: Any | None = None
        self._pending_preview: tuple[Any, int, int] | None = None
        self.last_result: dict[str, Any] | None = None
        self.plates: list[dict[str, Any]] = []
        self.thread: QThread | None = None
        self.worker: QObject | None = None
        self.setWindowTitle("Car Scan • Thai / Lao Licence Plate")
        self.resize(1440, 900)
        self._build_ui()
        # Rendering a Qt pixmap can take longer than a video frame interval.
        # Keep only the newest worker frame and repaint at a bounded rate so
        # queued preview signals cannot make the UI lag behind the scan.
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)
        self._preview_timer.timeout.connect(self._flush_preview)
        self._preview_timer.start()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f7fb; color: #17202b; font-size: 13px; }
            QFrame#header { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #102a43, stop:1 #176b87); border-radius: 14px; }
            QLabel#appTitle { color: white; font-size: 24px; font-weight: 700; }
            QLabel#appSubtitle, QLabel#statusLabel { color: #cbd8e6; }
            QGroupBox { background: white; border: 1px solid #dbe3ed; border-radius: 12px; margin-top: 12px; padding: 14px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #31516d; }
            QLabel#imageViewer { background: #172536; color: #cbd8e6; border-radius: 10px; border: 1px solid #29445c; }
            QLabel#cropPreview { background: #eef2f6; color: #718096; border-radius: 7px; }
            QFrame#plateCard { background: white; border: 1px solid #dbe3ed; border-radius: 10px; }
            QFrame#plateCard:hover { border: 2px solid #2e8bcb; }
            QLabel#cardTitle { color: #64748b; font-weight: 700; }
            QLabel#cardNumber { color: #132b42; font-size: 19px; font-weight: 700; }
            QLabel#cardProvince { color: #718096; }
            QPushButton { background: #2b7db5; color: white; border: 0; border-radius: 8px; padding: 9px 14px; font-weight: 700; }
            QPushButton:hover { background: #17668f; }
            QPushButton#primaryButton { background: #08a77a; }
            QPushButton#primaryButton:hover { background: #078763; }
            QPushButton#dangerButton { background: #d9535f; }
            QPushButton#dangerButton:hover { background: #b63f4a; }
            QPushButton#secondaryButton { background: #607d96; }
            QPushButton:disabled { background: #b7c4d1; }
            QComboBox, QDoubleSpinBox { min-height: 30px; border: 1px solid #cbd8e3; border-radius: 6px; padding: 2px 7px; background: #fbfdff; }
            QTableWidget::item:selected { background: #d8f2ec; color: #123d45; }
            QLabel#roiTip { color: #607d96; font-size: 12px; font-weight: 400; }
            QTableWidget { background: white; border: 1px solid #dbe3ed; border-radius: 8px; gridline-color: #e7edf3; }
            QHeaderView::section { background: #edf3f8; color: #31516d; padding: 8px; border: 0; font-weight: 700; }
            QScrollArea { border: 0; background: transparent; }
            """
        )
        root = QVBoxLayout()
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 16, 22, 16)
        title_layout = QVBoxLayout()
        title = QLabel("Car Scan")
        title.setObjectName("appTitle")
        subtitle = QLabel("ระบบตรวจจับและอ่านป้ายทะเบียนรถไทย / ลาว")
        subtitle.setObjectName("appSubtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        self.db_label = QLabel("● PostgreSQL: " + ("เชื่อมต่อจาก .env" if self.settings.database_url else "ยังไม่ได้ตั้งค่า"))
        self.db_label.setObjectName("statusLabel")
        header_layout.addWidget(self.db_label, alignment=Qt.AlignmentFlag.AlignTop)
        root.addWidget(header)

        controls = QHBoxLayout()
        self.image_button = QPushButton("อัปโหลดรูปภาพ")
        self.image_button.setObjectName("secondaryButton")
        self.image_button.setToolTip("เลือกรูปรถเพื่อดูและปรับ ROI")
        self.image_button.clicked.connect(self.choose_image)
        self.video_button = QPushButton("อัปโหลดวิดีโอ")
        self.video_button.setObjectName("secondaryButton")
        self.video_button.setToolTip("เลือกวิดีโอรถเพื่อสแกนทะเบียน")
        self.video_button.clicked.connect(self.choose_video)
        self.camera_button = QPushButton("เปิดกล้อง")
        self.camera_button.setObjectName("secondaryButton")
        self.camera_button.setToolTip("เปิดกล้องและตรวจจับทะเบียนแบบต่อเนื่อง")
        self.camera_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("หยุดสแกน")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_active_scan)
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["กล้อง 0", "กล้อง 1", "กล้อง 2"])
        self.stop_after_first = QCheckBox("หยุดเมื่อพบทะเบียนที่มั่นใจ")
        self.stop_after_first.setChecked(False)
        self.scan_button = QPushButton("เริ่มสแกน")
        self.scan_button.setObjectName("primaryButton")
        self.scan_button.setEnabled(False)
        self.scan_button.clicked.connect(self.start_scan)
        self.save_button = QPushButton("บันทึกผล JSON")
        self.save_button.setObjectName("secondaryButton")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_json)
        self.open_video_button = QPushButton("เปิดวิดีโอผลลัพธ์")
        self.open_video_button.setObjectName("secondaryButton")
        self.open_video_button.setEnabled(False)
        self.open_video_button.clicked.connect(self.open_output_video)
        controls.addWidget(self.image_button)
        controls.addWidget(self.video_button)
        controls.addWidget(self.camera_button)
        controls.addWidget(self.camera_selector)
        controls.addWidget(self.stop_after_first)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.scan_button)
        controls.addWidget(self.save_button)
        controls.addWidget(self.open_video_button)
        controls.addStretch()
        self.status_label = QLabel("พร้อมใช้งาน • เลือกรูปภาพเพื่อเริ่มต้น")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setStyleSheet("color: #55708a;")
        controls.addWidget(self.status_label)
        root.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([820, 540])
        root.addWidget(splitter, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.addWidget(self._build_roi_controls())
        image_group = QGroupBox("ภาพรถและภาพที่ตีกรอบป้ายทะเบียน")
        image_layout = QVBoxLayout(image_group)
        self.image_label = QLabel("ยังไม่ได้เลือกรูปภาพ")
        self.image_label = ROIImageLabel("Select an image")
        self.image_label.roi_dragged.connect(self._on_roi_dragged)
        self.image_label.roi_changed.connect(self._on_roi_changed)
        self.image_label.setObjectName("imageViewer")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(560, 400)
        self.image_label.set_roi(
            self.roi_x.value(), self.roi_y.value(), self.roi_width.value(), self.roi_height.value()
        )
        self.image_label.set_roi_enabled(self.roi_enabled.isChecked())
        image_layout.addWidget(self.image_label, 1)
        layout.addWidget(image_group, 1)

        crop_group = QGroupBox("ภาพ crop ที่ปรับก่อน OCR • คลิกการ์ดเพื่อดูรายละเอียด")
        crop_group.setMinimumHeight(190)
        crop_scroll = QScrollArea()
        crop_scroll.setWidgetResizable(True)
        self.crop_container = QWidget()
        self.crop_layout = QHBoxLayout(self.crop_container)
        self.crop_layout.setContentsMargins(4, 4, 4, 4)
        self.crop_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        crop_scroll.setWidget(self.crop_container)
        crop_group_layout = QVBoxLayout(crop_group)
        crop_group_layout.addWidget(crop_scroll)
        layout.addWidget(crop_group)
        return panel

    @staticmethod
    def _roi_spinbox(value: float) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setLocale(QLocale("en_US"))
        spinbox.setRange(0.0, 1.0)
        spinbox.setSingleStep(0.01)
        spinbox.setDecimals(2)
        spinbox.setValue(max(0.0, min(1.0, float(value))))
        spinbox.setFixedWidth(82)
        return spinbox

    def _build_roi_controls(self) -> QGroupBox:
        """Build live ROI controls used by image, video, and camera scans."""

        from .pipeline import load_pipeline_config

        config = load_pipeline_config(self.settings.pipeline_config)
        roi = self.settings.scan_roi_override
        if roi is None:
            roi = config.get("scan_roi", {})
        if not isinstance(roi, dict):
            roi = {}
        self.roi_group = QGroupBox("Scan ROI - adjust in GUI")
        group_layout = QVBoxLayout(self.roi_group)
        top = QHBoxLayout()
        self.roi_enabled = QCheckBox("Enabled")
        self.roi_enabled.setChecked(bool(roi.get("enabled", False)))
        self.roi_shape = QComboBox()
        self.roi_shape.addItem("Rectangle", "rectangle")
        self.roi_shape.addItem("Circle", "circle")
        self.roi_shape.addItem("Ellipse", "ellipse")
        shape_index = self.roi_shape.findData(str(roi.get("shape", "rectangle")).lower())
        self.roi_shape.setCurrentIndex(max(0, shape_index))
        top.addWidget(self.roi_enabled)
        top.addWidget(QLabel("Shape"))
        top.addWidget(self.roi_shape)
        top.addStretch()
        full_button = QPushButton("Full")
        full_button.clicked.connect(lambda: self._set_roi_fields(0.0, 0.0, 1.0, 1.0))
        centre_button = QPushButton("Center")
        centre_button.clicked.connect(lambda: self._set_roi_fields(0.10, 0.10, 0.80, 0.80))
        top.addWidget(full_button)
        top.addWidget(centre_button)
        self.roi_apply_button = QPushButton("Apply ROI")
        self.roi_apply_button.clicked.connect(self._apply_roi_from_gui)
        top.addWidget(self.roi_apply_button)
        group_layout.addLayout(top)

        values = QHBoxLayout()
        self.roi_x = self._roi_spinbox(float(roi.get("x", 0.05)))
        self.roi_y = self._roi_spinbox(float(roi.get("y", 0.10)))
        self.roi_width = self._roi_spinbox(float(roi.get("width", 0.90)))
        self.roi_height = self._roi_spinbox(float(roi.get("height", 0.80)))
        for label, spinbox in (
            ("X", self.roi_x),
            ("Y", self.roi_y),
            ("Width", self.roi_width),
            ("Height", self.roi_height),
        ):
            values.addWidget(QLabel(label))
            values.addWidget(spinbox)
        values.addWidget(QLabel("normalized 0-1"))
        values.addStretch()
        group_layout.addLayout(values)
        tip = QLabel("ลากภายในกรอบเพื่อย้าย • ลากจุดมุมเพื่อปรับขนาด • ลากนอกกรอบเพื่อสร้างใหม่")
        tip.setObjectName("roiTip")
        group_layout.addWidget(tip)
        return self.roi_group

    def _set_roi_fields(self, x: float, y: float, width: float, height: float) -> None:
        self.roi_x.setValue(x)
        self.roi_y.setValue(y)
        self.roi_width.setValue(width)
        self.roi_height.setValue(height)
        if hasattr(self, "image_label"):
            self.image_label.set_roi(x, y, width, height)

    @pyqtSlot(float, float, float, float)
    def _on_roi_changed(self, x: float, y: float, width: float, height: float) -> None:
        """Keep numeric controls and the active settings in sync while dragging."""

        self._set_roi_fields(x, y, width, height)
        self.roi_enabled.setChecked(True)
        self.image_label.set_roi_enabled(True)
        self._apply_roi_from_gui(show_status=False, refresh=False)

    @pyqtSlot(float, float, float, float)
    def _on_roi_dragged(self, x: float, y: float, width: float, height: float) -> None:
        self._set_roi_fields(x, y, width, height)
        self.roi_enabled.setChecked(True)
        self._apply_roi_from_gui()

    def _apply_roi_from_gui(self, show_status: bool = True, refresh: bool = True) -> None:
        from scan import LicensePlateScanner

        roi = {
            "enabled": self.roi_enabled.isChecked(),
            "shape": str(self.roi_shape.currentData() or "rectangle"),
            "unit": "normalized",
            "x": self.roi_x.value(),
            "y": self.roi_y.value(),
            "width": self.roi_width.value(),
            "height": self.roi_height.value(),
            "color": "#ff0000",
            "thickness": 3,
        }
        if roi["x"] + roi["width"] > 1.0 or roi["y"] + roi["height"] > 1.0:
            QMessageBox.warning(self, "Invalid ROI", "X + Width and Y + Height must stay within 1.00")
            return
        try:
            LicensePlateScanner._normalise_scan_roi(roi)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid ROI", str(error))
            return
        self.settings = replace(self.settings, scan_roi_override=roi)
        if hasattr(self, "image_label"):
            self.image_label.set_roi_enabled(roi["enabled"])
        if refresh:
            self.preview_frame = None
            if self.media_type == "image" and self.display_path is not None:
                self._show_image(self.display_path)
        if show_status:
            self.status_label.setText("ROI applied for the current session")

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        self.plate_layout = PlateLayoutWidget()
        plate_group = QGroupBox("รูปแบบป้ายทะเบียน")
        plate_group_layout = QVBoxLayout(plate_group)
        plate_group_layout.addWidget(self.plate_layout)
        layout.addWidget(plate_group)

        detail_group = QGroupBox("รายละเอียดป้ายที่เลือก")
        details = QFormLayout(detail_group)
        self.prefix_value = QLabel("-")
        self.number_value = QLabel("-")
        self.province_value = QLabel("-")
        self.country_value = QLabel("-")
        self.vehicle_type_value = QLabel("-")
        self.confidence_value = QLabel("-")
        self.method_value = QLabel("-")
        self.ocr_value = QLabel("-")
        self.ocr_value.setWordWrap(True)
        fields = (("ประเทศ", self.country_value), ("ประเภทรถ", self.vehicle_type_value), ("คำนำหน้าทะเบียน", self.prefix_value), ("เลขทะเบียน", self.number_value), ("จังหวัด/แขวง", self.province_value), ("ความมั่นใจ OCR", self.confidence_value), ("วิธีอ่าน", self.method_value), ("ข้อความ OCR", self.ocr_value))
        for label, value in fields:
            value.setStyleSheet("font-weight: 600; color: #1e4666;")
            details.addRow(label, value)
        layout.addWidget(detail_group)

        result_group = QGroupBox("รายการผลตรวจทั้งหมด")
        result_layout = QVBoxLayout(result_group)
        self.results = QTableWidget(0, 7)
        self.results.setHorizontalHeaderLabels(["#", "ประเทศ", "ประเภทรถ", "คำนำหน้า", "เลขทะเบียน", "จังหวัด/แขวง", "มั่นใจ"])
        self.results.horizontalHeader().setStretchLastSection(True)
        self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results.cellClicked.connect(self._table_plate_clicked)
        result_layout.addWidget(self.results)
        layout.addWidget(result_group, 1)
        return panel

    def choose_image(self) -> None:
        self.choose_media("image")

    def choose_video(self) -> None:
        self.choose_media("video")

    def choose_media(self, media_type: str) -> None:
        file_filter = "Images (*.jpg *.jpeg *.png *.bmp *.webp)" if media_type == "image" else "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.webm)"
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "อัปโหลดรูปภาพรถ" if media_type == "image" else "อัปโหลดวิดีโอรถ",
            str(self.settings.root),
            file_filter,
        )
        if not filename:
            return
        self.selected_image = Path(filename)
        self.media_type = media_type
        self.display_path = self.selected_image if self.media_type == "image" else None
        self.preview_frame = None
        self.last_result = None
        self.plates = []
        self._populate_results()
        self.open_video_button.setEnabled(False)
        if self.media_type == "image":
            self._show_image(self.display_path)
            self.status_label.setText(f"เลือกภาพแล้ว: {self.selected_image.name} • พร้อมสแกน")
        else:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"เลือกวิดีโอแล้ว\n{self.selected_image.name}\nพร้อมสแกน")
            self.status_label.setText(f"เลือกวิดีโอแล้ว: {self.selected_image.name} • จะสแกนทุก {self.settings.video_frame_stride} เฟรม")
        self.scan_button.setEnabled(True)
        self.save_button.setEnabled(False)

    def _show_image(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self.image_label.setText("ไม่มีภาพสำหรับแสดง")
            self.image_label.setPixmap(QPixmap())
            return
        try:
            # Use the same Unicode-safe reader as the scanner. This also
            # makes the configured ROI visible before the user starts a scan.
            from scan import LicensePlateScanner, read_image
            from src.car_scan.pipeline import load_pipeline_config

            if self.preview_frame is None or path != self.selected_image:
                frame = read_image(path)
                preview = LicensePlateScanner.__new__(LicensePlateScanner)
                config = load_pipeline_config(self.settings.pipeline_config)
                roi_config = self.settings.scan_roi_override
                if roi_config is None:
                    roi_config = config.get("scan_roi")
                preview.scan_roi = preview._normalise_scan_roi(roi_config)
                self.preview_frame = preview.draw_scan_roi(frame)
            self._show_frame(self.preview_frame)
        except Exception:
            pixmap = QPixmap(str(path))
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._show_image(self.display_path)

    def start_scan(self) -> None:
        if self.selected_image is None or self.thread is not None:
            return
        self._set_scanning(True, can_stop=self.media_type == "video")
        self.open_video_button.setEnabled(False)
        self.status_label.setText("กำลังโหลดโมเดลและสแกน... กรุณารอสักครู่")
        self.thread = QThread(self)
        if self.media_type == "video":
            worker = VideoScanWorker(self.settings, self.selected_image)
            worker.frame_ready.connect(self.on_video_frame)
            worker.plate_ready.connect(self.on_video_plate)
            self.worker = worker
        else:
            self.worker = ScanWorker(self.settings, self.selected_image)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)  # type: ignore[attr-defined]
        self.worker.finished.connect(self.on_scan_finished)  # type: ignore[attr-defined]
        self.worker.failed.connect(self.on_scan_failed)  # type: ignore[attr-defined]
        self.worker.finished.connect(self.thread.quit)  # type: ignore[attr-defined]
        self.worker.failed.connect(self.thread.quit)  # type: ignore[attr-defined]
        self.thread.finished.connect(self._clear_thread)
        self.thread.start()

    def start_camera(self) -> None:
        if self.thread is not None:
            return
        self.media_type = "camera"
        self.selected_image = None
        self.display_path = None
        self.last_result = None
        self.plates = []
        self._populate_results()
        self.open_video_button.setEnabled(False)
        self._set_scanning(True, can_stop=True)
        camera_index = self.camera_selector.currentIndex()
        self.status_label.setText(
            f"กำลังเปิดกล้อง {camera_index} • จะแสดงผลเมื่ออ่านทะเบียนเดิมได้อย่างน้อย "
            f"{self.settings.camera_min_confirmations} ครั้ง"
        )
        self.thread = QThread(self)
        worker = CameraScanWorker(
            self.settings,
            camera_index,
            self.stop_after_first.isChecked(),
        )
        worker.frame_ready.connect(self.on_camera_frame)
        worker.plate_ready.connect(self.on_camera_plate)
        self.worker = worker
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.finished.connect(self.on_scan_finished)
        worker.failed.connect(self.on_scan_failed)
        worker.finished.connect(self.thread.quit)
        worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._clear_thread)
        self.thread.start()

    def stop_active_scan(self) -> None:
        worker = self.worker
        if isinstance(worker, (VideoScanWorker, CameraScanWorker)):
            worker.stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText("กำลังหยุดสแกนและสรุปผลที่ยืนยันแล้ว...")

    def _set_scanning(self, active: bool, can_stop: bool = False) -> None:
        self.image_button.setEnabled(not active)
        self.video_button.setEnabled(not active)
        self.camera_button.setEnabled(not active)
        if hasattr(self, "roi_group"):
            self.roi_group.setEnabled(not active)
        self.camera_selector.setEnabled(not active)
        self.stop_after_first.setEnabled(not active)
        self.scan_button.setEnabled(not active and self.selected_image is not None)
        self.stop_button.setEnabled(active and can_stop)

    @pyqtSlot(object, int, int)
    def on_video_frame(self, frame: Any, frame_index: int, frame_count: int) -> None:
        """Show an annotated processing preview without blocking the worker."""

        if frame is None:
            return
        self._pending_preview = (frame, frame_index, frame_count)

    @pyqtSlot(object, int)
    def on_camera_frame(self, frame: Any, frame_index: int) -> None:
        if frame is None:
            return
        self._pending_preview = (frame, frame_index, 0)

    @pyqtSlot()
    def _flush_preview(self) -> None:
        pending = self._pending_preview
        self._pending_preview = None
        if pending is None:
            return
        frame, frame_index, frame_count = pending
        self._show_frame(frame)
        if self.media_type == "video":
            progress = f"{frame_index}/{frame_count}" if frame_count > 0 else str(frame_index)
            self.status_label.setText(f"กำลังสแกนวิดีโอ: เฟรม {progress}")
        elif self.media_type == "camera":
            self.status_label.setText(
                f"กำลังสแกนกล้องสด • เฟรม {frame_index} • ยืนยันแล้ว {len(self.plates)} ทะเบียน"
            )

    def _show_frame(self, frame: Any) -> None:
        rgb = frame[:, :, ::-1].copy()
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    @pyqtSlot(dict)
    def on_camera_plate(self, plate: dict[str, Any]) -> None:
        """Publish a camera result only after temporal confirmation."""

        self._publish_live_plate(plate, "กล้องสด")

    @pyqtSlot(dict)
    def on_video_plate(self, plate: dict[str, Any]) -> None:
        """Publish a confirmed video result before the file reaches EOF."""

        self._publish_live_plate(plate, "วิดีโอ")

    def _publish_live_plate(self, plate: dict[str, Any], source_label: str) -> None:
        """Update the table/card immediately when temporal voting confirms a plate."""

        key = f"{plate.get('country')}|{plate.get('plate_prefix')}{plate.get('plate_number')}"
        live_result_key = str(plate.get("live_result_key") or "")
        existing_index = next(
            (
                index
                for index, item in enumerate(self.plates)
                if (
                    live_result_key
                    and str(item.get("live_result_key") or "") == live_result_key
                )
                or f"{item.get('country')}|{item.get('plate_prefix')}{item.get('plate_number')}" == key
            ),
            None,
        )
        if existing_index is None:
            self.plates.append(plate)
            selected_index = len(self.plates) - 1
        else:
            plate["id"] = self.plates[existing_index].get("id", existing_index + 1)
            self.plates[existing_index] = plate
            selected_index = existing_index
        self.last_result = {
            "media_type": "camera" if source_label == "กล้องสด" else "video",
            "input": (
                f"camera://{self.camera_selector.currentIndex()}"
                if source_label == "กล้องสด"
                else str(self.selected_image or "")
            ),
            "plate_count": len(self.plates),
            "plates": list(self.plates),
        }
        self.save_button.setEnabled(True)
        self._populate_results()
        self._select_plate(selected_index)
        self.status_label.setText(
            f"{source_label}: ยืนยันทะเบียน {_registration_text(plate)} แล้วจาก "
            f"{plate.get('camera_occurrences', plate.get('video_occurrences', self.settings.camera_min_confirmations))} เฟรม"
        )

    @pyqtSlot(dict)
    def on_scan_finished(self, result: dict[str, Any]) -> None:
        self.last_result = result
        self.plates = list(result.get("plates", []))
        self.save_button.setEnabled(True)
        self._set_scanning(False)
        if result.get("media_type") == "video":
            state = "หยุดสแกนวิดีโอแล้ว" if result.get("cancelled") else "สแกนวิดีโอเสร็จ"
            self.status_label.setText(
                f"{state}: ยืนยัน {result['plate_count']} ทะเบียน จาก {result.get('sampled_frames', 0)} เฟรมตัวอย่าง "
                f"• {result.get('database_status', '')}"
            )
            self.open_video_button.setEnabled(Path(str(result.get("output_video", ""))).is_file())
        elif result.get("media_type") == "camera":
            self.status_label.setText(
                f"ปิดกล้องแล้ว: ยืนยัน {result['plate_count']} ทะเบียน จาก {result.get('sampled_frames', 0)} เฟรมตัวอย่าง "
                f"• {result.get('database_status', '')}"
            )
            snapshot = Path(str(result.get("annotated_image", "")))
            if snapshot.is_file():
                self.display_path = snapshot
        else:
            rejected = int(result.get("rejected_plate_count", 0))
            rejected_text = f" • ไม่แสดงผลที่ยังไม่มั่นใจ {rejected} ป้าย" if rejected else ""
            self.status_label.setText(
                f"ยืนยัน {result['plate_count']} ป้าย{rejected_text} • {result.get('database_status', '')}"
            )
            self.display_path = Path(result["annotated_image"])
            self._show_image(self.display_path)
        self._populate_results()
        if self.plates:
            self._select_plate(0)
        else:
            self.plate_layout.set_plate(None)

    @pyqtSlot(str)
    def on_scan_failed(self, message: str) -> None:
        self._set_scanning(False)
        self.status_label.setText("สแกนไม่สำเร็จ")
        QMessageBox.critical(self, "เกิดข้อผิดพลาด", message)

    def _clear_thread(self) -> None:
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self._set_scanning(False)

    def _populate_results(self) -> None:
        while self.crop_layout.count():
            item = self.crop_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for plate in self.plates:
            card = PlateCard(plate)
            card.clicked.connect(self._select_plate)
            self.crop_layout.addWidget(card)
        self.crop_layout.addStretch()

        self.results.setRowCount(0)
        for plate in self.plates:
            row = self.results.rowCount()
            self.results.insertRow(row)
            ocr = plate.get("ocr", {})
            confidence = float(plate.get("recognition_confidence", ocr.get("confidence", 0.0)))
            prefix_code, province_code = _plate_codes(plate)
            values = [
                str(plate.get("id", "")),
                _country_label(str(plate.get("country", "unknown"))),
                str(plate.get("vehicle_type") or "-"),
                _display_with_code(plate.get("plate_prefix"), prefix_code),
                str(plate.get("plate_number") or ocr.get("text") or "-"),
                _display_with_code(plate.get("province"), province_code),
                f"{confidence * 100:.2f}%",
            ]
            for column, value in enumerate(values):
                self.results.setItem(row, column, QTableWidgetItem(value))

    def _table_plate_clicked(self, row: int, column: int) -> None:
        del column
        self._select_plate(row)

    def _select_plate(self, index: int) -> None:
        if not 0 <= index < len(self.plates):
            return
        plate = self.plates[index]
        ocr = plate.get("ocr", {})
        self.plate_layout.set_plate(plate)
        self.country_value.setText(_country_label(str(plate.get("country", "unknown")), detailed=True))
        self.vehicle_type_value.setText(str(plate.get("vehicle_type") or "ไม่ทราบ"))
        prefix_code, province_code = _plate_codes(plate)
        self.prefix_value.setText(_display_with_code(plate.get("plate_prefix"), prefix_code))
        self.number_value.setText(str(plate.get("plate_number") or "-"))
        self.province_value.setText(_display_with_code(plate.get("province"), province_code))
        confidence = float(plate.get("recognition_confidence", ocr.get("confidence", 0.0)))
        self.confidence_value.setText(f"{confidence * 100:.2f}%")
        self.method_value.setText(str(ocr.get("method") or "-"))
        self.ocr_value.setText(str(ocr.get("text") or "-"))
        self.results.selectRow(index)

    def save_json(self) -> None:
        if not self.last_result:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "บันทึกผลการสแกน", "scan_result.json", "JSON (*.json)")
        if not filename:
            return
        Path(filename).write_text(json.dumps(self.last_result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_label.setText(f"บันทึกผลแล้ว: {filename}")

    def open_output_video(self) -> None:
        if not self.last_result:
            return
        path = Path(str(self.last_result.get("output_video", "")))
        if path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self, event: Any) -> None:
        if self.thread is not None and self.thread.isRunning():
            if isinstance(self.worker, (VideoScanWorker, CameraScanWorker)):
                self.worker.stop()
            self.thread.quit()
            if not self.thread.wait(5000):
                event.ignore()
                return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(Settings.from_env())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
