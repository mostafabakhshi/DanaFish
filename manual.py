"""
DanaFish — semi-automatic (human-in-the-loop) neuron counting.

Same models and same analysis as the fully automatic `main.py`, but the operator can
correct the model at every stage:

  1. The image is loaded and orientation-corrected automatically (as in main.py).
  2. Head / body / tail landmarks can be erased and re-placed by hand when the
     detector gets them wrong or misses them entirely; orientation can then be
     re-applied from the corrected landmarks.
  3. The ROI (spinal-cord region) is drawn manually by the operator instead of
     being taken from the detected body box.
  4. Neurons are detected inside that ROI and the spinal cord is fitted; the
     operator can then add missed neurons, delete false positives, and move or
     resize existing ones.
  5. Export produces the same annotated image and Excel metrics as main.py, plus a
     JSON sidecar recording exactly which edits were made.

`main.py` and the automatic pipeline are untouched — this is an additional entry point.

Run:  python manual.py
"""

import os
import sys
import json
import copy
import tempfile
import traceback
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, QTimer
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont,
                         QKeySequence, QCursor)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
                             QComboBox, QListWidget, QDockWidget, QToolBar, QAction,
                             QGroupBox, QDoubleSpinBox, QSizePolicy, QCheckBox,
                             QProgressDialog, QButtonGroup, QRadioButton, QDialog,
                             QDialogButtonBox, QFormLayout, QListWidgetItem)

# ── Project modules (shared with the automatic pipeline) ──────────────────────
from model import ZebraFishModel
from image_rotation_corrector import ImageRotationCorrector
from test_exact_body_region_pipeline import ExactBodyRegionAnalyzer
from config import OUTPUT_ANNOTATED_PATH

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

# Overlay colours (RGB, for Qt)
COL_HEAD = QColor(255, 140, 0)
COL_BODY = QColor(30, 144, 255)
COL_TAIL = QColor(220, 60, 200)
COL_ROI = QColor(0, 200, 255)
COL_NEURON_MODEL = QColor(0, 230, 0)
COL_NEURON_USER = QColor(255, 215, 0)
COL_SELECTED = QColor(255, 60, 60)
COL_CORD = QColor(255, 60, 60)

LANDMARK_COLORS = {'head': COL_HEAD, 'body': COL_BODY, 'tail': COL_TAIL}

HANDLE_PX = 8       # size of resize handles, in screen pixels
MIN_BOX_PX = 3      # smallest box the user can draw, in image pixels


class Mode(Enum):
    VIEW = 'view'
    LANDMARK = 'landmark'
    ROI = 'roi'
    CORD = 'cord'
    NEURON = 'neuron'
    MEASURE = 'measure'


# Edit targets, labelled with their keyboard shortcut so the panel is self-documenting.
TARGET_LABELS = {
    'head':    'Head  (H)',
    'tail':    'Tail  (T)',
    'roi':     'Region of interest  (R)',
    'cord':    'Spinal cord  (S)',
    'neurons': 'Neurons  (N)',
}

N_CORD_HANDLES = 10      # control points seeded from the automatic fit
CORD_SAMPLES = 240       # points used to render/export the smoothed curve


def smooth_curve_through(points, n_out=CORD_SAMPLES):
    """A smooth curve through the given control points, left to right.

    The spinal cord is a single-valued, gently bending line, so the control
    points are sorted by x and interpolated against x. Uses a cubic spline when
    there are enough points, quadratic/linear below that, so the curve stays
    defined however few handles remain.
    """
    pts = sorted((float(x), float(y)) for x, y in points)
    if len(pts) < 2:
        return np.array([]), np.array([])
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])

    # collapse duplicate x so interpolation stays well posed
    keep = np.concatenate(([True], np.diff(xs) > 1e-6))
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 2:
        return np.array([]), np.array([])

    x_new = np.linspace(xs[0], xs[-1], n_out)
    if len(xs) >= 4:
        try:
            from scipy.interpolate import make_interp_spline
            return x_new, make_interp_spline(xs, ys, k=3)(x_new)
        except Exception:
            pass
    if len(xs) == 3:
        return x_new, np.poly1d(np.polyfit(xs, ys, 2))(x_new)
    return x_new, np.interp(x_new, xs, ys)


def polyline_length(xs, ys):
    if xs is None or len(xs) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


# ── Physical scale ────────────────────────────────────────────────────────────
# Calibration is stored in ORIGINAL microscope pixels (µm per original pixel),
# because that is a property of the objective and camera and is therefore
# constant across a dataset. The pipeline resizes every image onto an 840x840
# canvas by scale_factor = min(840/w, 840/h), so the on-screen scale is
#     µm per displayed pixel = µm_per_original_px / scale_factor
# and must be recomputed per image — images of different original sizes get
# different scale factors.
CALIB_FILE = 'calibration.json'
UM = 'µm'
NICE_BAR_LENGTHS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500,
                    1000, 2000, 2500, 5000, 10000]


def nice_bar_length(span_um):
    """Largest 'round' scale-bar length that fits comfortably in the view."""
    target = span_um * 0.22
    candidates = [v for v in NICE_BAR_LENGTHS if v <= target]
    return candidates[-1] if candidates else NICE_BAR_LENGTHS[0]


def fmt_um(value):
    """Format a micrometre value with sensible precision."""
    if value >= 100:
        return f'{value:.0f} {UM}'
    if value >= 10:
        return f'{value:.1f} {UM}'
    return f'{value:.2f} {UM}'


class CalibrationDialog(QDialog):
    """Set µm per pixel, either directly or from a measured known distance."""

    def __init__(self, parent, um_per_orig_px, scale_factor, measured_display_px=None,
                 original_width_px=None):
        super().__init__(parent)
        self.setWindowTitle('Set image scale')
        self.scale_factor = scale_factor or 1.0
        self.original_width_px = original_width_px
        # Whether the operator pressed "Clear calibration". Kept as its own flag:
        # deriving it from the incoming value would make the very first
        # calibration (which starts from None) silently return None.
        self._cleared = False

        lay = QVBoxLayout(self)
        intro = QLabel(
            'Calibration is stored in <b>original</b> microscope pixels, so it stays valid '
            'for every image in the dataset regardless of how each one is rescaled onto '
            'the 840×840 canvas.')
        intro.setWordWrap(True)
        lay.addWidget(intro)

        form = QFormLayout()
        self.direct = QDoubleSpinBox()
        self.direct.setDecimals(5)
        self.direct.setRange(0.00001, 1000.0)
        self.direct.setSingleStep(0.01)
        self.direct.setValue(um_per_orig_px if um_per_orig_px else 1.0)
        self.direct.setSuffix(f'  {UM}/pixel (original image)')
        form.addRow('Scale:', self.direct)
        lay.addLayout(form)

        # Often the easiest number to get out of ZEN is the field-of-view width,
        # so offer that as an alternative to µm-per-pixel.
        if original_width_px:
            fov = QGroupBox('…or enter the field of view')
            fl = QFormLayout(fov)
            fl.addRow('This image is:', QLabel(f'{original_width_px} pixels wide (original)'))
            self.fov_um = QDoubleSpinBox()
            self.fov_um.setDecimals(1)
            self.fov_um.setRange(1.0, 1e6)
            self.fov_um.setValue(2000.0)
            self.fov_um.setSuffix(f'  {UM} across the full width')
            fl.addRow('Full width is:', self.fov_um)
            fbtn = QPushButton('Use this to set the scale')
            fbtn.clicked.connect(
                lambda: self.direct.setValue(self.fov_um.value() / original_width_px))
            fl.addRow(fbtn)
            lay.addWidget(fov)

        if measured_display_px:
            box = QGroupBox('From the distance you just measured')
            bl = QFormLayout(box)
            orig_px = measured_display_px / self.scale_factor
            bl.addRow('Measured:', QLabel(f'{measured_display_px:.1f} displayed px '
                                          f'= {orig_px:.1f} original px'))
            self.known = QDoubleSpinBox()
            self.known.setDecimals(2)
            self.known.setRange(0.01, 1e6)
            self.known.setValue(100.0)
            self.known.setSuffix(f'  {UM}')
            bl.addRow('That distance is:', self.known)
            btn = QPushButton('Use this to set the scale')
            def apply_known():
                self.direct.setValue(self.known.value() / orig_px)
            btn.clicked.connect(apply_known)
            bl.addRow(btn)
            lay.addWidget(box)
        else:
            hint = QLabel('Tip: press <b>M</b> and drag along a known distance (a scale bar in '
                          'the image, or a feature of known size), then reopen this dialog to '
                          'convert that measurement into a calibration.')
            hint.setWordWrap(True)
            lay.addWidget(hint)

        self.preview = QLabel()
        self.preview.setWordWrap(True)
        lay.addWidget(self.preview)
        self.direct.valueChanged.connect(self._update_preview)
        self._update_preview()

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        clear = bb.addButton('Clear calibration', QDialogButtonBox.ResetRole)
        clear.clicked.connect(self._clear)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _update_preview(self):
        v = self.direct.value()
        disp = v / self.scale_factor
        self.preview.setText(
            f'At this scale, one displayed pixel on the current image = '
            f'<b>{disp:.4f} {UM}</b> (image rescaled by {self.scale_factor:.3f}×).')

    def _clear(self):
        self._cleared = True
        self.accept()

    def value(self):
        """The chosen µm per original pixel, or None if the operator cleared it."""
        return None if self._cleared else self.direct.value()


# ═══════════════════════════════════════════════════════════════════════════════
# Canvas
# ═══════════════════════════════════════════════════════════════════════════════

class ImageCanvas(QWidget):
    """Displays the corrected image and hosts all mouse editing."""

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(600, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image = None            # BGR ndarray, the corrected 840x840 image
        self._qimg = None

        self.scale = 1.0
        self.offset = QPointF(0, 0)

        self.mode = Mode.VIEW
        self.landmark_target = 'head'      # only 'head' or 'tail' are user-editable

        # transient drag state
        self._dragging = False
        self._drag_start = None      # image coords
        self._drag_now = None
        self._pan_last = None
        self._move_kind = None       # None | 'move' | 'resize'
        self._move_corner = None
        self._orig_box = None

        # ruler / measurement
        self.show_ruler = True
        self.measure_line = None     # ((x1,y1),(x2,y2)) in image coords

    # ── image / view ──────────────────────────────────────────────────────────
    def set_image(self, bgr):
        self.image = bgr
        if bgr is None:
            self._qimg = None
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            self._qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.fit_to_window()

    def fit_to_window(self):
        if self._qimg is None:
            return
        iw, ih = self._qimg.width(), self._qimg.height()
        s = min(self.width() / iw, self.height() / ih) * 0.98
        self.scale = s
        self.offset = QPointF((self.width() - iw * s) / 2, (self.height() - ih * s) / 2)
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._qimg is not None and self.scale <= 0:
            self.fit_to_window()

    def img_to_widget(self, x, y):
        return QPointF(x * self.scale + self.offset.x(), y * self.scale + self.offset.y())

    def widget_to_img(self, pos):
        return ((pos.x() - self.offset.x()) / self.scale,
                (pos.y() - self.offset.y()) / self.scale)

    def rect_img_to_widget(self, box):
        p1 = self.img_to_widget(box[0], box[1])
        p2 = self.img_to_widget(box[2], box[3])
        return QRectF(p1, p2).normalized()

    # ── painting ──────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 28))
        if self._qimg is None:
            p.setPen(QColor(150, 150, 150))
            p.setFont(QFont('Segoe UI', 12))
            p.drawText(self.rect(), Qt.AlignCenter,
                       'Open an image or a folder to begin  (Ctrl+O / Ctrl+Shift+O)')
            return

        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 4)
        target = QRectF(self.offset,
                        QPointF(self.offset.x() + self._qimg.width() * self.scale,
                                self.offset.y() + self._qimg.height() * self.scale))
        p.drawImage(target, self._qimg)

        st = self.win.state

        # spinal cord curve
        if st.cord_xy is not None and len(st.cord_xy[0]) > 1:
            p.setPen(QPen(COL_CORD, 3 if self.mode == Mode.CORD else 2))
            xs, ys = st.cord_xy
            prev = self.img_to_widget(xs[0], ys[0])
            for i in range(1, len(xs)):
                cur = self.img_to_widget(xs[i], ys[i])
                p.drawLine(prev, cur)
                prev = cur

        # spinal cord control points
        if self.mode == Mode.CORD:
            for i, (cx, cy) in enumerate(st.cord_points):
                c = self.img_to_widget(cx, cy)
                sel = (i == st.cord_selected)
                p.setBrush(QBrush(COL_SELECTED if sel else QColor(255, 255, 255)))
                p.setPen(QPen(QColor(20, 20, 20), 1))
                rr = HANDLE_PX + (3 if sel else 0)
                p.drawEllipse(QRectF(c.x() - rr / 2, c.y() - rr / 2, rr, rr))
            p.setBrush(Qt.NoBrush)

        # landmarks — 'body' is kept in the data (orientation, ROI seeding) but
        # never drawn: it nearly coincides with the ROI box and only adds clutter.
        for name, lm in st.landmarks.items():
            if not lm or name == 'body':
                continue
            r = self.rect_img_to_widget(lm['bbox'])
            col = LANDMARK_COLORS[name]
            dashed = self.mode == Mode.LANDMARK and name == self.landmark_target
            pen = QPen(col, 3 if dashed else 2)
            if dashed:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawRect(r)
            p.setFont(QFont('Segoe UI', 9, QFont.Bold))
            p.drawText(r.topLeft() + QPointF(3, -4), name)

        # ROI
        if st.roi is not None:
            r = self.rect_img_to_widget(st.roi)
            p.setPen(QPen(COL_ROI, 2, Qt.DashLine))
            p.drawRect(r)
            # Labelled at the bottom-left: the ROI is often seeded from the body box,
            # and a top-left label would sit on top of the 'body' landmark label.
            p.setFont(QFont('Segoe UI', 9, QFont.Bold))
            self._draw_tag(p, r.bottomLeft() + QPointF(4, -5),
                           'ROI  ' + self.win.roi_size_text())
            if self.mode == Mode.ROI:
                p.setBrush(QBrush(COL_ROI))
                p.setPen(Qt.NoPen)
                for cx, cy in self._corners(st.roi):
                    c = self.img_to_widget(cx, cy)
                    p.drawRect(QRectF(c.x() - HANDLE_PX / 2, c.y() - HANDLE_PX / 2,
                                      HANDLE_PX, HANDLE_PX))
                p.setBrush(Qt.NoBrush)

        # neurons
        for i, n in enumerate(st.neurons):
            r = self.rect_img_to_widget(n['box'])
            selected = (i == st.selected)
            col = COL_SELECTED if selected else (
                COL_NEURON_USER if n['source'] == 'user' else COL_NEURON_MODEL)
            p.setPen(QPen(col, 2 if selected else 1))
            p.drawRect(r)
            if selected and self.mode == Mode.NEURON:
                p.setBrush(QBrush(col))
                p.setPen(Qt.NoPen)
                for cx, cy in self._corners(n['box']):
                    c = self.img_to_widget(cx, cy)
                    p.drawRect(QRectF(c.x() - HANDLE_PX / 2, c.y() - HANDLE_PX / 2,
                                      HANDLE_PX, HANDLE_PX))
                p.setBrush(Qt.NoBrush)

        # rubber band while drawing
        if self._dragging and self._drag_start and self._drag_now and self._move_kind is None:
            box = self._norm(self._drag_start, self._drag_now)
            r = self.rect_img_to_widget(box)
            col = {Mode.ROI: COL_ROI, Mode.NEURON: COL_NEURON_USER}.get(
                self.mode, LANDMARK_COLORS.get(self.landmark_target, COL_ROI))
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.drawRect(r)

        # measurement line
        if self.measure_line is not None:
            (mx1, my1), (mx2, my2) = self.measure_line
            a = self.img_to_widget(mx1, my1)
            b = self.img_to_widget(mx2, my2)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(a, b)
            for endp in (a, b):          # end caps
                p.drawLine(QPointF(endp.x() - 4, endp.y() - 4), QPointF(endp.x() + 4, endp.y() + 4))
                p.drawLine(QPointF(endp.x() - 4, endp.y() + 4), QPointF(endp.x() + 4, endp.y() - 4))
            dist_px = float(np.hypot(mx2 - mx1, my2 - my1))
            label = self.win.length_text(dist_px)
            mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
            self._draw_tag(p, mid + QPointF(8, -8), label)

        self._draw_ruler(p)
        self._draw_scale_bar(p)

        # mode banner
        p.setPen(QColor(235, 235, 235))
        p.setFont(QFont('Consolas', 10, QFont.Bold))
        p.drawText(10, 20, f'MODE: {self.mode.value.upper()}' +
                   (f'  →  {self.landmark_target}' if self.mode == Mode.LANDMARK else ''))

    # ── ruler / scale bar ─────────────────────────────────────────────────────
    @staticmethod
    def _draw_tag(p, pos, text):
        """White text on a dark plate, so labels stay readable over the image."""
        p.setFont(QFont('Segoe UI', 9, QFont.Bold))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 8
        h = fm.height() + 4
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 170)))
        p.drawRect(QRectF(pos.x() - 4, pos.y() - h + 4, w, h))
        p.setBrush(Qt.NoBrush)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QPointF(pos.x(), pos.y()), text)

    def _draw_ruler(self, p):
        """Tick marks along the top and left edges, labelled in µm when calibrated."""
        if not self.show_ruler or self._qimg is None:
            return
        upp = self.win.um_per_display_px()          # None when uncalibrated
        iw, ih = self._qimg.width(), self._qimg.height()

        # choose a tick spacing in image pixels that is ~60 screen px apart
        target_img_px = 60 / self.scale
        if upp:
            step_um = nice_bar_length(target_img_px * upp / 0.22)
            step_img = step_um / upp
        else:
            nice_px = [5, 10, 25, 50, 100, 200, 250, 500, 1000]
            step_img = next((v for v in nice_px if v >= target_img_px), nice_px[-1])
            step_um = None

        p.setPen(QPen(QColor(200, 200, 210, 200), 1))
        p.setFont(QFont('Segoe UI', 7))

        n = 0
        x = 0.0
        while x <= iw and n < 400:
            sx = self.img_to_widget(x, 0).x()
            if 0 <= sx <= self.width():
                p.drawLine(QPointF(sx, 0), QPointF(sx, 8))
                txt = fmt_um(x * upp) if upp else f'{int(x)}px'
                p.drawText(QPointF(sx + 2, 18), txt)
            x += step_img
            n += 1

        n = 0
        y = 0.0
        while y <= ih and n < 400:
            sy = self.img_to_widget(0, y).y()
            if 0 <= sy <= self.height():
                p.drawLine(QPointF(0, sy), QPointF(8, sy))
                txt = fmt_um(y * upp) if upp else f'{int(y)}px'
                p.drawText(QPointF(10, sy - 2), txt)
            y += step_img
            n += 1

    def _draw_scale_bar(self, p):
        """Microscopy-style scale bar, bottom right."""
        upp = self.win.um_per_display_px()
        if not upp or self._qimg is None:
            return
        span_um = (self.width() / self.scale) * upp
        bar_um = nice_bar_length(span_um)
        bar_px = (bar_um / upp) * self.scale
        if bar_px < 20:
            return

        x2 = self.width() - 24
        x1 = x2 - bar_px
        y = self.height() - 30

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 150)))
        p.drawRect(QRectF(x1 - 10, y - 26, bar_px + 20, 42))
        p.setBrush(Qt.NoBrush)

        p.setPen(QPen(QColor(255, 255, 255), 4))
        p.drawLine(QPointF(x1, y), QPointF(x2, y))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(QPointF(x1, y - 5), QPointF(x1, y + 5))
        p.drawLine(QPointF(x2, y - 5), QPointF(x2, y + 5))

        p.setFont(QFont('Segoe UI', 10, QFont.Bold))
        fm = p.fontMetrics()
        label = f'{bar_um:g} {UM}'
        p.drawText(QPointF((x1 + x2) / 2 - fm.horizontalAdvance(label) / 2, y - 8), label)

    @staticmethod
    def _corners(box):
        x1, y1, x2, y2 = box
        return [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    @staticmethod
    def _norm(a, b):
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]

    # ── hit testing ───────────────────────────────────────────────────────────
    def _corner_at(self, box, ix, iy):
        tol = HANDLE_PX / self.scale
        for idx, (cx, cy) in enumerate(self._corners(box)):
            if abs(ix - cx) <= tol and abs(iy - cy) <= tol:
                return idx
        return None

    def _cord_point_at(self, ix, iy):
        tol = (HANDLE_PX + 4) / self.scale
        best, best_d = None, None
        for i, (cx, cy) in enumerate(self.win.state.cord_points):
            d = np.hypot(ix - cx, iy - cy)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = i, d
        return best

    def _neuron_at(self, ix, iy):
        """Topmost neuron containing the point; smallest wins so overlaps stay reachable."""
        hits = []
        for i, n in enumerate(self.win.state.neurons):
            x1, y1, x2, y2 = n['box']
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                hits.append((abs((x2 - x1) * (y2 - y1)), i))
        return min(hits)[1] if hits else None

    # ── mouse ─────────────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if self._qimg is None:
            return
        ix, iy = self.widget_to_img(e.pos())
        st = self.win.state

        # middle button (or space-less right in VIEW) pans
        if e.button() == Qt.MiddleButton:
            self._pan_last = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self.mode == Mode.NEURON:
            if e.button() == Qt.RightButton:
                i = self._neuron_at(ix, iy)
                if i is not None:
                    self.win.push_undo()
                    removed = st.neurons.pop(i)
                    st.selected = None
                    st.edits['removed'].append(
                        {'box': removed['box'], 'source': removed['source']})
                    self.win.after_edit(f'Deleted neuron ({removed["source"]})')
                return

            if e.button() == Qt.LeftButton:
                # resize handle of the selected neuron?
                if st.selected is not None:
                    c = self._corner_at(st.neurons[st.selected]['box'], ix, iy)
                    if c is not None:
                        self.win.push_undo()
                        self._dragging = True
                        self._move_kind = 'resize'
                        self._move_corner = c
                        self._orig_box = list(st.neurons[st.selected]['box'])
                        return
                i = self._neuron_at(ix, iy)
                if i is not None:
                    st.selected = i
                    self.win.push_undo()
                    self._dragging = True
                    self._move_kind = 'move'
                    self._drag_start = (ix, iy)
                    self._orig_box = list(st.neurons[i]['box'])
                    self.win.refresh_list()
                    self.update()
                    return
                # empty space → draw a new neuron
                st.selected = None
                self._dragging = True
                self._move_kind = None
                self._drag_start = (ix, iy)
                self._drag_now = (ix, iy)
                self.update()
                return

        if self.mode == Mode.CORD:
            i = self._cord_point_at(ix, iy)
            if e.button() == Qt.RightButton:
                if i is not None and len(st.cord_points) > 2:
                    self.win.push_undo()
                    st.cord_points.pop(i)
                    st.cord_selected = None
                    st.cord_edited = True
                    st.rebuild_cord()
                    self.win.after_edit(f'Removed cord point ({len(st.cord_points)} left)')
                elif i is not None:
                    self.win.statusBar().showMessage(
                        'A curve needs at least two points — add one before removing this.')
                return
            if e.button() == Qt.LeftButton:
                self.win.push_undo()
                if i is None:
                    # insert a new control point at the click
                    st.cord_points.append((ix, iy))
                    st.cord_points.sort(key=lambda pt: pt[0])
                    i = st.cord_points.index((ix, iy))
                    st.cord_edited = True
                    st.rebuild_cord()
                st.cord_selected = i
                self._dragging = True
                self._move_kind = 'cord'
                self.win.after_edit(f'Cord: {len(st.cord_points)} points, '
                                    f'{self.win.length_text(st.spinal_length)}')
                return

        if self.mode == Mode.MEASURE and e.button() == Qt.LeftButton:
            self._dragging = True
            self._move_kind = None
            self._drag_start = (ix, iy)
            self._drag_now = (ix, iy)
            self.measure_line = ((ix, iy), (ix, iy))
            self.update()
            return

        if self.mode == Mode.ROI and e.button() == Qt.LeftButton:
            if st.roi is not None:
                c = self._corner_at(st.roi, ix, iy)
                if c is not None:
                    self.win.push_undo()
                    self._dragging = True
                    self._move_kind = 'resize'
                    self._move_corner = c
                    self._orig_box = list(st.roi)
                    return
            self._dragging = True
            self._move_kind = None
            self._drag_start = (ix, iy)
            self._drag_now = (ix, iy)
            return

        if self.mode == Mode.LANDMARK and e.button() == Qt.LeftButton:
            self._dragging = True
            self._move_kind = None
            self._drag_start = (ix, iy)
            self._drag_now = (ix, iy)
            return

        if e.button() == Qt.LeftButton:      # VIEW mode → pan
            self._pan_last = e.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._qimg is None:
            return
        ix, iy = self.widget_to_img(e.pos())
        st = self.win.state

        if self._pan_last is not None:
            d = e.pos() - self._pan_last
            self.offset += QPointF(d.x(), d.y())
            self._pan_last = e.pos()
            self.update()
            return

        if self._dragging:
            if self._move_kind == 'cord' and st.cord_selected is not None:
                st.cord_points[st.cord_selected] = (ix, iy)
                st.cord_edited = True
                st.rebuild_cord()
                self.update()
                self.win.show_coords(ix, iy)
                return
            if self._move_kind == 'move' and st.selected is not None:
                dx = ix - self._drag_start[0]
                dy = iy - self._drag_start[1]
                b = self._orig_box
                st.neurons[st.selected]['box'] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
            elif self._move_kind == 'resize':
                b = list(self._orig_box)
                if self._move_corner in (0, 2):
                    b[0] = ix
                else:
                    b[2] = ix
                if self._move_corner in (0, 1):
                    b[1] = iy
                else:
                    b[3] = iy
                b = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]
                if self.mode == Mode.ROI:
                    st.roi = b
                elif st.selected is not None:
                    st.neurons[st.selected]['box'] = b
            elif self.mode == Mode.MEASURE:
                self._drag_now = (ix, iy)
                self.measure_line = (self._drag_start, (ix, iy))
            else:
                self._drag_now = (ix, iy)
            self.update()

        self.win.show_coords(ix, iy)

    def mouseReleaseEvent(self, e):
        if self._pan_last is not None:
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor)
            return
        if not self._dragging:
            return

        st = self.win.state
        finished_draw = self._move_kind is None
        self._dragging = False

        if self._move_kind == 'cord':
            self._move_kind = None
            # keep handles ordered so the curve stays single valued
            sel = st.cord_points[st.cord_selected] if st.cord_selected is not None else None
            st.cord_points.sort(key=lambda pt: pt[0])
            if sel is not None and sel in st.cord_points:
                st.cord_selected = st.cord_points.index(sel)
            st.rebuild_cord()
            self.win.after_edit(f'Cord edited — {len(st.cord_points)} points, '
                                f'{self.win.length_text(st.spinal_length)}')
            return

        if not finished_draw:
            self._move_kind = None
            self._orig_box = None
            self.win.after_edit('Adjusted box')
            return

        if self._drag_start is None or self._drag_now is None:
            return

        if self.mode == Mode.MEASURE:
            (a, b) = self.measure_line
            dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            self._drag_start = self._drag_now = None
            self.win.last_measure_px = dist
            self.win.after_edit(
                f'Measured {self.win.length_text(dist)}' +
                ('' if self.win.um_per_orig_px else
                 '  —  set the scale (Ctrl+K) to read this in µm'))
            return

        box = self._norm(self._drag_start, self._drag_now)
        self._drag_start = self._drag_now = None

        if (box[2] - box[0]) < MIN_BOX_PX or (box[3] - box[1]) < MIN_BOX_PX:
            self.update()
            return

        box = self._clip(box)

        if self.mode == Mode.ROI:
            self.win.push_undo()
            st.roi = box
            self.win.after_edit('ROI set — now click "Detect in ROI"')
            self.win.offer_calibration_once()
        elif self.mode == Mode.LANDMARK:
            self.win.push_undo()
            name = self.landmark_target
            st.landmarks[name] = {
                'bbox': box,
                'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                'confidence': 1.0,
            }
            st.edits['landmarks_edited'].append(name)
            self.win.after_edit(f'Placed {name} landmark by hand')
        elif self.mode == Mode.NEURON:
            self.win.push_undo()
            st.neurons.append({'box': box, 'conf': 1.0, 'source': 'user'})
            st.selected = len(st.neurons) - 1
            st.edits['added'].append({'box': box})
            self.win.after_edit('Added neuron by hand')

    def _clip(self, box):
        if self.image is None:
            return box
        h, w = self.image.shape[:2]
        return [max(0, min(box[0], w - 1)), max(0, min(box[1], h - 1)),
                max(0, min(box[2], w - 1)), max(0, min(box[3], h - 1))]

    def wheelEvent(self, e):
        if self._qimg is None:
            return
        before = self.widget_to_img(e.pos())
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale = max(0.1, min(40.0, self.scale * factor))
        after = self.widget_to_img(e.pos())
        self.offset += QPointF((after[0] - before[0]) * self.scale,
                               (after[1] - before[1]) * self.scale)
        self.update()

    def keyPressEvent(self, e):
        st = self.win.state
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.mode == Mode.NEURON and st.selected is not None:
                self.win.push_undo()
                removed = st.neurons.pop(st.selected)
                st.selected = None
                st.edits['removed'].append(
                    {'box': removed['box'], 'source': removed['source']})
                self.win.after_edit('Deleted neuron')
            elif self.mode == Mode.LANDMARK:
                name = self.landmark_target
                if st.landmarks.get(name):
                    self.win.push_undo()
                    st.landmarks[name] = None
                    st.edits['landmarks_edited'].append(f'{name}:erased')
                    self.win.after_edit(f'Erased {name} landmark')
            elif self.mode == Mode.ROI and st.roi is not None:
                self.win.push_undo()
                st.roi = None
                self.win.after_edit('Cleared ROI')
            elif self.mode == Mode.CORD and st.cord_selected is not None:
                if len(st.cord_points) > 2:
                    self.win.push_undo()
                    st.cord_points.pop(st.cord_selected)
                    st.cord_selected = None
                    st.cord_edited = True
                    st.rebuild_cord()
                    self.win.after_edit(f'Removed cord point ({len(st.cord_points)} left)')
        else:
            super().keyPressEvent(e)


# ═══════════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════════

class State:
    def __init__(self):
        self.image_path = None
        self.corrected = None          # BGR 840x840
        self.info = {}                 # correction_info from ImageRotationCorrector
        self.landmarks = {'head': None, 'body': None, 'tail': None}
        self.roi = None                # [x1,y1,x2,y2]
        self.neurons = []              # [{'box':[...], 'conf':float, 'source':'model'|'user'}]
        self.selected = None
        self.cord_xy = None            # (xs, ys) smoothed curve, for display and export
        self.cord_points = []          # [(x, y)] draggable control points
        self.cord_edited = False       # True once the operator has touched the cord
        self.cord_selected = None
        self.spinal_length = 0.0
        self.edits = {'added': [], 'removed': [], 'landmarks_edited': [],
                      'roi_manual': False, 'reoriented': False}

    def snapshot(self):
        return copy.deepcopy((self.landmarks, self.roi, self.neurons, self.selected,
                              self.cord_xy, self.spinal_length, self.cord_points,
                              self.cord_edited, self.cord_selected))

    def restore(self, snap):
        (self.landmarks, self.roi, self.neurons, self.selected,
         self.cord_xy, self.spinal_length, self.cord_points,
         self.cord_edited, self.cord_selected) = copy.deepcopy(snap)

    def rebuild_cord(self):
        """Recompute the smoothed curve from the current control points."""
        xs, ys = smooth_curve_through(self.cord_points)
        if len(xs) > 1:
            self.cord_xy = (xs, ys)
            self.spinal_length = polyline_length(xs, ys)
        else:
            self.cord_xy = None
            self.spinal_length = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DanaFish — Semi-Automatic Neuron Counting')
        self.resize(1450, 950)

        self.state = State()
        self.undo_stack = []
        self.image_list = []
        self.image_index = -1
        self.output_dir = None

        # physical scale, in µm per ORIGINAL microscope pixel (None = uncalibrated)
        self.um_per_orig_px = None
        self.last_measure_px = None
        self._calibration_offered = False
        self._last_edit_mode = Mode.NEURON
        self._load_calibration()

        # models are loaded lazily so the window appears immediately
        self._rotator = None
        self._detector = None
        self._analyzer = None

        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)

        self._build_toolbar()
        self._build_side_panel()
        self._update_scale_label()
        self.statusBar().showMessage('Ready — open an image or a folder to begin.')
        self._set_mode(self._last_edit_mode)

    # ── physical scale ────────────────────────────────────────────────────────
    def _load_calibration(self):
        try:
            if os.path.exists(CALIB_FILE):
                with open(CALIB_FILE, encoding='utf-8') as fh:
                    self.um_per_orig_px = json.load(fh).get('um_per_original_px')
        except Exception:
            self.um_per_orig_px = None

    def _save_calibration(self):
        try:
            with open(CALIB_FILE, 'w', encoding='utf-8') as fh:
                json.dump({'um_per_original_px': self.um_per_orig_px,
                           'note': 'Micrometres per pixel of the ORIGINAL microscope image, '
                                   'before the pipeline rescales it onto the 840x840 canvas.'},
                          fh, indent=2)
        except Exception as exc:
            self.statusBar().showMessage(f'Could not save calibration: {exc}')

    def scale_factor(self):
        """How much the current image was shrunk onto the 840x840 canvas."""
        try:
            sf = float(self.state.info.get('scale_factor', 1.0))
            return sf if sf > 0 else 1.0
        except Exception:
            return 1.0

    def um_per_display_px(self):
        """µm per displayed pixel, or None when uncalibrated or showing pixels.

        Returns None whenever the operator has chosen pixel units, so every
        readout follows the unit selector from one place.
        """
        if hasattr(self, 'unit_um') and not self.unit_um.isChecked():
            return None
        return self.um_per_display_px_raw()

    def um_per_display_px_raw(self):
        """µm per displayed pixel whenever a calibration exists, ignoring the unit
        selector. Export uses this so choosing pixel *display* never discards the
        physical measurements from the saved results."""
        if not self.um_per_orig_px:
            return None
        return self.um_per_orig_px / self.scale_factor()

    def _unit_changed(self):
        self._update_scale_label()
        self.canvas.update()
        self.statusBar().showMessage(
            f'Showing lengths in {"micrometres" if self.unit_um.isChecked() else "pixels"}.')

    def length_text(self, px):
        """Format a length given in displayed pixels."""
        upp = self.um_per_display_px()
        return fmt_um(px * upp) if upp else f'{px:.0f} px'

    def roi_size_text(self):
        st = self.state
        if st.roi is None:
            return ''
        w = st.roi[2] - st.roi[0]
        h = st.roi[3] - st.roi[1]
        upp = self.um_per_display_px()
        if upp:
            return f'{fmt_um(w * upp)} × {fmt_um(h * upp)}'
        if self.um_per_orig_px:          # calibrated, but pixels chosen
            return f'{w:.0f} × {h:.0f} px'
        # Say why it is in pixels, rather than leaving the operator to wonder.
        return f'{w:.0f} × {h:.0f} px  —  Ctrl+K for {UM}'

    def original_width_px(self):
        """Width of the current image before the pipeline rescaled it."""
        try:
            size = self.state.info.get('original_size')   # (height, width)
            return int(size[1]) if size else None
        except Exception:
            return None

    def set_scale(self):
        dlg = CalibrationDialog(self, self.um_per_orig_px, self.scale_factor(),
                                self.last_measure_px, self.original_width_px())
        if dlg.exec_() != QDialog.Accepted:
            return
        self.um_per_orig_px = dlg.value()
        self._save_calibration()
        # Just having set a scale, switch to µm — but leave the choice reversible.
        if self.um_per_orig_px:
            self.unit_um.setEnabled(True)
            self.unit_um.setChecked(True)
        self._update_scale_label()
        self.canvas.update()
        if self.um_per_orig_px:
            self.statusBar().showMessage(
                f'Scale set: {self.um_per_orig_px:.5f} {UM}/original px  '
                f'({self.um_per_display_px_raw():.4f} {UM} per displayed px on this image).')
        else:
            self.statusBar().showMessage('Calibration cleared — lengths shown in pixels.')

    def offer_calibration_once(self):
        """First ROI of the session with no scale set: say so, and offer to fix it.

        Without this the ROI simply reads in pixels and it is not obvious that a
        calibration is missing rather than broken.
        """
        if self.um_per_orig_px or self._calibration_offered:
            return
        self._calibration_offered = True
        ans = QMessageBox.question(
            self, 'No scale set yet',
            f'Measurements are being shown in <b>pixels</b> because no {UM} scale has been set.<br><br>'
            f'The scale cannot be read from the image — these JPEGs carry no resolution '
            f'metadata — so it has to be entered once, from your objective and camera.<br><br>'
            f'Set it now?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ans == QMessageBox.Yes:
            self.set_scale()

    def toggle_ruler(self, checked):
        self.canvas.show_ruler = checked
        self.canvas.update()

    def _update_scale_label(self):
        calibrated = bool(self.um_per_orig_px)
        self.unit_um.setEnabled(calibrated)
        if not calibrated and self.unit_um.isChecked():
            self.unit_px.setChecked(True)
        if calibrated:
            raw = self.um_per_display_px_raw()
            self.unit_um.setToolTip('')
            self.scale_label.setText(
                f'{self.um_per_orig_px:.5f} {UM}/px original\n'
                f'{raw:.4f} {UM}/px displayed  (×{self.scale_factor():.3f})')
        else:
            self.unit_um.setToolTip('Set a scale first (Ctrl+K)')
            self.scale_label.setText('Not calibrated.\nSet a scale to enable µm.')
        self._update_measure_label()

    def _update_measure_label(self):
        st = self.state
        roi_txt = self.roi_size_text() or '—'
        cord = self.length_text(st.spinal_length) if st.spinal_length else '—'
        if st.cord_edited:
            cord += '  (edited)'
        self.measure_label.setText(f'ROI: {roi_txt}\nSpinal cord: {cord}')

    # ── lazy model loading ────────────────────────────────────────────────────
    def _models(self):
        if self._rotator is None:
            dlg = QProgressDialog('Loading models (first use only)…', None, 0, 0, self)
            dlg.setWindowTitle('DanaFish')
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.setCancelButton(None)
            dlg.show()
            QApplication.processEvents()
            try:
                self._rotator = ImageRotationCorrector()
                self._detector = ZebraFishModel()
                self._analyzer = ExactBodyRegionAnalyzer()
            finally:
                dlg.close()
        return self._rotator, self._detector, self._analyzer

    # ── UI construction ───────────────────────────────────────────────────────
    def _act(self, text, slot, shortcut=None, tip=None):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setToolTip(tip or text)
        return a

    def _build_toolbar(self):
        tb = QToolBar('Main')
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        tb.addAction(self._act('Open Image', self.open_image, 'Ctrl+O'))
        tb.addAction(self._act('Open Folder', self.open_folder, 'Ctrl+Shift+O'))
        tb.addSeparator()
        tb.addAction(self._act('◀ Prev', self.prev_image, 'PgUp'))
        tb.addAction(self._act('Next ▶', self.next_image, 'PgDown'))
        tb.addSeparator()
        tb.addAction(self._act('Re-apply orientation', self.reapply_orientation, 'Ctrl+R',
                               'Recompute the flips from the current (possibly hand-corrected) landmarks'))
        tb.addAction(self._act('Detect in ROI', self.detect_in_roi, 'Ctrl+D',
                               'Run neuron detection and fit the spinal cord inside the drawn ROI'))
        tb.addAction(self._act('Refit cord', self.reset_cord, 'Ctrl+Shift+R',
                               'Discard hand edits to the spinal cord and refit it from the image'))
        tb.addSeparator()
        tb.addAction(self._act('Set scale (µm)', self.set_scale, 'Ctrl+K',
                               'Calibrate µm per pixel so lengths are reported in micrometres'))
        self.measure_action = QAction('Measure', self)
        self.measure_action.setCheckable(True)
        self.measure_action.setShortcut(QKeySequence('M'))
        self.measure_action.setToolTip('Drag to measure any distance on the image')
        self.measure_action.toggled.connect(
            lambda on: self._set_mode(Mode.MEASURE if on else Mode.VIEW))
        tb.addAction(self.measure_action)
        ruler_act = QAction('Ruler', self)
        ruler_act.setCheckable(True)
        ruler_act.setChecked(True)
        ruler_act.setToolTip('Show ruler ticks along the top and left edges')
        ruler_act.toggled.connect(self.toggle_ruler)
        tb.addAction(ruler_act)
        tb.addSeparator()
        tb.addAction(self._act('Undo', self.undo, 'Ctrl+Z'))
        tb.addAction(self._act('Fit view', self.canvas.fit_to_window, 'Ctrl+0'))
        tb.addSeparator()
        tb.addAction(self._act('Export', self.export_current, 'Ctrl+S',
                               'Write annotated image, Excel metrics and edit log'))
        tb.addAction(self._act('Help', self.show_help, 'F1'))

    def _build_side_panel(self):
        dock = QDockWidget('Workflow', self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        panel = QWidget()
        lay = QVBoxLayout(panel)

        # ── View / Edit ───────────────────────────────────────────────────────
        gb = QGroupBox('Mode')
        gl = QVBoxLayout(gb)
        self.mode_group = QButtonGroup(self)
        self.rb_view = QRadioButton('View  (V)')
        self.rb_view.setToolTip('Look around: drag to pan, wheel to zoom. Nothing can be changed.')
        self.rb_edit = QRadioButton('Edit  (E)')
        self.rb_edit.setToolTip('Correct the result. Choose what to edit below.')
        self.rb_edit.setChecked(True)      # Edit is the default: correcting is the job
        for rb in (self.rb_view, self.rb_edit):
            self.mode_group.addButton(rb)
            gl.addWidget(rb)
        self.rb_view.clicked.connect(lambda: self._set_mode(Mode.VIEW))
        self.rb_edit.clicked.connect(lambda: self._set_mode(self._last_edit_mode))
        lay.addWidget(gb)

        # ── what to edit ──────────────────────────────────────────────────────
        # 'body' is deliberately absent: it is only used internally to work out
        # the orientation and to seed the ROI, and showing it alongside the ROI
        # box just gives the operator two near-identical rectangles to worry about.
        self.edit_box = QGroupBox('Edit')
        g2 = QVBoxLayout(self.edit_box)
        self.target_group = QButtonGroup(self)
        self.target_buttons = {}
        for key, tip in [
            ('head', 'Drag to place the head; Delete erases it'),
            ('tail', 'Drag to place the tail; Delete erases it'),
            ('roi', 'Drag to draw the region; drag a corner to adjust'),
            ('cord', 'Drag a point to bend the curve · click to insert · right-click to delete'),
            ('neurons', 'Drag empty space to add · right-click to delete · '
                        'drag a box to move · drag a corner to resize'),
        ]:
            rb = QRadioButton(TARGET_LABELS[key])
            rb.setToolTip(tip)
            rb.clicked.connect(lambda _, k=key: self._set_edit_target(k))
            self.target_group.addButton(rb)
            self.target_buttons[key] = rb
            g2.addWidget(rb)
        self.target_buttons['neurons'].setChecked(True)
        self.landmark_status = QLabel('')
        self.landmark_status.setWordWrap(True)
        g2.addWidget(self.landmark_status)
        self.edit_box.setEnabled(False)
        lay.addWidget(self.edit_box)

        # ── detection settings ────────────────────────────────────────────────
        gb3 = QGroupBox('Detection')
        g3 = QVBoxLayout(gb3)
        row = QHBoxLayout()
        row.addWidget(QLabel('Confidence'))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.30)
        self.conf_spin.setToolTip('Lower finds more neurons (and more false positives). '
                                  'The automatic pipeline uses 0.30.')
        row.addWidget(self.conf_spin)
        g3.addLayout(row)
        self.keep_user_cb = QCheckBox('Keep my edits when re-detecting')
        self.keep_user_cb.setChecked(True)
        self.keep_user_cb.setToolTip('Hand-placed neurons survive a re-run of the detector')
        g3.addWidget(self.keep_user_cb)
        btn = QPushButton('Detect in ROI  (Ctrl+D)')
        btn.clicked.connect(self.detect_in_roi)
        g3.addWidget(btn)
        lay.addWidget(gb3)

        # ── physical scale ────────────────────────────────────────────────────
        gbs = QGroupBox('Units and scale')
        gs = QVBoxLayout(gbs)

        # Explicit unit choice. Pixels is the default; µm only becomes selectable
        # once a calibration exists, so the interface never implies a physical
        # measurement it cannot actually make.
        self.unit_group = QButtonGroup(self)
        self.unit_px = QRadioButton('Pixels')
        self.unit_um = QRadioButton(f'Micrometres ({UM})')
        self.unit_px.setChecked(True)
        for rb in (self.unit_px, self.unit_um):
            self.unit_group.addButton(rb)
            rb.clicked.connect(self._unit_changed)
            gs.addWidget(rb)
        self.unit_um.setEnabled(False)
        self.unit_um.setToolTip('Set a scale first (Ctrl+K)')

        self.scale_label = QLabel()
        self.scale_label.setWordWrap(True)
        gs.addWidget(self.scale_label)
        sbtn = QPushButton('Set scale…  (Ctrl+K)')
        sbtn.clicked.connect(self.set_scale)
        gs.addWidget(sbtn)
        self.measure_label = QLabel('ROI: —\nSpinal cord: —')
        self.measure_label.setWordWrap(True)
        self.measure_label.setFont(QFont('Segoe UI', 10, QFont.Bold))
        gs.addWidget(self.measure_label)
        lay.addWidget(gbs)

        # ── neuron list ───────────────────────────────────────────────────────
        gb4 = QGroupBox('Neurons')
        g4 = QVBoxLayout(gb4)
        self.count_label = QLabel('Count: 0')
        f = QFont('Segoe UI', 13, QFont.Bold)
        self.count_label.setFont(f)
        g4.addWidget(self.count_label)
        self.neuron_list = QListWidget()
        self.neuron_list.currentRowChanged.connect(self._select_from_list)
        g4.addWidget(self.neuron_list)

        row2 = QHBoxLayout()
        self.check_all_cb = QCheckBox('Select all')
        self.check_all_cb.setToolTip('Tick every neuron, then remove them together')
        self.check_all_cb.clicked.connect(self._check_all)
        row2.addWidget(self.check_all_cb)
        del_btn = QPushButton('Remove ticked')
        del_btn.setToolTip('Delete every ticked neuron')
        del_btn.clicked.connect(self.delete_checked)
        row2.addWidget(del_btn)
        g4.addLayout(row2)

        clr = QPushButton('Clear all neurons')
        clr.setToolTip('Remove every neuron and start counting by hand')
        clr.clicked.connect(self.clear_all_neurons)
        g4.addWidget(clr)
        lay.addWidget(gb4, 1)

        exp = QPushButton('Export results  (Ctrl+S)')
        exp.clicked.connect(self.export_current)
        lay.addWidget(exp)

        dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setMinimumWidth(310)

    # ── mode handling ─────────────────────────────────────────────────────────
    # The interface exposes two modes, View and Edit, with Edit fanning out into
    # what is being edited. Internally these remain distinct canvas modes.
    TARGET_MODE = {'head': Mode.LANDMARK, 'tail': Mode.LANDMARK, 'roi': Mode.ROI,
                   'cord': Mode.CORD, 'neurons': Mode.NEURON}

    def _set_edit_target(self, key):
        self._last_edit_mode = self.TARGET_MODE[key]
        if key in ('head', 'tail'):
            self.canvas.landmark_target = key
        self.rb_edit.setChecked(True)
        self._set_mode(self._last_edit_mode)

    def _sync_mode_buttons(self, mode):
        """Reflect an internally chosen mode back onto the two-level selector."""
        editing = mode not in (Mode.VIEW, Mode.MEASURE)
        self.rb_edit.setChecked(editing)
        self.rb_view.setChecked(not editing)
        self.edit_box.setEnabled(editing)
        if not editing:
            return
        if mode == Mode.LANDMARK:
            key = self.canvas.landmark_target if self.canvas.landmark_target in ('head', 'tail') else 'head'
        else:
            key = {Mode.ROI: 'roi', Mode.CORD: 'cord', Mode.NEURON: 'neurons'}[mode]
        self.target_buttons[key].setChecked(True)

    def _set_mode(self, mode):
        self.canvas.mode = mode
        if mode not in (Mode.VIEW, Mode.MEASURE):
            self._last_edit_mode = mode
        self._sync_mode_buttons(mode)
        if hasattr(self, 'measure_action'):
            self.measure_action.blockSignals(True)
            self.measure_action.setChecked(mode == Mode.MEASURE)
            self.measure_action.blockSignals(False)
        hints = {
            Mode.VIEW: 'View — drag to pan, wheel to zoom.',
            Mode.LANDMARK: 'Landmark — drag to place the selected landmark; Delete erases it. '
                           'Then "Re-apply orientation" if head/tail moved.',
            Mode.ROI: 'ROI — drag to draw the spinal-cord region; drag a corner to adjust; Delete clears.',
            Mode.CORD: 'Spinal cord — drag a white point to bend the curve · click empty space to '
                       'insert a point · right-click (or Delete) to remove one. '
                       '"Refit cord" restores the automatic fit.',
            Mode.NEURON: 'Neurons — drag empty space to add · right-click to delete · '
                         'drag a box to move · drag a corner to resize · Delete removes the selected one.',
            Mode.MEASURE: 'Measure — drag along any distance. Set the scale (Ctrl+K) to read it in µm; '
                          'you can also measure a known distance and convert it into the calibration.',
        }
        self.statusBar().showMessage(hints[mode])
        self.canvas.setFocus()
        self.canvas.update()

    def _set_landmark_target(self, name):
        self.canvas.landmark_target = name
        self._update_landmark_status()
        self.canvas.update()

    def keyPressEvent(self, e):
        # S is the advertised key for the spinal cord; C stays as an alias.
        mapping = {Qt.Key_V: Mode.VIEW, Qt.Key_E: self._last_edit_mode,
                   Qt.Key_H: Mode.LANDMARK, Qt.Key_T: Mode.LANDMARK, Qt.Key_R: Mode.ROI,
                   Qt.Key_S: Mode.CORD, Qt.Key_C: Mode.CORD,
                   Qt.Key_N: Mode.NEURON, Qt.Key_M: Mode.MEASURE}
        if e.key() in (Qt.Key_H, Qt.Key_T) and not e.modifiers():
            self._set_edit_target('head' if e.key() == Qt.Key_H else 'tail')
            return
        if e.key() in mapping and not e.modifiers():
            self._set_mode(mapping[e.key()])
        else:
            super().keyPressEvent(e)

    # ── undo ──────────────────────────────────────────────────────────────────
    def push_undo(self):
        self.undo_stack.append(self.state.snapshot())
        if len(self.undo_stack) > 60:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            self.statusBar().showMessage('Nothing to undo.')
            return
        self.state.restore(self.undo_stack.pop())
        self.after_edit('Undo', push=False)

    def after_edit(self, msg, push=False):
        self.refresh_list()
        self.canvas.update()
        self._update_landmark_status()
        self._update_measure_label()
        self.statusBar().showMessage(msg)

    def show_coords(self, x, y):
        if self.state.corrected is not None:
            self.statusBar().showMessage(f'x={int(x)}  y={int(y)}    '
                                         f'neurons={len(self.state.neurons)}', 1500)

    # ── panel refresh ─────────────────────────────────────────────────────────
    def refresh_list(self):
        st = self.state
        self.count_label.setText(f'Count: {len(st.neurons)}')
        self.neuron_list.blockSignals(True)
        self.neuron_list.clear()
        for i, n in enumerate(st.neurons):
            tag = 'by hand' if n['source'] == 'user' else f'{n["conf"]:.2f}'
            item = QListWidgetItem(f'{i + 1:3d}   {tag}')
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if n.get('checked') else Qt.Unchecked)
            self.neuron_list.addItem(item)
        if st.selected is not None and 0 <= st.selected < self.neuron_list.count():
            self.neuron_list.setCurrentRow(st.selected)
        self.neuron_list.blockSignals(False)
        self._sync_check_all()

    def _read_checks(self):
        """Copy tick state out of the list widget and back onto the neurons."""
        for i in range(self.neuron_list.count()):
            if i < len(self.state.neurons):
                self.state.neurons[i]['checked'] = \
                    self.neuron_list.item(i).checkState() == Qt.Checked

    def _sync_check_all(self):
        n = len(self.state.neurons)
        checked = sum(1 for x in self.state.neurons if x.get('checked'))
        self.check_all_cb.blockSignals(True)
        self.check_all_cb.setChecked(n > 0 and checked == n)
        self.check_all_cb.blockSignals(False)

    def _check_all(self, on):
        for n in self.state.neurons:
            n['checked'] = bool(on)
        self.refresh_list()
        self.canvas.update()

    def delete_checked(self):
        self._read_checks()
        keep = [n for n in self.state.neurons if not n.get('checked')]
        removed = [n for n in self.state.neurons if n.get('checked')]
        if not removed:
            self.statusBar().showMessage('No neurons are ticked.')
            return
        self.push_undo()
        for n in removed:
            self.state.edits['removed'].append({'box': n['box'], 'source': n['source']})
        self.state.neurons = keep
        self.state.selected = None
        self.after_edit(f'Removed {len(removed)} ticked neuron(s).')

    def clear_all_neurons(self):
        st = self.state
        if not st.neurons:
            return
        if QMessageBox.question(
                self, 'Clear all neurons',
                f'Remove all {len(st.neurons)} neurons from this image?\n\n'
                'You can undo this with Ctrl+Z.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.push_undo()
        for n in st.neurons:
            st.edits['removed'].append({'box': n['box'], 'source': n['source']})
        n_was = len(st.neurons)
        st.neurons = []
        st.selected = None
        self.after_edit(f'Cleared all {n_was} neurons.')

    def _select_from_list(self, row):
        self._read_checks()
        self.state.selected = row if row >= 0 else None
        self._sync_check_all()
        self.canvas.update()

    def _update_landmark_status(self):
        # flag missing landmarks on the radio buttons, keeping the shortcut visible
        for name in ('head', 'tail'):
            base = TARGET_LABELS[name]
            found = bool(self.state.landmarks.get(name))
            self.target_buttons[name].setText(base if found else f'{base}   ✗ not found')
        name = self.canvas.landmark_target
        if self.canvas.mode != Mode.LANDMARK or name not in ('head', 'tail'):
            self.landmark_status.setText('')
            return
        lm = self.state.landmarks.get(name)
        if lm:
            conf = lm.get('confidence', 0)
            src = 'placed by hand' if conf >= 0.999 else f'detected, {conf:.2f}'
            self.landmark_status.setText(f'{name}: {src}. Drag to replace, Delete to erase.')
        else:
            self.landmark_status.setText(f'{name} not found — drag on the image to place it.')

    # ── file handling ─────────────────────────────────────────────────────────
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open image', '', 'Images (*.jpg *.jpeg *.png)')
        if path:
            self.image_list = [path]
            self.image_index = 0
            self.load_current()

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, 'Open folder of images')
        if not d:
            return
        files = []
        for root, _, names in os.walk(d):
            for nm in names:
                if nm.lower().endswith(IMAGE_EXTS):
                    files.append(os.path.join(root, nm))
        if not files:
            QMessageBox.warning(self, 'DanaFish', 'No .jpg/.jpeg/.png images found in that folder.')
            return
        self.image_list = sorted(files)
        self.image_index = 0
        self.load_current()

    def next_image(self):
        if self.image_index + 1 < len(self.image_list):
            self.image_index += 1
            self.load_current()

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load_current()

    def load_current(self):
        if not (0 <= self.image_index < len(self.image_list)):
            return
        path = self.image_list[self.image_index]
        rotator, _, _ = self._models()

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            corrected, info = rotator.correct_image_orientation(path, save_annotated=False)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Orientation failed',
                                 f'Could not process:\n{path}\n\n{exc}')
            return
        finally:
            QApplication.restoreOverrideCursor()

        if corrected is None:
            QMessageBox.warning(self, 'DanaFish',
                                f'Rotation correction returned nothing for:\n{path}\n\n'
                                'You can still place landmarks and an ROI by hand.')
            corrected = cv2.imread(path)
            info = {}

        self.state = State()
        self.undo_stack.clear()
        st = self.state
        st.image_path = path
        st.corrected = corrected
        st.info = info or {}

        lms = (info or {}).get('corrected_landmarks') or {}
        for name in ('head', 'body', 'tail'):
            lm = lms.get(name)
            st.landmarks[name] = copy.deepcopy(lm) if lm else None

        # The auto body box is offered as a starting ROI, but the operator is
        # expected to draw the real one — that is the point of this tool.
        if st.landmarks.get('body'):
            st.roi = list(st.landmarks['body']['bbox'])

        self.canvas.measure_line = None
        self.last_measure_px = None
        self.canvas.set_image(corrected)
        self.refresh_list()
        self._update_landmark_status()
        # scale_factor is per-image, so the µm-per-displayed-pixel readout must refresh
        self._update_scale_label()

        self.setWindowTitle(f'DanaFish — Semi-Automatic  |  '
                            f'[{self.image_index + 1}/{len(self.image_list)}]  {Path(path).name}')

        # Run the whole automatic pipeline straight away, so the operator is
        # presented with a finished result and only has to intervene where it is
        # wrong — rather than being asked to draw an ROI before seeing anything.
        if st.roi is not None:
            self.detect_in_roi(auto=True)
            self.statusBar().showMessage(
                f'{Path(path).name}: {len(st.neurons)} neurons detected automatically. '
                f'Correct them directly, or press V to just look around.')
            self._set_edit_target('neurons')
        else:
            missing = [n for n in ('head', 'body', 'tail') if not st.landmarks[n]]
            self.statusBar().showMessage(
                f'{Path(path).name}: could not locate the fish '
                f'({", ".join(missing) or "no body"} not found). '
                f'Draw the region of interest by hand, then press Ctrl+D.')
            self._set_edit_target('roi')

    # ── orientation ───────────────────────────────────────────────────────────
    def reapply_orientation(self):
        st = self.state
        if st.corrected is None:
            return
        if not all(st.landmarks[k] for k in ('head', 'body', 'tail')):
            QMessageBox.information(
                self, 'DanaFish',
                'Head, body and tail are all needed to compute the orientation.\n\n'
                'Switch to landmark mode (L) and place the missing ones first.')
            return
        rotator, _, _ = self._models()
        h_flip, v_flip = rotator.calculate_orientation_flips(st.landmarks)
        if not (h_flip or v_flip):
            self.statusBar().showMessage(
                'Orientation already correct for these landmarks — nothing to flip.')
            return

        self.push_undo()
        st.corrected = rotator.flip_image(st.corrected, h_flip, v_flip)
        h, w = st.corrected.shape[:2]

        def flip_box(b):
            x1, y1, x2, y2 = b
            if h_flip:
                x1, x2 = w - x2, w - x1
            if v_flip:
                y1, y2 = h - y2, h - y1
            return [x1, y1, x2, y2]

        for name, lm in st.landmarks.items():
            if lm:
                nb = flip_box(lm['bbox'])
                lm['bbox'] = nb
                lm['center'] = [(nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2]
        if st.roi:
            st.roi = flip_box(st.roi)
        for n in st.neurons:
            n['box'] = flip_box(n['box'])
        st.cord_xy = None
        st.edits['reoriented'] = True

        self.canvas.set_image(st.corrected)
        self.after_edit(f'Re-applied orientation (h_flip={h_flip}, v_flip={v_flip}). '
                        'Re-run detection to refresh the spinal cord.')

    # ── detection ─────────────────────────────────────────────────────────────
    def detect_in_roi(self, auto=False):
        """Detect neurons in the ROI. `auto` marks the automatic run on load,
        which must not prompt or steal the mode from the operator."""
        st = self.state
        if st.corrected is None:
            return
        if st.roi is None:
            if auto:
                return
            QMessageBox.information(
                self, 'DanaFish',
                'Draw the region of interest first, then press Ctrl+D.')
            self._set_edit_target('roi')
            return

        _, detector, analyzer = self._models()
        self.push_undo()
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, st.corrected)
            try:
                old_conf = detector.confidence
                detector.confidence = float(self.conf_spin.value())
                labels, boxes, confs = detector.get_predictions(tmp.name)
                detector.confidence = old_conf
            finally:
                os.unlink(tmp.name)

            x1, y1, x2, y2 = st.roi
            kept = []
            for box, conf in zip(boxes, confs):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    kept.append({'box': [float(v) for v in box],
                                 'conf': float(conf), 'source': 'model'})

            user_boxes = [n for n in st.neurons if n['source'] == 'user'] \
                if self.keep_user_cb.isChecked() else []
            st.neurons = kept + user_boxes
            st.selected = None
            st.edits['roi_manual'] = True

            self._fit_cord()
        except Exception as exc:
            QMessageBox.critical(self, 'Detection failed',
                                 f'{exc}\n\n{traceback.format_exc()}')
        finally:
            QApplication.restoreOverrideCursor()

        n_model = len([n for n in st.neurons if n['source'] == 'model'])
        n_user = len([n for n in st.neurons if n['source'] == 'user'])
        self.after_edit(f'Detected {n_model} neurons'
                        + (f' (+{n_user} placed by hand)' if n_user else ''))
        if not auto:
            self._set_edit_target('neurons')

    def _fit_cord(self, force=False):
        """Fit the spinal cord and seed the draggable control points from it.

        A cord the operator has already edited is left alone unless `force`,
        so re-running detection does not throw away their corrections.
        """
        st = self.state
        if st.cord_edited and not force:
            st.rebuild_cord()
            return
        _, _, analyzer = self._models()
        try:
            boxes = [np.array(n['box']) for n in st.neurons]
            pts = analyzer.find_brightest_points_in_exact_region(st.corrected, st.roi, boxes)
            if len(pts) >= 3:
                xs, ys, _ = analyzer.fit_curve_to_points(pts)
                xs, ys = np.asarray(xs), np.asarray(ys)
                if len(xs) > 1:
                    idx = np.linspace(0, len(xs) - 1, min(N_CORD_HANDLES, len(xs))).astype(int)
                    st.cord_points = [(float(xs[i]), float(ys[i])) for i in idx]
                    st.cord_edited = False
                    st.cord_selected = None
                    st.rebuild_cord()
                    return
        except Exception:
            pass
        st.cord_points = []
        st.cord_xy = None
        st.spinal_length = 0.0

    def reset_cord(self):
        """Throw away cord edits and refit from the image."""
        if self.state.roi is None:
            return
        self.push_undo()
        self._fit_cord(force=True)
        self.after_edit('Spinal cord refitted automatically.')

    # ── export ────────────────────────────────────────────────────────────────
    def export_current(self):
        st = self.state
        if st.corrected is None or not st.neurons:
            QMessageBox.information(self, 'DanaFish',
                                    'Nothing to export yet — load an image, draw the ROI and detect.')
            return
        if self.output_dir is None:
            d = QFileDialog.getExistingDirectory(self, 'Choose an output folder')
            if not d:
                return
            self.output_dir = d

        _, _, analyzer = self._models()
        stem = Path(st.image_path).stem
        os.makedirs(self.output_dir, exist_ok=True)

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            boxes = np.array([n['box'] for n in st.neurons], dtype=float)
            labels = [1] * len(st.neurons)
            confs = [n['conf'] for n in st.neurons]
            protected = [n['box'] for n in st.neurons if n['source'] == 'user']

            cord_override = st.cord_xy if st.cord_edited else None
            annotated, results = analyzer.analyze_exact_body_region(
                st.corrected, st.roi, labels, boxes, confs,
                protected_boxes=protected, cord_override=cord_override)

            annotated = self._burn_scale_bar(annotated)
            ann_path = os.path.join(self.output_dir, f'annotated_{Path(st.image_path).name}')
            cv2.imwrite(ann_path, annotated)

            xlsx_path = os.path.join(self.output_dir, f'metrics_{stem}.xlsx')
            if results.get('segment_data'):
                analyzer.export_to_excel(results, xlsx_path)
            else:
                xlsx_path = None

            upp = self.um_per_display_px_raw()
            roi_w = st.roi[2] - st.roi[0]
            roi_h = st.roi[3] - st.roi[1]
            cord_px = float(results.get('spinal_length', 0.0))

            sidecar = {
                'image': st.image_path,
                'mode': 'semi-automatic',
                'calibration': {
                    'um_per_original_px': self.um_per_orig_px,
                    'image_scale_factor': self.scale_factor(),
                    'um_per_displayed_px': upp,
                    'calibrated': bool(upp),
                },
                'roi': [float(v) for v in st.roi],
                'roi_width_px': float(roi_w),
                'roi_height_px': float(roi_h),
                'roi_width_um': (roi_w * upp) if upp else None,
                'roi_height_um': (roi_h * upp) if upp else None,
                'spinal_length_um': (cord_px * upp) if upp else None,
                'roi_drawn_manually': st.edits['roi_manual'],
                'spinal_cord_edited_by_hand': st.cord_edited,
                'spinal_cord_points': [[float(a), float(b)] for a, b in st.cord_points],
                'orientation_reapplied': st.edits['reoriented'],
                'landmarks_edited': st.edits['landmarks_edited'],
                'detection_confidence': float(self.conf_spin.value()),
                'neurons_total': len(st.neurons),
                'neurons_from_model': sum(1 for n in st.neurons if n['source'] == 'model'),
                'neurons_added_by_hand': sum(1 for n in st.neurons if n['source'] == 'user'),
                'neurons_deleted_by_hand': len(st.edits['removed']),
                'neurons_in_region_after_analysis': results.get('neurons_in_region', 0),
                'spinal_length_px': results.get('spinal_length', 0.0),
                'neurons': st.neurons,
                'deleted': st.edits['removed'],
            }
            json_path = os.path.join(self.output_dir, f'edits_{stem}.json')
            with open(json_path, 'w', encoding='utf-8') as fh:
                json.dump(sidecar, fh, indent=2, default=float)

            self._append_summary(results)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Export failed', f'{exc}\n\n{traceback.format_exc()}')
            return
        finally:
            QApplication.restoreOverrideCursor()

        QMessageBox.information(
            self, 'Exported',
            f'Saved to:\n{self.output_dir}\n\n'
            f'• {Path(ann_path).name}\n'
            f'• {Path(xlsx_path).name if xlsx_path else "(no Excel — no segment data)"}\n'
            f'• {Path(json_path).name}\n'
            f'• summary_manual.xlsx\n\n'
            f'Neurons counted: {len(st.neurons)}')
        self.statusBar().showMessage(f'Exported {stem} — {len(st.neurons)} neurons.')

    def _burn_scale_bar(self, img):
        """Draw a scale bar into the exported annotated image, for figures."""
        upp = self.um_per_display_px_raw()
        if not upp:
            return img
        out = img.copy()
        h, w = out.shape[:2]
        bar_um = nice_bar_length(w * upp)
        bar_px = int(round(bar_um / upp))
        if bar_px < 10 or bar_px > w - 40:
            return out

        x2 = w - 24
        x1 = x2 - bar_px
        y = h - 28
        cv2.rectangle(out, (x1 - 12, y - 30), (x2 + 12, y + 12), (0, 0, 0), -1)
        cv2.line(out, (x1, y), (x2, y), (255, 255, 255), 4)
        cv2.line(out, (x1, y - 6), (x1, y + 6), (255, 255, 255), 2)
        cv2.line(out, (x2, y - 6), (x2, y + 6), (255, 255, 255), 2)

        label = f'{bar_um:g} um'
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(out, label, (int((x1 + x2) / 2 - tw / 2), y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    def _append_summary(self, results):
        """Append this image to summary_manual.xlsx, replacing any earlier row for it."""
        st = self.state
        path = os.path.join(self.output_dir, 'summary_manual.xlsx')
        upp = self.um_per_display_px_raw()
        cord_px = float(results.get('spinal_length', 0.0))
        roi_w = st.roi[2] - st.roi[0]
        roi_h = st.roi[3] - st.roi[1]
        row = {
            'Image': Path(st.image_path).name,
            'Neurons': len(st.neurons),
            'From_model': sum(1 for n in st.neurons if n['source'] == 'model'),
            'Added_by_hand': sum(1 for n in st.neurons if n['source'] == 'user'),
            'Deleted_by_hand': len(st.edits['removed']),
            'Line_px': round(cord_px, 1),
            'Line_um': round(cord_px * upp, 2) if upp else None,
            'ROI_width_um': round(roi_w * upp, 2) if upp else None,
            'ROI_height_um': round(roi_h * upp, 2) if upp else None,
            'Neurons_per_100um': (round(len(st.neurons) / (cord_px * upp) * 100, 2)
                                  if upp and cord_px > 0 else None),
            'um_per_px': round(upp, 5) if upp else None,
            'ROI': str([int(v) for v in st.roi]),
            'Confidence': float(self.conf_spin.value()),
            'Landmarks_edited': ', '.join(st.edits['landmarks_edited']) or '',
            'Reoriented': st.edits['reoriented'],
        }
        try:
            df = pd.read_excel(path) if os.path.exists(path) else pd.DataFrame()
            if not df.empty and 'Image' in df.columns:
                df = df[df['Image'] != row['Image']]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([row])
        df.to_excel(path, index=False)

    # ── help ──────────────────────────────────────────────────────────────────
    def show_help(self):
        QMessageBox.information(self, 'DanaFish — semi-automatic', """
<b>Workflow</b><br>
1. <b>Open Image</b> or <b>Open Folder</b> — orientation correction runs automatically.<br>
2. <b>L</b> — fix head / body / tail if the model got them wrong; drag to place, Delete to erase.
   Then <b>Ctrl+R</b> to re-apply orientation from your corrected landmarks.<br>
3. <b>R</b> — drag to draw the ROI over the spinal-cord region (a corner handle adjusts it).<br>
4. <b>Ctrl+D</b> — detect neurons inside the ROI and fit the spinal cord.<br>
5. <b>S</b> — correct the spinal cord. It is drawn as a smooth curve through a handful of white
control points seeded from the automatic fit:<br>
&nbsp;&nbsp;• drag a point → bend the curve there<br>
&nbsp;&nbsp;• click empty space → insert a new point<br>
&nbsp;&nbsp;• right-click a point (or Delete) → remove it<br>
&nbsp;&nbsp;• <b>Ctrl+Shift+R</b> → discard your edits and refit automatically<br>
An edited cord drives the neuron filtering, the distances and the reported length — not just the
picture.<br>
6. <b>N</b> — correct the neurons:<br>
&nbsp;&nbsp;• drag on empty space → add a neuron<br>
&nbsp;&nbsp;• right-click a neuron → delete it<br>
&nbsp;&nbsp;• drag a neuron box → move it<br>
&nbsp;&nbsp;• drag a corner handle → resize it<br>
&nbsp;&nbsp;• Delete key → remove the selected neuron<br>
7. <b>Ctrl+S</b> — export annotated image, Excel metrics, and an edit log.<br><br>

<b>Units</b> — lengths are shown in <b>pixels by default</b>. Choose Micrometres in the side panel
to switch; that option becomes available once a scale is set. The choice affects only what is
displayed — if a scale exists, exported results always carry the µm columns regardless.<br><br>

<b>Scale in micrometres</b><br>
<b>Ctrl+K</b> sets µm per pixel. Either type it in, or press <b>M</b> and drag along a distance you
know (a scale bar in the image, or a feature of known size), then open Ctrl+K and enter what that
distance actually is — it converts the measurement into a calibration for you.<br>
Once set, the ruler, the scale bar, the ROI label and the exported spreadsheet all report µm, and
the calibration is remembered in <tt>calibration.json</tt> for later sessions.<br>
The value is stored per <i>original</i> microscope pixel, so it stays correct even though each
image is rescaled onto the 840×840 canvas by a different factor.<br><br>

<b>Keys</b><br>
<b>V</b> view · <b>E</b> edit<br>
<b>H</b> head · <b>T</b> tail · <b>R</b> region of interest · <b>S</b> spinal cord · <b>N</b> neurons<br>
<b>M</b> measure · <b>Ctrl+K</b> scale · <b>Ctrl+D</b> detect · <b>Ctrl+Shift+R</b> refit cord ·
<b>Ctrl+Z</b> undo · <b>Ctrl+0</b> fit view · <b>PgUp/PgDn</b> previous/next image ·
<b>Ctrl+S</b> export · <b>F1</b> this help<br>
<b>Mouse</b> — wheel zooms, middle-drag pans (left-drag pans in View mode).<br><br>

<b>Colours</b> — <span style="color:#1E90FF">body</span>,
<span style="color:#FF8C00">head</span>,
<span style="color:#DC3CC8">tail</span>,
<span style="color:#00C8FF">ROI</span>,
<span style="color:#00E600">model neuron</span>,
<span style="color:#FFD700">hand-placed neuron</span>,
<span style="color:#FF3C3C">selected / spinal cord</span>.
""")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
