#   ________________________________________ Knee OsteoArthritis CAD ________________________________________________
'''
 *  Feature Extraction and Visualization - Traditional CVML Methods - Deep Learning Method - Hybrid Model (CVML and DL).
 *  Created on: Friday Aug 15 2023 --> Sat July 06 2024
 *  Author    : Mohammad Sayed, Ossamah Qubati - BME-GP-Team 13
 *  Supervised by: Professor Dr. Ahmed M. Badawi
'''
#  _____________________________________________ Libraries ____________________________________________________________
from PyQt5.QtWidgets import QComboBox, QProgressBar, QAbstractItemView, QTableWidget, QTableWidgetItem, QGridLayout, QDesktopWidget, QApplication, QRadioButton, QScrollArea, QMainWindow, QToolTip,QGraphicsOpacityEffect, QFileDialog, QLabel,QMessageBox,QVBoxLayout,QPushButton, QWidget, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPalette, QImage, QTransform, QPixmap, QColor, QIcon, QFont, QPainter, QPainterPath, QPen, QPolygonF, QBrush, QPainter
from PyQt5.QtCore import pyqtSlot, QPoint, QEvent, Qt, pyqtSignal,QTimer, QRect, QDir, QSize, QPoint, QPointF, QPropertyAnimation, QEasingCurve, pyqtProperty
from scipy.stats import skew, kurtosis
from CountdownTimer import CountdownTimer
import threading
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import subprocess
import sys
import cv2
import os
import glob
import time
import csv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage import exposure
from ultralytics import YOLO
from Models.NewFeatureExtractionAndVisualizationModels.modelCNN import SimpleCNN
from joblib import load
from threading import Thread
from time import sleep
import pydicom
import tempfile
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import UID, generate_uid
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet201
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#  _____________________________________________ Splash Screen ____________________________________________________________
class SplashScreen(QWidget):
    variableChanged_Full = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        self.setWindowTitle('Progress')
        self.setFixedSize(int(0.3645833 * screen_width), int(0.0520833 * screen_width))
        self.center_window()

        # Set the window to be transparent and remove window decorations
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(1.0)

        self.full = False

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(int(0.0520833 * screen_width))
        self.progress_bar.setFixedWidth(int(0.33854167 * screen_width))
        self.progress_bar.setFixedHeight(int(0.02604167 * screen_width))
        self.progress_bar.setTextVisible(False)  # Hide the text on the progress bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: transparent;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 200);
            }
            QProgressBar::chunk {
                background-color: rgba(34, 200, 34, 200);
                border-radius: 10px;
            }
        """)
        self.setCentralProgress()

        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        self.timer.start(1)  # Update every 10 ms for smoother animation

        self.show()

    def center_window(self):
        screen = QDesktopWidget().screenGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def setCentralProgress(self):
        progress_rect = self.progress_bar.geometry()
        parent_rect = self.rect()
        new_x = (parent_rect.width() - progress_rect.width()) // 2
        new_y = (parent_rect.height() - progress_rect.height()) // 2
        self.progress_bar.setGeometry(new_x, new_y, 240, 25)

    def update_progress(self):
        self.progress_value += 1
        self.progress_bar.setValue(int(self.progress_value))
        if self.progress_value >= 100:
            
            self.full = True
            self.variableChanged_Full.emit(self.full)

            self.timer.stop()
            self.close()
            
            self.full = False
            self.variableChanged_Full.emit(self.full)
    
#  ____________________________________________ TranparentNotifier _____________________________________________________
# class TranparentNotifier(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Black Screen Note")
        
#         screen_resolution = QDesktopWidget().screenGeometry()
#         screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        
#         # self.resize(350, 150)
#         self.resize(int(0.18229167 * screen_width), int(0.078125 * screen_height))
        
#         self.screen_width = screen_width
#         self.screen_height = screen_height
        
#         self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)

#         self.setAttribute(Qt.WA_TranslucentBackground)
#         self.setStyleSheet("background-color: transparent;")

#         self.screen_geometry = QApplication.primaryScreen().availableGeometry()

#         self.messages = [
#             """
#             When you hit the "Diagnose" button, don't panic if KOA CAD looks stuck, It will continue on its own.<br>
#             Don't hit any button unexpectedly.<br>
#             """,
#             """
#             Hint: The Prediction of the only First Radiographic image takes a bit longer.<br>
#             due to the models load.<br>
#             """
#         ]
        
#         self.current_message_index = 0

#         self.statement_label = QLabel()
#         self.statement_label.setAlignment(Qt.AlignCenter)
#         self.statement_label.setToolTip("• Double click to hide this Notification! •")
#         self.statement_label.setStyleSheet("""
#             color: white;
#             background-color: rgba(0, 0, 0, 150);
#             border-radius: 15px;
#             padding: 20px;
#         """)
#         self.statement_label.setFont(QFont("Arial", int(0.0078125 * self.screen_width)))
#         # self.statement_label.setFont(QFont("Arial", 15))

#         self.statement_layout = QVBoxLayout(self)
#         self.statement_layout.addWidget(self.statement_label)
        
#         mar = int(0.005208333 * self.screen_width)
#         self.statement_layout.setContentsMargins(mar, mar, mar, mar)  # Add margins to the label layout for padding

#         self.setLayout(self.statement_layout)
        
#         if self.current_message_index == 0:
#             self.move_to_bottom_right(int(0.3125 * self.screen_width), int(0.004629629 * self.screen_height))
#             # self.move_to_bottom_right(450, 5)
            
            
#         self.animation = None  # Animation instance

#         # Start showing messages
#         self.show_next_message()

#     def move_to_bottom_right(self, X, Y):
#         window_geometry = self.frameGeometry()
#         x = self.screen_geometry.width() - window_geometry.width() - X  # 450 px margin from the right
#         y = self.screen_geometry.height() - window_geometry.height() - Y  # 5 px margin from the bottom
#         self.move(x, y)

#     def slide_out(self):
#         # Animate slide out to the left
#         start_value = self.frameGeometry()
#         end_value = QRect(self.screen_geometry.width(), self.y(), self.width(), self.height())
#         self.animation = QPropertyAnimation(self, b"geometry")
#         self.animation.setDuration(300)
#         self.animation.setStartValue(start_value)
#         self.animation.setEndValue(end_value)
#         self.animation.setEasingCurve(QEasingCurve.OutCubic)
#         self.animation.finished.connect(self.show_next_message)  # Connect to show next message after animation
#         self.animation.start()

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.slide_out()
#             self.show_next_message()

#     def slide_in(self):
#         # Animate slide in from the right
#         start_value = QRect(self.screen_geometry.width(), self.y(), self.width(), self.height())
#         end_value = self.frameGeometry()
#         self.animation = QPropertyAnimation(self, b"geometry")
#         self.animation.setDuration(300)
#         self.animation.setStartValue(start_value)
#         self.animation.setEndValue(end_value)
#         self.animation.setEasingCurve(QEasingCurve.OutCubic)
#         self.animation.start()

#     def show_next_message(self):

#         if self.current_message_index < len(self.messages):
#             self.statement_label.setText(self.messages[self.current_message_index])
#             self.slide_in()  # Immediately slide in the next message after setting text
#             self.current_message_index += 1
#         else:
#             self.close()  # Close the widget if no more messages

#     def hideEvent(self, event):
#         # Clean up animation object when the widget is hidden or closed
#         if self.animation and self.animation.state() == QPropertyAnimation.Running:
#             self.animation.stop()          
#  ____________________________________________ GuideMessage _____________________________________________________
            
class GuideMessage(QWidget):
    def __init__(self, messages, positions):
        super().__init__()
        self.messages = messages
        self.positions = positions
        self.current_index = 0
        self.initUI()

    def initUI(self):
        self.setFixedSize(350, 250)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        main_layout = QVBoxLayout(self)

        self.message_area = QLabel(self.messages[self.current_index])
        self.message_area.setAlignment(Qt.AlignCenter)
        self.message_area.setStyleSheet("background-color: white; border-radius: 10px;")
        self.message_area.setFont(QFont("Arial", 12))
        main_layout.addWidget(self.message_area)

        button_layout = QHBoxLayout()
        self.previous_button = QPushButton('Previous', self)
        self.next_button = QPushButton('Next', self)

        button_color = QColor(64, 164, 64, 150)
        hover_color = QColor(64, 164, 64, 200)
        self.set_button_colors(self.previous_button, button_color, hover_color)
        self.set_button_colors(self.next_button, button_color, hover_color)

        self.previous_button.clicked.connect(self.previous_message)
        self.next_button.clicked.connect(self.next_message)

        self.previous_button.setFont(QFont("Arial", 14))
        self.next_button.setFont(QFont("Arial", 14))

        button_layout.addWidget(self.previous_button)
        button_layout.addWidget(self.next_button)
        main_layout.addLayout(button_layout)

        finish_layout = QHBoxLayout()

        self.finish_button = QPushButton('Finish', self)
        self.set_button_colors(self.finish_button, button_color, hover_color)
        self.finish_button.clicked.connect(self.close_guide)

        self.finish_button.setFont(QFont("Arial", 14))

        finish_layout.addWidget(self.finish_button)
        main_layout.addLayout(finish_layout)

        self.update_position()
        self.show()

    def set_button_colors(self, button, color, hover_color):
        palette = button.palette()
        palette.setColor(QPalette.Button, QColor(255, 255, 255, 0))
        palette.setColor(QPalette.ButtonText, Qt.white)
        button.setPalette(palette)
        button.setAutoFillBackground(True)
        button.setStyleSheet(f"QPushButton:hover {{ background-color: {hover_color.name()}; color: white; border-radius: 5px; }}"
                             f"QPushButton {{ background-color: {color.name()}; border-radius: 5px; }}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 25, 25)

    def previous_message(self):
        self.current_index -= 1
        if self.current_index >= 0:
            self.message_area.setText(self.messages[self.current_index])
            self.update_position()

    def next_message(self):
        self.current_index += 1
        if self.current_index < len(self.messages):
            self.message_area.setText(self.messages[self.current_index])
            self.update_position()
        else:
            self.close_guide()

    def update_position(self):
        pos = self.positions[self.current_index]
        self.move(pos)
        self.raise_()

    def close_guide(self):
        self.close()
#  ____________________________________________ ZoomingLabel _____________________________________________________

class ZoomingLabel(QLabel):
    def __init__(self, x, y, F, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignCenter)

        self.x = x
        self.y = y
        self.F = F

        self.dragging = False
        self.zoom_factor = 1.0
        self.image = None
        self.offset = QPoint(self.x, self.y)

    def setPixmap(self, pixmap):
        self.image = pixmap
        self.zoom_factor = self.F
        super().setPixmap(self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start = event.pos()
            self.image_start = self.offset

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.drag_start
            self.offset = self.image_start + delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        self.zoom_factor *= factor
        self.update()

    def paintEvent(self, event):
        if self.image:
            painter = QPainter(self)
            scaled_image = self.image.scaled(self.size() * self.zoom_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            rect = QRect(QPoint(0, 0), scaled_image.size())
            rect.moveCenter(self.rect().center() + self.offset)
            painter.drawPixmap(rect, scaled_image)
            
#  ____________________________________________ AnimatedButton _____________________________________________________

class AnimatedButton(QPushButton):
    def __init__(self, icon_path, parent=None):
        super().__init__(parent)
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.setFixedSize(int(0.0234375 * self.screen_width), int(0.0234375 * self.screen_width))
        self.setIconSize(self.size())
        self.icon_path = icon_path
        self._rotation = 0
        self._pixmap = QPixmap(icon_path)

        self.BorderRadius = int(0.02604167 * self.screen_width)
        # Set the icon with correct scaling initially
        self.update_icon()
        
        self.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: {self.BorderRadius} px;
            }
        """)

        self.press_animation = QPropertyAnimation(self, b"rotation")
        self.press_animation.setDuration(350)
        self.press_animation.setEasingCurve(QEasingCurve.InOutQuad)

    @pyqtProperty(float)
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self.update_icon()

    def update_icon(self):
        transform = QTransform()
        transform.rotate(self._rotation)
        rotated_pixmap = self._pixmap.transformed(transform, mode=Qt.SmoothTransformation)
        scaled_pixmap = rotated_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setIcon(QIcon(scaled_pixmap))

    def mousePressEvent(self, event):
        self.press_animation.setStartValue(0)
        self.press_animation.setEndValue(360)
        self.press_animation.start()
        super().mousePressEvent(event)
#  ____________________________________________ CustomMessageBox _____________________________________________________
            
class CustomMessageBox(QMessageBox):
    def __init__(self, message, icon_path, button_text, flag_cancel_button):
        super(CustomMessageBox, self).__init__()
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.setText(message)
        font = QFont()
        font.setPointSize(int(0.005208333 * self.screen_width))
        self.setFont(font)
        
        primary_button = self.addButton(button_text, QMessageBox.AcceptRole)
        if flag_cancel_button == 1:
            self.addButton("Cancel", QMessageBox.RejectRole)
        primary_button.setStyleSheet("color: black;")
        self.setStyleSheet("QMessageBox { background-color: white; color: white; }")

#  _____________________________________________ ImageLabel __________________________________________________________
class ImageLabel(QLabel):
    def __init__(self, parent, apply_effects=True):
        super().__init__(parent)
        self.crop_rect = QRect()
        self.mouse_pressed = False
        self.setScaledContents(True)
        
        self.apply_effects = apply_effects
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.OutQuart)
        self.leaveEvent(True)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.mouse_pressed:
            painter = QPainter(self)
            color = QColor(0, 255, 0)
            color.setAlpha(255)
            pen = QPen(color)
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(self.crop_rect)

    def enterEvent(self, event):
        if self.apply_effects:
            self.animation.stop()
            self.animation.setDirection(QPropertyAnimation.Forward)
            self.animation.start()

    def leaveEvent(self, event):
        if self.apply_effects:
            self.animation.stop()
            self.animation.setDirection(QPropertyAnimation.Backward)
            self.animation.start()
    
    def enableEffects(self):
        self.apply_effects = True
        self.opacity_effect.setOpacity(1.0)

    def disableEffects(self):
        self.apply_effects = False
        self.opacity_effect.setOpacity(1.0)

    def hasText(self):
        return bool(self.text())

    def hasBackgroundImage(self):
        style_sheet = self.styleSheet()
        return "background-image:" in style_sheet

    def updateEffectsBasedOnStyleAndPixmap(self):
        if self.hasBackgroundImage():
            self.disableEffects()
        else:
            if self.hasText():
                self.enableEffects()
            else:
                self.disableEffects()
#  _____________________________________________ VideoPlayer  ____________________________________________________________
class VideoPlayer(QWidget):
    def __init__(self, video_path):
        super().__init__()
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Manual Feature Extractor")
        self.resize(int(0.208333333 * self.screen_width), int(0.104167 * self.screen_width))
        self.setWindowIcon(QIcon('imgs/help-circle.svg'))
        self.label = QLabel("Loading Video...", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.video_path = video_path
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        self.video_thread = Thread(target=self.playVideo)
        self.video_thread.start()

    def playVideo(self):
        while True:
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(img))
                sleep(0.005)

            cap.release()
#  _____________________________________________ Fused CVML and Ai CAD Screen ____________________________________________________
class Fused_CVML_and_Ai_CAD_Screen(QWidget):
    variableChanged = pyqtSignal(int)
    variableChanged_Equalize_Feature_Visualization = pyqtSignal(bool)
    variableChanged_Auto_mode = pyqtSignal(bool)
    variableChanged_Visualization = pyqtSignal(bool)
    
    def __init__(self, main_window, conventional_class, Ai_class):
        super().__init__()
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        self.resize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setMinimumSize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setMaximumSize(screen_width, screen_height)
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.main_window = main_window
        self.main_window.variableChanged3.connect(self.handle_variable_changed)
        self.main_window.VariableChanged_Combo_Fused.connect(self.handle_variable_changed_SR)
        
        
        self.Conventional_CAD_Screen = conventional_class
        self.AI_Automated_CAD_Screen = Ai_class
        
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen(self, None)
# ______________________________________________ initialization __________________________________________________
        self.pred_class_DONE = None
        self.predict_Class_NN = None
        self.predict_probability_NN = None
        self.score_NN_Class5 = 0
        
        self.ThirdSetVariable = 1
        self.load_NN_Flag = 0
        self.load_NN_70_30_Flag = 0
        self.load_NN_60_40_Flag = 0
        
        self.SR = 2
# ______________________________________________ show_main_content __________________________________________________
        self.show_main_content()
 
    def handle_variable_changed_SR(self, new_value):
        self.SR = new_value
        print("SR", self.SR)
        
        if self.main_window.clasify_indicator == 1:
            self.show_predictions()
        else:
            pass
            print(f"clasify_indicator {self.main_window.clasify_indicator}")
            
    def handle_variable_changed(self, new_value):
        self.ThirdSetVariable = new_value
        # print("ThirdSetVariable", self.ThirdSetVariable)
        
    def show_main_content(self):
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)
        self.image_label = ImageLabel(self, apply_effects = False)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setFixedWidth(int(0.3645833 * self.screen_width))
        self.image_label.setFixedHeight(int(0.3125 * self.screen_width))
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        
        self.image_label.updateEffectsBasedOnStyleAndPixmap()        
        
        table_width = self.AI_Automated_CAD_Screen.table.width()
        table_height = self.AI_Automated_CAD_Screen.table.height()
        
        self.table3 = QTableWidget()
        self.table3.setFixedHeight(int(0.25 * table_height))
        self.table3.verticalHeader().setVisible(False)
        self.table3.horizontalHeader().setVisible(False)
        self.table3.setRowCount(1)
        self.table3.setColumnCount(5)
        self.table3.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
        self.table3.setShowGrid(False)
        self.table3.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table3.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table3's fixed size
        table3_width = self.table3.width()
        table3_height = self.table3.height()
        row3_height = table3_height // self.table3.rowCount()
        column3_width = table3_width // self.table3.columnCount()

        for row3 in range(self.table3.rowCount()):
            self.table3.setRowHeight(row3, row3_height)
        
        for column3 in range(self.table3.columnCount()):
            self.table3.setColumnWidth(column3, column3_width)
            
        
        self.table4 = QTableWidget()
        self.table4.setFixedHeight(int(0.25 * table_height))
        self.table4.verticalHeader().setVisible(False)
        self.table4.horizontalHeader().setVisible(False)
        self.table4.setRowCount(1)
        self.table4.setColumnCount(5)
        self.table4.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
        self.table4.setShowGrid(False)
        self.table4.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table4.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table4's fixed size
        table4_width = self.table4.width()
        table4_height = self.table4.height()
        row4_height = table4_height // self.table4.rowCount()
        column4_width = table4_width // self.table4.columnCount()

        for row4 in range(self.table4.rowCount()):
            self.table4.setRowHeight(row4, row4_height)
        
        for column4 in range(self.table4.columnCount()):
            self.table4.setColumnWidth(column4, column4_width)
            
        
        self.image_label2 = ImageLabel("Fused CVML and Ai", apply_effects = False)
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setScaledContents(True)
        self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
        self.image_label2.setFixedWidth(int(0.4244791667 * self.screen_width))
        self.image_label2.setFixedHeight(int(0.25 * table_height))
        self.image_label2.show()
        
        
        
        font = QFont()
        font.setPointSize(int(0.009895833 * self.screen_width))
        font.setBold(False)
        self.image_label2.setFont(font)
        self.table3.setFont(font)
        self.table4.setFont(font)
        
        self.table_layout_VBOX = QVBoxLayout()
        self.table_layout_VBOX.setSpacing(0)
        self.table_layout_VBOX.setContentsMargins(0, 0, 0, 0)
        self.table_layout_VBOX.addWidget(self.table3)
        self.table_layout_VBOX.addWidget(self.table4)

        self.table_layout_VBOX_Widget = QWidget()
        self.table_layout_VBOX_Widget.setFixedHeight(table_height)
        self.table_layout_VBOX_Widget.setFixedWidth(int(2.15 *table_height))
        self.table_layout_VBOX_Widget.setLayout(self.table_layout_VBOX)
        self.table_layout_VBOX_Widget.setStyleSheet("background-color: transparent; color: rgba(254,229,2,255);")
        
        self.table_layout_VBOX_Widget.setVisible(False)
        self.table3.setVisible(False)
        self.table4.setVisible(False)
        
        spacerdashed1 = QSpacerItem(int(0.002604166 * self.screen_width), 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacerdashed2 = QSpacerItem(int(0.002604166 * self.screen_width), 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        self.table_layout = QHBoxLayout()
        self.table_layout.setSpacing(0)
        self.table_layout.setContentsMargins(0, 0, 0, 0)
        self.table_layout.addSpacerItem(spacerdashed1)
        self.table_layout.addWidget(self.image_label2)
        self.table_layout.addWidget(self.table_layout_VBOX_Widget)
        self.table_layout.addSpacerItem(spacerdashed2)

        self.table_layout_Widget = QWidget()
        self.table_layout_Widget.setLayout(self.table_layout)
        self.table_layout_Widget.setStyleSheet("background-color: transparent; color: rgba(254,229,2,255);")
        self.table_layout_Widget.setVisible(True)
        
        
        HBoxLayout1 = QHBoxLayout()
        HBoxLayout1.addWidget(self.image_label)
        VBoxLayout1 = QVBoxLayout()
        VBoxLayout1.addLayout(HBoxLayout1)
        VBoxLayout1.addWidget(self.table_layout_Widget)
        self.setLayout(VBoxLayout1)
        
# __________________________________________________ Functions _______________________________________________________
    def mouseDoubleClickEvent(self, event):
        self.main_window.load_main_img()
    
    def on_button_click(self):
        self.image_label.clear()
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        
        self.on_combo_clear()
        
    def on_combo_clear(self):
        self.pred_class_DONE = None
        self.predict_Class_NN = None
        self.predict_probability_NN = None
        self.score_NN_Class5 = 0
        
        self.table3.clearContents()
        self.table4.clearContents()
        self.table_layout_VBOX_Widget.setVisible(False)
        self.table3.setVisible(False)
        self.table4.setVisible(False)

        self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
        self.image_label2.setText("Fused CVML and Ai")
        self.image_label2.setVisible(True)
        
        
    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")
    
    def load_NN_model(self):
        if not hasattr(self, 'NN_model'):
            with open("Models/Fusion Models/NN_model_80_20.pkl", "rb") as nn:
                self.NN_model = load(nn)
    
        if not hasattr(self, 'NN_scaler'):
            with open("Models/Fusion Models/NN_scaler_80_20.pkl", "rb") as nn_sc:
                self.NN_scaler = load(nn_sc)
        self.load_NN_Flag = 1
        
    
    def load_NN_model_70_30(self):
        if not hasattr(self, 'NN_model_70_30'):
            with open("Models/Fusion Models/NN_model.pkl", "rb") as nn:
                self.NN_model_70_30 = load(nn)
    
        if not hasattr(self, 'NN_scaler_70_30'):
            with open("Models/Fusion Models/NN_scaler.pkl", "rb") as nn_sc:
                self.NN_scaler_70_30 = load(nn_sc)
        self.load_NN_70_30_Flag = 1
        
        
    def load_NN_model_60_40(self):
        if not hasattr(self, 'NN_model_60_40'):
            with open("Models/Fusion Models/NN_model_60_40.pkl", "rb") as nn:
                self.NN_model_60_40 = load(nn)
    
        if not hasattr(self, 'NN_scaler_60_40'):
            with open("Models/Fusion Models/NN_scaler_60_40.pkl", "rb") as nn_sc:
                self.NN_scaler_60_40 = load(nn_sc)
        self.load_NN_60_40_Flag = 1
        
        
        
    def carry_out(self):
        
        # Weighted AVG Fusion Method
        Class_Conventional_Tree = self.Conventional_CAD_Screen.class5
        score_list_Conventional_Tree = self.Conventional_CAD_Screen.output5[0]
        
        Class_Ai = self.AI_Automated_CAD_Screen.class5
        score_list_Ai = self.AI_Automated_CAD_Screen.output5

        if Class_Conventional_Tree == Class_Ai:
                final_class = Class_Conventional_Tree
        else:
                final_class = Class_Ai
        
        combined_probs = []
        for i in range(len(score_list_Conventional_Tree)):
                ml_prob = score_list_Conventional_Tree[i]
                dl_prob = score_list_Ai[i]
                combined_prob = 0.3 * ml_prob + 0.7 * dl_prob
                combined_probs.append(combined_prob)
            

        self.combined_probs = combined_probs
        pred_class = np.argmax(self.combined_probs)
                
        if (pred_class != final_class) and (final_class in [2,3,4]):
            
                self.pred_class_DONE = final_class
        else :
                self.pred_class_DONE = pred_class
                
        if (pred_class != final_class) and (final_class in [0,1]):
                if (score_list_Ai[0] < 0.52)  < score_list_Ai[1]:
                    self.pred_class_DONE = pred_class
                else :
                    self.pred_class_DONE = final_class
                
        
        
        
        # _______________________________________________________________________________________________________#
        
        # BackPrppagation NN Fusion Method
        score_list_Conventional_output5 = self.Conventional_CAD_Screen.output5
        score_list_Ai_output5 = self.AI_Automated_CAD_Screen.output5
        input_list_NN = score_list_Ai_output5.tolist() + score_list_Conventional_output5[0].tolist()
    
        
        if self.SR == 1:
            
            if self.load_NN_Flag == 0:
                self.load_NN_model()
            print(f"self.load_NN_Flag: {self.load_NN_Flag}")
            
            
            sample = self.NN_scaler.transform([input_list_NN])
            self.predict_Class_NN = self.NN_model.predict(sample)
            self.predict_probability_NN = self.NN_model.predict_proba(sample)

                
        
        if self.SR == 2:
            
            if self.load_NN_70_30_Flag == 0:
                self.load_NN_model_70_30()
            print(f"self.load_NN_Flag: {self.load_NN_70_30_Flag}")
            
            
            sample = self.NN_scaler_70_30.transform([input_list_NN])
            self.predict_Class_NN = self.NN_model_70_30.predict(sample)
            self.predict_probability_NN = self.NN_model_70_30.predict_proba(sample)
                
                       
        if self.SR == 3:
            
            if self.load_NN_60_40_Flag == 0:
                self.load_NN_model_60_40()
            print(f"self.load_NN_Flag: {self.load_NN_60_40_Flag}")

            
            sample = self.NN_scaler_60_40.transform([input_list_NN])
            self.predict_Class_NN = self.NN_model_60_40.predict(sample)
            self.predict_probability_NN = self.NN_model_60_40.predict_proba(sample)



        self.predict_Class_NN = self.predict_Class_NN[0]
        self.predict_probability_NN = self.predict_probability_NN[0]
        self.score_NN_Class5 = 100 * self.predict_probability_NN[self.predict_Class_NN]
            
        self.show_predictions() 
                
                
                  
                  
    def show_predictions(self):
        if self.ThirdSetVariable == 1:

            if self.pred_class_DONE is not None:
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(False)

                self.image_label2.setVisible(True)
            
                # if self.pred_class_DONE == 0:
                #     self.image_label2.setText(f"Weighted Average Fusion: <b>Normal</b>, with Highest score: <b>{100 * self.combined_probs[self.pred_class_DONE]:0.1f}</b>%")
                #     color = QColor(0, 200, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                # if self.pred_class_DONE == 1:
                #     self.image_label2.setText(f"Weighted Average Fusion: <b>Doubtful</b>, with Highest score: <b>{100 * self.combined_probs[self.pred_class_DONE]:0.1f}</b>%")
                #     color = QColor(100, 200, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                # if self.pred_class_DONE == 2:
                #     self.image_label2.setText(f"Weighted Average Fusion: <b>Mild</b>, with Highest score: <b>{100 * self.combined_probs[self.pred_class_DONE]:0.1f}</b>%")
                #     color = QColor(200, 200, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                # if self.pred_class_DONE == 3:
                #     self.image_label2.setText(f"Weighted Average Fusion: <b>Moderate</b>, with Highest score: <b>{100 * self.combined_probs[self.pred_class_DONE]:0.1f}</b>%")
                #     color = QColor(227, 100, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                # if self.pred_class_DONE == 4:
                #     self.image_label2.setText(f"Weighted Average Fusion: <b>Severe</b>, with Highest score: <b>{100 * self.combined_probs[self.pred_class_DONE]:0.1f}</b>%")
                #     color = QColor(255, 0, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                labels_colors = [
                    ("Normal", QColor(0, 200, 0, 140)),
                    ("Doubtful", QColor(100, 200, 0, 140)),
                    ("Mild", QColor(200, 200, 0, 140)),
                    ("Moderate", QColor(227, 100, 0, 140)),
                    ("Severe", QColor(255, 0, 0, 140))
                ]

                for i, (label, color) in enumerate(labels_colors):
                    if self.pred_class_DONE == i:
                        self.image_label2.setText(
                            f"Weighted Average Fusion: <b>{label}</b>, with Highest score: <b>{100 * self.combined_probs[self.pred_class_DONE]:0.1f}</b>%"
                        )
                        self.image_label2.setStyleSheet(
                            f"color: white; background-color: {color.name(QColor.HexArgb)};"
                        )
                        break


                self.item6 = QTableWidgetItem(f"{100*self.combined_probs[0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item6.setBackground(QBrush(color))
                self.item6.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 0, self.item6)
                
                self.item7 = QTableWidgetItem(f"{100*self.combined_probs[1]:0.1f}%")
                color = QColor(*(100, 200, 0))
                color.setAlpha(140)
                self.item7.setBackground(QBrush(color))
                self.item7.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 1, self.item7)
                
                self.item8 = QTableWidgetItem(f"{100*self.combined_probs[2]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                self.item8.setBackground(QBrush(color))
                self.item8.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 2, self.item8)
                
                self.item9 = QTableWidgetItem(f"{100*self.combined_probs[3]:0.1f}%")
                color = QColor(*(227, 100, 0))
                color.setAlpha(140)
                self.item9.setBackground(QBrush(color))
                self.item9.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 3, self.item9)
                
                self.item10 = QTableWidgetItem(f"{100*self.combined_probs[4]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item10.setBackground(QBrush(color))
                self.item10.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 4, self.item10)
            
            else:                
                self.table3.setItem(0, 0, QTableWidgetItem(""))
                self.table3.setItem(0, 1, QTableWidgetItem(""))
                self.table3.setItem(0, 2, QTableWidgetItem(""))
                self.table3.setItem(0, 3, QTableWidgetItem(""))
                self.table3.setItem(0, 4, QTableWidgetItem(""))
                
                pass
                self.table3.setVisible(False)
                self.table_layout_VBOX_Widget.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table3.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                if self.AI_Automated_CAD_Screen.not_detection_flag == 1:
                    pass
                    self.image_label2.clear()
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("Fused CVML and Ai")
            
            
        
        elif self.ThirdSetVariable == 2:
            
            if self.predict_Class_NN is not None:
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table3.setVisible(False)
                self.table4.setVisible(True)

                self.image_label2.setVisible(True)
            
                # if self.predict_Class_NN == 0:
                #     self.image_label2.setText(f"Back Propagation NN Fusion: <b>Normal</b>, with Highest score: <b>{self.score_NN_Class5:0.1f}</b>%")
                #     color = QColor(0, 200, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                # if self.predict_Class_NN == 1:
                #     self.image_label2.setText(f"Back Propagation NN Fusion: <b>Doubtful</b>, with Highest score: <b>{self.score_NN_Class5:0.1f}</b>%")
                #     color = QColor(100, 200, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                # if self.predict_Class_NN == 2:
                #     self.image_label2.setText(f"Back Propagation NN Fusion: <b>Mild</b>, with Highest score: <b>{self.score_NN_Class5:0.1f}</b>%")
                #     color = QColor(200, 200, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                # if self.predict_Class_NN == 3:
                #     self.image_label2.setText(f"Back Propagation NN Fusion: <b>Moderate</b>, with Highest score: <b>{self.score_NN_Class5:0.1f}</b>%")
                #     color = QColor(227, 100, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                # if self.predict_Class_NN == 4:
                #     self.image_label2.setText(f"Back Propagation NN Fusion: <b>Severe</b>, with Highest score: <b>{self.score_NN_Class5:0.1f}</b>%")
                #     color = QColor(255, 0, 0, 140)
                #     self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                labels_colors_nn = [
                    ("Normal", QColor(0, 200, 0, 140)),
                    ("Doubtful", QColor(100, 200, 0, 140)),
                    ("Mild", QColor(200, 200, 0, 140)),
                    ("Moderate", QColor(227, 100, 0, 140)),
                    ("Severe", QColor(255, 0, 0, 140))
                ]

                for i, (label, color) in enumerate(labels_colors_nn):
                    if self.predict_Class_NN == i:
                        self.image_label2.setText(
                            f"Back Propagation NN Fusion: <b>{label}</b>, with Highest score: <b>{self.score_NN_Class5:0.1f}</b>%"
                        )
                        self.image_label2.setStyleSheet(
                            f"color: white; background-color: {color.name(QColor.HexArgb)};"
                        )
                        break


                self.item11 = QTableWidgetItem(f"{100*self.predict_probability_NN[0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item11.setBackground(QBrush(color))
                self.item11.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 0, self.item11)
                
                self.item12 = QTableWidgetItem(f"{100*self.predict_probability_NN[1]:0.1f}%")
                color = QColor(*(100, 200, 0))
                color.setAlpha(140)
                self.item12.setBackground(QBrush(color))
                self.item12.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 1, self.item12)
                
                self.item13 = QTableWidgetItem(f"{100*self.predict_probability_NN[2]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                self.item13.setBackground(QBrush(color))
                self.item13.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 2, self.item13)
                
                self.item14 = QTableWidgetItem(f"{100*self.predict_probability_NN[3]:0.1f}%")
                color = QColor(*(227, 100, 0))
                color.setAlpha(140)
                self.item14.setBackground(QBrush(color))
                self.item14.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 3, self.item14)
                
                self.item15 = QTableWidgetItem(f"{100*self.predict_probability_NN[4]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item15.setBackground(QBrush(color))
                self.item15.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 4, self.item15)
            
            else:                
                self.table4.setItem(0, 0, QTableWidgetItem(""))
                self.table4.setItem(0, 1, QTableWidgetItem(""))
                self.table4.setItem(0, 2, QTableWidgetItem(""))
                self.table4.setItem(0, 3, QTableWidgetItem(""))
                self.table4.setItem(0, 4, QTableWidgetItem(""))
                
                pass
                self.table4.setVisible(False)
                self.table_layout_VBOX_Widget.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table4.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                if self.AI_Automated_CAD_Screen.not_detection_flag == 1:
                    pass
                    self.image_label2.clear()
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("Fused CVML and Ai")
        
        else:
            pass
            self.image_label2.setVisible(True)
            self.image_label2.clear()
            self.image_label2.setText("Fused CVML and Ai")
            self.table_layout_VBOX_Widget.setVisible(False)
            self.table3.setVisible(False)
            self.table4.setVisible(False)
            self.table3.clearContents()
            self.table4.clearContents()

#  _____________________________________________ AI_Automated_CAD Screen ____________________________________________________
class AI_Automated_CAD_Screen(QWidget):
    variableChanged_Equalize_Feature_Visualization = pyqtSignal(bool)
    variableChanged_Auto_mode = pyqtSignal(bool)
    variableChanged_Visualization = pyqtSignal(bool)
    
    def __init__(self, main_window):
        super().__init__()
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        # self.resize(1500, 1000)
        self.resize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setMinimumSize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setMaximumSize(screen_width, screen_height)
        
        self.screen_width = screen_width
        self.screen_height = screen_height
    
        self.main_window = main_window
        self.main_window.variableChanged2.connect(self.handle_variable_changed)
        self.main_window.VariableChanged_Combo_AI.connect(self.handle_variable_changed_SR)
        self.main_window.VariableChanged_Combo_Fused_to_AI.connect(self.handle_variable_changed_SR)
        
        
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen(self, None)
# ______________________________________________ initialization __________________________________________________
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.SecondSetVariable = 1
        self.not_detection_flag = 0
        
        self.load_pytorch_model_2C_Flag = 0
        self.load_pytorch_model_3C_Flag = 0
        self.load_pytorch_model_5C_Flag = 0
        self.load_pytorch_model_5C_Fusion_Flag = 0
        
        self.load_pytorch_model_2C_70_30_Flag = 0
        self.load_pytorch_model_3C_70_30_Flag = 0
        self.load_pytorch_model_5C_70_30_Flag = 0
        self.load_pytorch_model_5C_Fusion_70_30_Flag = 0
        
        self.load_pytorch_model_2C_60_40_Flag = 0
        self.load_pytorch_model_3C_60_40_Flag = 0
        self.load_pytorch_model_5C_60_40_Flag = 0
        self.load_pytorch_model_5C_60_40_Fusion_Flag = 0
        
        self.output2 = None
        self.class2 = None
        self.output3 = None
        self.class3 = None
        self.output5 = None
        self.class5 = None
        self.output5_Tree = None
        self.class5_Tree = None
        self.score_Class2 = None
        self.score_Class3 = None
        self.score_Class5 = None
        self.score_Class5_Tree = None
        
        self.SR = 1
        
        self.mouse_double_clicked = 0
        
# ______________________________________________ show_main_content __________________________________________________
        
        self.show_main_content()
    
    def show_main_content(self):
        self.image_label = ImageLabel(self, apply_effects = False)
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        
        # self.image_label.setFixedWidth(int(0.3125 * self.screen_width))
        self.image_label.setFixedSize(int(0.3645833 * self.screen_width), int(0.3125 * self.screen_width))
        self.image_label.updateEffectsBasedOnStyleAndPixmap()        
        
        # Create the table
        self.table = QTableWidget()
        self.table.setFixedSize(int(0.4244791667 * self.screen_width), int(0.15625 * self.screen_width))
        # self.table.setFixedHeight(int(0.15625 * self.screen_width))
        
        # self.table.setFixedSize(825, 300)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.setRowCount(4)
        self.table.setColumnCount(2)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
        
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table's fixed size
        table_width = self.table.width()
        table_height = self.table.height()
        row_height = table_height // self.table.rowCount()
        column_width = table_width // self.table.columnCount()

        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, row_height)
        
        for column in range(self.table.columnCount()):
            self.table.setColumnWidth(column, column_width)
        
        
        
        
        
        self.table1 = QTableWidget()
        self.table1.setFixedHeight(int(0.25 * table_height))
        self.table1.verticalHeader().setVisible(False)
        self.table1.horizontalHeader().setVisible(False)
        self.table1.setRowCount(1)
        self.table1.setColumnCount(2)
        self.table1.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
                
                
        self.table1.setShowGrid(False)
        self.table1.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table1.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table1's fixed size
        table1_width = self.table1.width()
        table1_height = self.table1.height()
        row1_height = table1_height // self.table1.rowCount()
        column1_width = table1_width // self.table1.columnCount()

        for row1 in range(self.table1.rowCount()):
            self.table1.setRowHeight(row1, row1_height)
        
        for column1 in range(self.table1.columnCount()):
            self.table1.setColumnWidth(column1, column1_width)
        
        self.table2 = QTableWidget()
        self.table2.setFixedHeight(int(0.25 * table_height))
        self.table2.verticalHeader().setVisible(False)
        self.table2.horizontalHeader().setVisible(False)
        self.table2.setRowCount(1)
        self.table2.setColumnCount(3)
        self.table2.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
                
        self.table2.setShowGrid(False)
        self.table2.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table2.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table2's fixed size
        table2_width = self.table2.width()
        table2_height = self.table2.height()
        row2_height = table2_height // self.table2.rowCount()
        column2_width = table2_width // self.table2.columnCount()

        for row2 in range(self.table2.rowCount()):
            self.table2.setRowHeight(row2, row2_height)
        
        for column2 in range(self.table2.columnCount()):
            self.table2.setColumnWidth(column2, column2_width)
        
        
        self.table3 = QTableWidget()
        self.table3.setFixedHeight(int(0.25 * table_height))
        self.table3.verticalHeader().setVisible(False)
        self.table3.horizontalHeader().setVisible(False)
        self.table3.setRowCount(1)
        self.table3.setColumnCount(5)
        self.table3.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
        self.table3.setShowGrid(False)
        self.table3.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table3.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table3's fixed size
        table3_width = self.table3.width()
        table3_height = self.table3.height()
        row3_height = table3_height // self.table3.rowCount()
        column3_width = table3_width // self.table3.columnCount()

        for row3 in range(self.table3.rowCount()):
            self.table3.setRowHeight(row3, row3_height)
        
        for column3 in range(self.table3.columnCount()):
            self.table3.setColumnWidth(column3, column3_width)


        
        
        self.table4 = QTableWidget()
        self.table4.setFixedHeight(int(0.25 * table_height))
        self.table4.verticalHeader().setVisible(False)
        self.table4.horizontalHeader().setVisible(False)
        self.table4.setRowCount(1)
        self.table4.setColumnCount(5)
        self.table4.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
        self.table4.setShowGrid(False)
        self.table4.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table4.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table4's fixed size
        table4_width = self.table4.width()
        table4_height = self.table4.height()
        row4_height = table4_height // self.table4.rowCount()
        column4_width = table4_width // self.table4.columnCount()

        for row4 in range(self.table4.rowCount()):
            self.table4.setRowHeight(row4, row4_height)
        
        for column4 in range(self.table4.columnCount()):
            self.table4.setColumnWidth(column4, column4_width)
            
        self.image_label2 = ImageLabel("AI Automated", apply_effects = False)
        self.image_label2.setFixedWidth(int(0.4244791667 * self.screen_width))
        self.image_label2.setFixedHeight(int(0.25 * table_height))
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setScaledContents(True)
        self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
        
            
        font = QFont()
        font.setPointSize(int(0.009895833 * self.screen_width))
        font.setBold(False)
        self.image_label2.setFont(font)
        self.table.setFont(font)
        self.table1.setFont(font)
        self.table2.setFont(font)
        self.table3.setFont(font)
        self.table4.setFont(font)
        
        self.table_layout_VBOX = QVBoxLayout()
        self.table_layout_VBOX.setSpacing(0)
        self.table_layout_VBOX.setContentsMargins(0, 0, 0, 0)
        
        self.table_layout_VBOX.addWidget(self.table1)
        self.table_layout_VBOX.addWidget(self.table2)
        self.table_layout_VBOX.addWidget(self.table3)
        self.table_layout_VBOX.addWidget(self.table4)


        self.table_layout_VBOX_Widget = QWidget()

        self.table_layout_VBOX_Widget.setFixedHeight(table_height)
        self.table_layout_VBOX_Widget.setFixedWidth(int(2.15 *table_height))
        
        self.table_layout_VBOX_Widget.setLayout(self.table_layout_VBOX)
        self.table_layout_VBOX_Widget.setStyleSheet("background-color: transparent; color: rgba(254,229,2,255);")
        
        
        self.table.setVisible(False)
        
        self.table_layout_VBOX_Widget.setVisible(False)
        self.table1.setVisible(False)
        self.table2.setVisible(False)
        self.table3.setVisible(False)
        self.table4.setVisible(False)
        
          
        
        spacerdashed1 = QSpacerItem(int(0.002604166 * self.screen_width), 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacerdashed2 = QSpacerItem(int(0.002604166 * self.screen_width), 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        self.table_layout = QHBoxLayout()
        self.table_layout.setSpacing(0)
        self.table_layout.setContentsMargins(0, 0, 0, 0)
        self.table_layout.addSpacerItem(spacerdashed1)
        self.table_layout.addWidget(self.image_label2)
        self.table_layout.addWidget(self.table)
        self.table_layout.addWidget(self.table_layout_VBOX_Widget)
        self.table_layout.addSpacerItem(spacerdashed2)

        self.table_layout_Widget = QWidget()
        self.table_layout_Widget.setLayout(self.table_layout)
        self.table_layout_Widget.setStyleSheet("background-color: transparent; color: rgba(254,229,2,255);")
        self.table_layout_Widget.setVisible(True)
        
        
        HBoxLayout1 = QHBoxLayout()
        HBoxLayout1.addWidget(self.image_label)
        VBoxLayout1 = QVBoxLayout()
        VBoxLayout1.addLayout(HBoxLayout1)
        VBoxLayout1.addWidget(self.table_layout_Widget)
        self.setLayout(VBoxLayout1)
# __________________________________________________ Functions _______________________________________________________
    def handle_variable_changed_SR(self, new_value):
        self.SR = new_value
        print("SR", self.SR)
        
        if self.main_window.clasify_indicator == 1:
            self.show_predictions()
        else:
            pass
            print(f"clasify_indicator {self.main_window.clasify_indicator}")
            
        
    def handle_variable_changed(self, new_value):
        self.SecondSetVariable = new_value
        # print("SecondSetVariable", self.SecondSetVariable)
    
            
    def mouseDoubleClickEvent(self, event):
        # self.mouse_double_clicked = 1
        self.main_window.load_main_img()
    
    def load_pytorch_model_2C(self):
        if not hasattr(self, 'pytorch_model_2C'):
            self.pytorch_model_2C = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_2C_80_5_15.pt', 2)
        self.load_pytorch_model_2C_Flag = 1
    
    def load_pytorch_model_3C(self):
        if not hasattr(self, 'pytorch_model_3C'):
            self.pytorch_model_3C = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_3C_80_5_15.pt', 3)
        self.load_pytorch_model_3C_Flag = 1

    def load_pytorch_model_5C(self):
        if not hasattr(self, 'pytorch_model_5C'):
                self.pytorch_model_5C = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_5C_80_5_15.pt', 5)
        self.load_pytorch_model_5C_Flag = 1


    def load_pytorch_model_5C_Fusion(self):
        if not hasattr(self, 'pytorch_model_0_1'):
            self.pytorch_model_0_1 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_0vs1_80_5_15.pt',2)

        if not hasattr(self, 'pytorch_model_0_2'):
            self.pytorch_model_0_2 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_0vs2_80_5_15.pt',2)

        if not hasattr(self, 'pytorch_model_0_3'):
            self.pytorch_model_0_3 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_0vs3_80_5_15.pt',2)

        if not hasattr(self, 'pytorch_model_0_4'):
            self.pytorch_model_0_4 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_0vs4_80_5_15.pt',2)
        
        if not hasattr(self, 'pytorch_model_1_2'):
            self.pytorch_model_1_2 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_1vs2_80_5_15.pt',2)

        if not hasattr(self, 'pytorch_model_1_3'):
            self.pytorch_model_1_3 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_1vs3_80_5_15.pt',2)

        if not hasattr(self, 'pytorch_model_1_4'):
            self.pytorch_model_1_4 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_1vs4_80_5_15.pt',2)

        if not hasattr(self, 'pytorch_model_2_3'):
            self.pytorch_model_2_3 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_2vs3_80_5_15.pt',2)
        
        if not hasattr(self, 'pytorch_model_2_4'):
            self.pytorch_model_2_4 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_2vs4_80_5_15.pt',2)
        
        if not hasattr(self, 'pytorch_model_3_4'):
            self.pytorch_model_3_4 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch/Best_DenseNet201_model_case5_3vs4_80_5_15.pt',2)
        
        self.load_pytorch_model_5C_Fusion_Flag = 1

    
    
    
    
    def load_pytorch_model_2C_70_30(self):
        if not hasattr(self, 'pytorch_model_2C_70_30'):
            self.pytorch_model_2C_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_2C_70_10_20.pt', 2)
        self.load_pytorch_model_2C_70_30_Flag = 1
    
    def load_pytorch_model_3C_70_30(self):
        if not hasattr(self, 'pytorch_model_3C_70_30'):
            self.pytorch_model_3C_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_3C_70_10_20.pt', 3)
        self.load_pytorch_model_3C_70_30_Flag = 1

    def load_pytorch_model_5C_70_30(self):
        if not hasattr(self, 'pytorch_model_5C_70_30'):
                self.pytorch_model_5C_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_5C_70_10_20.pt', 5)
        self.load_pytorch_model_5C_70_30_Flag = 1


    def load_pytorch_model_5C_Fusion_70_30(self):
        if not hasattr(self, 'pytorch_model_0_1_70_30'):
            self.pytorch_model_0_1_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_0vs1_70_10_20.pt',2)

        if not hasattr(self, 'pytorch_model_0_2_70_30'):
            self.pytorch_model_0_2_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_0vs2_70_10_20.pt',2)

        if not hasattr(self, 'pytorch_model_0_3_70_30'):
            self.pytorch_model_0_3_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_0vs3_70_10_20.pt',2)

        if not hasattr(self, 'pytorch_model_0_4_70_30'):
            self.pytorch_model_0_4_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_0vs4_70_10_20.pt',2)
        
        if not hasattr(self, 'pytorch_model_1_2_70_30'):
            self.pytorch_model_1_2_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_1vs2_70_10_20.pt',2)

        if not hasattr(self, 'pytorch_model_1_3_70_30'):
            self.pytorch_model_1_3_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_1vs3_70_10_20.pt',2)

        if not hasattr(self, 'pytorch_model_1_4_70_30'):
            self.pytorch_model_1_4_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_1vs4_70_10_20.pt',2)

        if not hasattr(self, 'pytorch_model_2_3_70_30'):
            self.pytorch_model_2_3_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_2vs3_70_10_20.pt',2)
        
        if not hasattr(self, 'pytorch_model_2_4_70_30'):
            self.pytorch_model_2_4_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_2vs4_70_10_20.pt',2)
        
        if not hasattr(self, 'pytorch_model_3_4_70_30'):
            self.pytorch_model_3_4_70_30 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 70-30/Best_DenseNet201_model_case5_3vs4_70_10_20.pt',2)
        
        self.load_pytorch_model_5C_Fusion_70_30_Flag = 1
        
        
        
        
        
        
    def load_pytorch_model_2C_60_40(self):
        if not hasattr(self, 'pytorch_model_2C_60_40'):
            self.pytorch_model_2C_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_2C_60_15_25.pt', 2)
        self.load_pytorch_model_2C_60_40_Flag = 1
    
    def load_pytorch_model_3C_60_40(self):
        if not hasattr(self, 'pytorch_model_3C_60_40'):
            self.pytorch_model_3C_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_3C_60_15_25.pt', 3)
        self.load_pytorch_model_3C_60_40_Flag = 1

    def load_pytorch_model_5C_60_40(self):
        if not hasattr(self, 'pytorch_model_5C_60_40'):
                self.pytorch_model_5C_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_5C_60_15_25.pt', 5)
        self.load_pytorch_model_5C_60_40_Flag = 1


    def load_pytorch_model_5C_60_40_Fusion(self):
        if not hasattr(self, 'pytorch_model_0_1_60_40'):
            self.pytorch_model_0_1_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_0vs1_60_15_25.pt',2)

        if not hasattr(self, 'pytorch_model_0_2_60_40'):
            self.pytorch_model_0_2_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_0vs2_60_15_25.pt',2)

        if not hasattr(self, 'pytorch_model_0_3_60_40'):
            self.pytorch_model_0_3_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_0vs3_60_15_25.pt',2)

        if not hasattr(self, 'pytorch_model_0_4_60_40'):
            self.pytorch_model_0_4_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_0vs4_60_15_25.pt',2)
        
        if not hasattr(self, 'pytorch_model_1_2_60_40'):
            self.pytorch_model_1_2_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_1vs2_60_15_25.pt',2)

        if not hasattr(self, 'pytorch_model_1_3_60_40'):
            self.pytorch_model_1_3_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_1vs3_60_15_25.pt',2)

        if not hasattr(self, 'pytorch_model_1_4_60_40'):
            self.pytorch_model_1_4_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_1vs4_60_15_25.pt',2)

        if not hasattr(self, 'pytorch_model_2_3_60_40'):
            self.pytorch_model_2_3_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_2vs3_60_15_25.pt',2)
        
        if not hasattr(self, 'pytorch_model_2_4_60_40'):
            self.pytorch_model_2_4_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_2vs4_60_15_25.pt',2)
        
        if not hasattr(self, 'pytorch_model_3_4_60_40'):
            self.pytorch_model_3_4_60_40 = self.load_pytorch_model('Models/New Ai Automated Models/PyTorch 60-40/Best_DenseNet201_model_case5_3vs4_60_15_25.pt',2)
        
        self.load_pytorch_model_5C_60_40_Fusion_Flag = 1
        
        
        
    
    def carry_out_Ai_CAD(self):
    
        if self.SR == 1:
            if self.load_pytorch_model_2C_Flag == 0:
                self.load_pytorch_model_2C()
                
            if self.load_pytorch_model_3C_Flag == 0:
                self.load_pytorch_model_3C()
                
            if self.load_pytorch_model_5C_Flag == 0:
                self.load_pytorch_model_5C()
                
            if self.load_pytorch_model_5C_Fusion_Flag == 0:
                self.load_pytorch_model_5C_Fusion()
                
            
            self.output2, self.class2 = self.pytorch_classifier(self.file_path, 1, 1)
            self.output3, self.class3 = self.pytorch_classifier(self.file_path, 2, 1)
            self.output5, self.class5 = self.pytorch_classifier(self.file_path, 3, 1)
            self.output5_Tree, self.class5_Tree = self.pytorch_classifier(self.file_path, 4, 1)
        
        
        if self.SR == 2:
            
            if self.load_pytorch_model_2C_70_30_Flag == 0:
                self.load_pytorch_model_2C_70_30()
                
            if self.load_pytorch_model_3C_70_30_Flag == 0:
                self.load_pytorch_model_3C_70_30()
                
            if self.load_pytorch_model_5C_70_30_Flag == 0:
                self.load_pytorch_model_5C_70_30()
                
            if self.load_pytorch_model_5C_Fusion_70_30_Flag == 0:
                self.load_pytorch_model_5C_Fusion_70_30()
                
            
                
            self.output2, self.class2 = self.pytorch_classifier(self.file_path, 1, 2)
            self.output3, self.class3 = self.pytorch_classifier(self.file_path, 2, 2)
            self.output5, self.class5 = self.pytorch_classifier(self.file_path, 3, 2)
            self.output5_Tree, self.class5_Tree = self.pytorch_classifier(self.file_path, 4, 2)
                
        
        
        if self.SR == 3:
                
            if self.load_pytorch_model_2C_60_40_Flag == 0:
                self.load_pytorch_model_2C_60_40()
                
            if self.load_pytorch_model_3C_60_40_Flag == 0:
                self.load_pytorch_model_3C_60_40()
                
            if self.load_pytorch_model_5C_60_40_Flag == 0:
                self.load_pytorch_model_5C_60_40()
                
            if self.load_pytorch_model_5C_60_40_Fusion_Flag == 0:
                self.load_pytorch_model_5C_60_40_Fusion()
            
            
            self.output2, self.class2 = self.pytorch_classifier(self.file_path, 1, 3)
            self.output3, self.class3 = self.pytorch_classifier(self.file_path, 2, 3)
            self.output5, self.class5 = self.pytorch_classifier(self.file_path, 3, 3)
            self.output5_Tree, self.class5_Tree = self.pytorch_classifier(self.file_path, 4, 3)
        
        
        
        print(f"{self.class2}, {self.output2},{self.class3}, {self.output3},{self.class5}, {self.output5}, {self.class5_Tree}, {self.output5_Tree}")
        
        self.score_Class2 = 100*self.output2[self.class2]
        self.score_Class3 = 100*self.output3[self.class3]
        self.score_Class5 = 100*self.output5[self.class5]
        self.score_Class5_Tree = 100*self.output5_Tree[self.class5_Tree]
        
        self.show_predictions()
   
   
   
    def show_predictions(self):
        if self.SecondSetVariable == 1:
            if self.class2 is not None:
                                    
                self.table.setVisible(False)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(False)
                self.table3.setVisible(False)
                self.table4.setVisible(False)
                self.image_label2.setVisible(True)
                
                descriptions = ["Normal", "OsteoArthrits"]
                colors = [(0, 200, 0, 140), (255, 0, 0, 140)]
                
                self.image_label2.setText(f"Normal Vs OsteoArthritis: <b>{descriptions[self.class2]}</b>, with Highest score: <b>{self.score_Class2:0.1f}</b>%")
                color = QColor(*colors[self.class2])
                self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                
                
                self.item1 = QTableWidgetItem(f"{100*self.output2[0]:0.1f}%")
                self.item1.setTextAlignment(Qt.AlignCenter)
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item1.setBackground(QBrush(color))
                self.table1.setItem(0, 0, self.item1)
                
                
                self.item2 = QTableWidgetItem(f"{100*self.output2[1]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item2.setBackground(QBrush(color))
                self.item2.setTextAlignment(Qt.AlignCenter)
                self.table1.setItem(0, 1, self.item2)
                        
            else:
                self.table.setItem(0, 1, QTableWidgetItem(""))
                
                self.table1.setItem(0, 0, QTableWidgetItem(""))
                self.table1.setItem(0, 1, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table1.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table1.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                if self.not_detection_flag == 1:
                    pass
                    self.image_label2.clear()
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("AI Automated")
                    
                
        elif self.SecondSetVariable == 2:
            if self.class3 is not None:
                self.table.setVisible(False)
            
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(False)
                self.table2.setVisible(True)
                self.table3.setVisible(False)
                self.table4.setVisible(False)
                
                self.image_label2.setVisible(True)
                
                class_labels = ["Normal", "Mild", "Severe"]
                colors = [
                    QColor(0, 200, 0, 140),
                    QColor(200, 200, 0, 140),
                    QColor(255, 0, 0, 140)
                ]

                
                for i in range(3):
                    if self.class3 == i:
                        self.image_label2.setText(f"Normal Vs Mild Vs Severe: <b>{class_labels[i]}</b>, with Highest score: <b>{self.score_Class3:0.1f}</b>%")
                        color = colors[i]
                        self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                        break

                self.item3 = QTableWidgetItem(f"{100*self.output3[0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item3.setBackground(QBrush(color))
                self.item3.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 0, self.item3)

                self.item4 = QTableWidgetItem(f"{100*self.output3[1]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                self.item4.setBackground(QBrush(color))
                self.item4.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 1, self.item4)
                
                self.item5 = QTableWidgetItem(f"{100*self.output3[2]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item5.setBackground(QBrush(color))
                self.item5.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 2, self.item5)

            else:
                self.table.setItem(1, 1, QTableWidgetItem(""))
                
                self.table2.setItem(0, 0, QTableWidgetItem(""))
                self.table2.setItem(0, 1, QTableWidgetItem(""))
                self.table2.setItem(0, 2, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table2.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table2.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                if self.not_detection_flag == 1:
                    pass
                    self.image_label2.clear()
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("AI Automated")
                
                
        elif self.SecondSetVariable == 3:
            if self.class5 is not None:
                self.table.setVisible(False)
                
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(False)
                self.table2.setVisible(False)
                self.table3.setVisible(True)
                self.table4.setVisible(False)
                self.image_label2.setVisible(True)
            
            
                class_labels = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
                colors = [
                    QColor(0, 200, 0, 140),
                    QColor(100, 200, 0, 140),
                    QColor(200, 200, 0, 140),
                    QColor(227, 100, 0, 140),
                    QColor(255, 0, 0, 140)
                ]

                for i in range(5):
                    if self.class5 == i:
                        self.image_label2.setText(f"Kellgren-Lawrence 5-Classes: <b>{class_labels[i]}</b>, with Highest score: <b>{self.score_Class5:0.1f}</b>%")
                        color = colors[i]
                        self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                        break

                self.item6 = QTableWidgetItem(f"{100*self.output5[0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item6.setBackground(QBrush(color))
                self.item6.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 0, self.item6)
                
                self.item7 = QTableWidgetItem(f"{100*self.output5[1]:0.1f}%")
                color = QColor(*(100, 200, 0))
                color.setAlpha(140)
                self.item7.setBackground(QBrush(color))
                self.item7.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 1, self.item7)
                
                
                self.item8 = QTableWidgetItem(f"{100*self.output5[2]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                self.item8.setBackground(QBrush(color))
                self.item8.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 2, self.item8)
                
                
                self.item9 = QTableWidgetItem(f"{100*self.output5[3]:0.1f}%")
                color = QColor(*(227, 100, 0))
                color.setAlpha(140)
                self.item9.setBackground(QBrush(color))
                self.item9.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 3, self.item9)
                
                
                
                
                self.item10 = QTableWidgetItem(f"{100*self.output5[4]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item10.setBackground(QBrush(color))
                self.item10.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 4, self.item10)
            
            else:
                self.table.setItem(2, 1, QTableWidgetItem(""))
                
                self.table3.setItem(0, 0, QTableWidgetItem(""))
                self.table3.setItem(0, 1, QTableWidgetItem(""))
                self.table3.setItem(0, 2, QTableWidgetItem(""))
                self.table3.setItem(0, 3, QTableWidgetItem(""))
                self.table3.setItem(0, 4, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table3.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table3.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                if self.not_detection_flag == 1:
                    pass
                    self.image_label2.clear()
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("AI Automated")
            
                
        elif self.SecondSetVariable == 4:
            if self.class5_Tree is not None:
                self.table.setVisible(False)
                
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(False)
                self.table2.setVisible(False)
                self.table3.setVisible(False)
                self.table4.setVisible(True)
                self.image_label2.setVisible(True)
            
                class_labels = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
                colors = [
                    QColor(0, 200, 0, 140),
                    QColor(100, 200, 0, 140),
                    QColor(200, 200, 0, 140),
                    QColor(227, 100, 0, 140),
                    QColor(255, 0, 0, 140)
                ]

                for i in range(5):
                    if self.class5_Tree == i:
                        self.image_label2.setText(f"Probability voting 5-Classes: <b>{class_labels[i]}</b>, with Highest score: <b>{self.score_Class5_Tree:0.1f}</b>%")
                        color = colors[i]
                        self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                        break
                
                self.item11 = QTableWidgetItem(f"{100*self.output5_Tree[0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item11.setBackground(QBrush(color))
                self.item11.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 0, self.item11)
                
                
                self.item12 = QTableWidgetItem(f"{100*self.output5_Tree[1]:0.1f}%")
                color = QColor(*(100, 200, 0))
                color.setAlpha(140)
                self.item12.setBackground(QBrush(color))
                self.item12.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 1, self.item12)
                
                
                
                self.item13 = QTableWidgetItem(f"{100*self.output5_Tree[2]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                self.item13.setBackground(QBrush(color))
                self.item13.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 2, self.item13)
                
                
                self.item14 = QTableWidgetItem(f"{100*self.output5_Tree[3]:0.1f}%")
                color = QColor(*(227, 100, 0))
                color.setAlpha(140)
                self.item14.setBackground(QBrush(color))
                self.item14.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 3, self.item14)
                
                
                self.item15 = QTableWidgetItem(f"{100*self.output5_Tree[4]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item15.setBackground(QBrush(color))
                self.item15.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 4, self.item15)
                    
            else:
                
                self.table4.setItem(0, 0, QTableWidgetItem(""))
                self.table4.setItem(0, 1, QTableWidgetItem(""))
                self.table4.setItem(0, 2, QTableWidgetItem(""))
                self.table4.setItem(0, 3, QTableWidgetItem(""))
                self.table4.setItem(0, 4, QTableWidgetItem(""))
                    
                pass
                self.table.setVisible(False)
                self.table4.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                
                self.table.clearContents()
                self.table4.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                
                if self.not_detection_flag == 1:
                    pass
                    self.image_label2.clear()
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("AI Automated")
                
        
        elif self.SecondSetVariable == 5:
            
            self.item_main1 = QTableWidgetItem("Normal Vs OsteoArthritis")
            self.table.setItem(0, 0, self.item_main1)
            self.item_main2 = QTableWidgetItem("Normal Vs Mild Vs Severe")
            self.table.setItem(1, 0, self.item_main2)
            self.item_main3 = QTableWidgetItem("Kellgren-Lawrence 5-Classes")
            self.table.setItem(2, 0, self.item_main3)
            self.item_main4 = QTableWidgetItem("Probability voting 5-Classes")
            self.table.setItem(3, 0, self.item_main4)
    
            if (self.class5_Tree is not None):
                
                self.table.setVisible(True)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(True)
                
                self.image_label2.setVisible(False)
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
            
                colors = [(0, 200, 0), (100, 200, 0),(200, 200, 0), (227, 100, 0), (255, 0, 0)]
                descriptions = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
                
                if 0 <= self.class5_Tree < len(descriptions):
                    item_tree = QTableWidgetItem(f"{descriptions[self.class5_Tree]}, with Highest score: {self.score_Class5_Tree:0.1f}%")
                    color = QColor(*colors[self.class5_Tree])
                    color.setAlpha(140)
                    item_tree.setBackground(QBrush(color))
                    self.table.setItem(3, 1, item_tree)
                    
                    self.item_main4.setBackground(QBrush(color))
                    self.item_main4.setTextAlignment(Qt.AlignCenter)
                    
                    item11 = QTableWidgetItem(f"{100*self.output5_Tree[0]:0.1f}%")
                    color = QColor(*(0, 200, 0))
                    color.setAlpha(140)
                    item11.setBackground(QBrush(color))
                    item11.setTextAlignment(Qt.AlignCenter)
                    self.table4.setItem(0, 0, item11)
                
                    item12 = QTableWidgetItem(f"{100*self.output5_Tree[1]:0.1f}%")
                    color = QColor(*(100, 200, 0))
                    color.setAlpha(140)
                    item12.setBackground(QBrush(color))
                    item12.setTextAlignment(Qt.AlignCenter)
                    self.table4.setItem(0, 1, item12)
                
                    item13 = QTableWidgetItem(f"{100*self.output5_Tree[2]:0.1f}%")
                    color = QColor(*(200, 200, 0))
                    color.setAlpha(140)
                    item13.setBackground(QBrush(color))
                    item13.setTextAlignment(Qt.AlignCenter)
                    self.table4.setItem(0, 2, item13)
                    
                    item14 = QTableWidgetItem(f"{100*self.output5_Tree[3]:0.1f}%")
                    color = QColor(*(227, 100, 0))
                    color.setAlpha(140)
                    item14.setBackground(QBrush(color))
                    item14.setTextAlignment(Qt.AlignCenter)
                    self.table4.setItem(0, 3, item14)
                
                    item15 = QTableWidgetItem(f"{100*self.output5_Tree[4]:0.1f}%")
                    color = QColor(*(255, 0, 0))
                    color.setAlpha(140)
                    item15.setBackground(QBrush(color))
                    item15.setTextAlignment(Qt.AlignCenter)
                    self.table4.setItem(0, 4, item15)

            else:
                self.table.setItem(3, 1, QTableWidgetItem(""))
                
                self.table4.setItem(0, 0, QTableWidgetItem(""))
                self.table4.setItem(0, 1, QTableWidgetItem(""))
                self.table4.setItem(0, 2, QTableWidgetItem(""))
                self.table4.setItem(0, 3, QTableWidgetItem(""))
                self.table4.setItem(0, 4, QTableWidgetItem(""))
                    
                pass
                self.table.setVisible(False)
                self.table4.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                self.table.clearContents()
                self.table4.clearContents()
                
                if self.not_detection_flag == 1:
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("AI Automated")
                
                
            if (self.class5 is not None):
                
                self.table.setVisible(True)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(True)
                
                self.image_label2.setVisible(False)
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                colors = [(0, 200, 0), (100, 200, 0),(200, 200, 0), (227, 100, 0), (255, 0, 0)]
                descriptions = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
                
                if self.class5 in range(len(descriptions)):
                    item_class5 = QTableWidgetItem(f"{descriptions[self.class5]}, with Highest score: {self.score_Class5:0.1f}%")
                    color = QColor(*colors[self.class5])
                    color.setAlpha(140)
                    item_class5.setBackground(QBrush(color))
                    self.table.setItem(2, 1, item_class5)
                    
                    self.item_main3.setBackground(QBrush(color))
                    self.item_main3.setTextAlignment(Qt.AlignCenter)
                    
                    
                    item6 = QTableWidgetItem(f"{100*self.output5[0]:0.1f}%")
                    color = QColor(*(0, 200, 0))
                    color.setAlpha(140)
                    item6.setBackground(QBrush(color))
                    item6.setTextAlignment(Qt.AlignCenter)
                    self.table3.setItem(0, 0, item6)
                
                
                    item7 = QTableWidgetItem(f"{100*self.output5[1]:0.1f}%")
                    color = QColor(*(100, 200, 0))
                    color.setAlpha(140)
                    item7.setBackground(QBrush(color))
                    item7.setTextAlignment(Qt.AlignCenter)
                    self.table3.setItem(0, 1, item7)
                    
                    item8 = QTableWidgetItem(f"{100*self.output5[2]:0.1f}%")
                    color = QColor(*(200, 200, 0))
                    color.setAlpha(140)
                    item8.setBackground(QBrush(color))
                    item8.setTextAlignment(Qt.AlignCenter)
                    self.table3.setItem(0, 2, item8)
                
                    item9 = QTableWidgetItem(f"{100*self.output5[3]:0.1f}%")
                    color = QColor(*(227, 100, 0))
                    color.setAlpha(140)
                    item9.setBackground(QBrush(color))
                    item9.setTextAlignment(Qt.AlignCenter)
                    self.table3.setItem(0, 3, item9)
                    
                    item10 = QTableWidgetItem(f"{100*self.output5[4]:0.1f}%")
                    color = QColor(*(255, 0, 0))
                    color.setAlpha(140)
                    item10.setBackground(QBrush(color))
                    item10.setTextAlignment(Qt.AlignCenter)
                    self.table3.setItem(0, 4, item10)
                
            else:
                self.table.setItem(2, 1, QTableWidgetItem(""))
                
                self.table3.setItem(0, 0, QTableWidgetItem(""))
                self.table3.setItem(0, 1, QTableWidgetItem(""))
                self.table3.setItem(0, 2, QTableWidgetItem(""))
                self.table3.setItem(0, 3, QTableWidgetItem(""))
                self.table3.setItem(0, 4, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table3.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table3.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                if self.not_detection_flag == 1:
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("AI Automated")
                
            # Update table for class3
            if (self.class3 is not None):
                
                self.table.setVisible(True)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(True)
                
                self.image_label2.setVisible(False)
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                colors = [(0, 200, 0), (200, 200, 0), (255, 0, 0)]
                descriptions = ["Normal", "Mild", "Severe"]
                
                if self.class3 in range(len(descriptions)):
                    item_class3 = QTableWidgetItem(f"{descriptions[self.class3]}, with Highest score: {self.score_Class3:0.1f}%")
                    color = QColor(*colors[self.class3])
                    color.setAlpha(140)
                    item_class3.setBackground(QBrush(color))
                    self.table.setItem(1, 1, item_class3)
                    
                    self.item_main2.setBackground(QBrush(color))
                    self.item_main2.setTextAlignment(Qt.AlignCenter)
                    
                    
                    item3 = QTableWidgetItem(f"{100*self.output3[0]:0.1f}%")
                    color = QColor(*(0, 200, 0))
                    color.setAlpha(140)
                    item3.setBackground(QBrush(color))
                    item3.setTextAlignment(Qt.AlignCenter)
                    self.table2.setItem(0, 0, item3)
                
                    item4 = QTableWidgetItem(f"{100*self.output3[1]:0.1f}%")
                    color = QColor(*(200, 200, 0))
                    color.setAlpha(140)
                    item4.setBackground(QBrush(color))
                    item4.setTextAlignment(Qt.AlignCenter)
                    self.table2.setItem(0, 1, item4)
                
                    item5 = QTableWidgetItem(f"{100*self.output3[2]:0.1f}%")
                    color = QColor(*(255, 0, 0))
                    color.setAlpha(140)
                    item5.setBackground(QBrush(color))
                    item5.setTextAlignment(Qt.AlignCenter)
                    self.table2.setItem(0, 2, item5)
                
            else:
                self.table.setItem(1, 1, QTableWidgetItem(""))
                
                self.table2.setItem(0, 0, QTableWidgetItem(""))
                self.table2.setItem(0, 1, QTableWidgetItem(""))
                self.table2.setItem(0, 2, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table2.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table2.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                if self.not_detection_flag == 1:
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("AI Automated")
                
                
            # Update table for class2
            if (self.class2 is not None):
                
                self.table.setVisible(True)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(True)
                
                self.image_label2.setVisible(False)
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                colors = [(0, 200, 0), (255, 0, 0)]
                descriptions = ["Normal", "Osteo"]
                
                if self.class2 in range(len(descriptions)):
                    item_clas2 = QTableWidgetItem(f"{descriptions[self.class2]}, with Highest score: {self.score_Class2:0.1f}%")
                    color = QColor(*colors[self.class2])
                    color.setAlpha(140)
                    item_clas2.setBackground(QBrush(color))
                    self.table.setItem(0, 1, item_clas2)
                    
                    self.item_main1.setBackground(QBrush(color))
                    self.item_main1.setTextAlignment(Qt.AlignCenter)
                    
                    item1 = QTableWidgetItem(f"{100*self.output2[0]:0.1f}%")
                    color = QColor(*(0, 200, 0))
                    color.setAlpha(140)
                    item1.setBackground(QBrush(color))
                    item1.setTextAlignment(Qt.AlignCenter)
                    self.table1.setItem(0, 0, item1)
                    
                    item2 = QTableWidgetItem(f"{100*self.output2[1]:0.1f}%")
                    color = QColor(*(255, 0, 0))
                    color.setAlpha(140)
                    item2.setBackground(QBrush(color))
                    item2.setTextAlignment(Qt.AlignCenter)
                    self.table1.setItem(0, 1, item2)  
                    
            else:
                self.table.setItem(0, 1, QTableWidgetItem(""))
                
                self.table1.setItem(0, 0, QTableWidgetItem(""))
                self.table1.setItem(0, 1, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table1.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table1.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                if self.not_detection_flag == 1:
                    self.image_label2.setText("Countour can not be detected. Try with another image")
                else:
                    self.image_label2.setText("AI Automated")
               
        else:
            pass    
            self.table.setVisible(False)
            
            self.table1.setVisible(False)
            self.table2.setVisible(False)
            self.table3.setVisible(False)
            self.table4.setVisible(False)
            
            self.image_label2.setVisible(True)
            self.table.clearContents()
            self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
            
            self.table1.clearContents()
            self.table2.clearContents()
            self.table3.clearContents()
            self.table4.clearContents()
            
            self.image_label2.clear()
            self.image_label2.setText("AI Automated")
            
                         # ___________ Ai Automated CAD Models ____________#

    def load_pytorch_model(self, path, num_classes):
        model = densenet201(weights = None)
        classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier.in_features, num_classes),
            nn.Softmax(dim=1) if num_classes > 2 else nn.Sigmoid()
        )
        model.classifier = classifier

        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(path,map_location=device))
        model.eval()
        return model


    def pytorch_classifier(self, image_path, classification_approach, split_ratio_called):
        if not hasattr(self.Feature_Extraction_and_Visualization_Screen, 'old_model_detection'):
            self.Feature_Extraction_and_Visualization_Screen.old_model_detection =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_ROI detection.pt")
  
        if not hasattr(self.Feature_Extraction_and_Visualization_Screen, 'new_model_detection'):
            self.Feature_Extraction_and_Visualization_Screen.new_model_detection =YOLO("Models/NewFeatureExtractionAndVisualizationModels/best.pt")
        
        image = cv2.imread(image_path)
        # ROI detection and croping
        if image.shape[0] != image.shape[1]:
            
            image = self.Feature_Extraction_and_Visualization_Screen.Apply_Padding(image)
            
            ROI_prediction = self.Feature_Extraction_and_Visualization_Screen.new_model_detection( cv2.cvtColor( cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR ))
            
            x1,y1,x2,y2 = self.Feature_Extraction_and_Visualization_Screen.get_coordinate_Predict_image(ROI_prediction)[0][0:4]
            image = image[y1:y2, x1:x2]
            
            image = cv2.resize(image, (224, 224))
            
            
            try:
                ROI_prediction = self.Feature_Extraction_and_Visualization_Screen.old_model_detection(cv2.cvtColor( cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR ))
                x1,y1,x2,y2 = self.Feature_Extraction_and_Visualization_Screen.get_coordinate_Predict_image(ROI_prediction)[0][0:4] 
                image = image[y1:y2, x1:x2]

            except:
                self.not_detection_flag = 1
                print("Countour can not be detected")
    
        else: 
            pass
        
        if self.not_detection_flag == 0:
            # Preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(gray)
            
            
            # Transformation
            transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                ])
        
            input_tensor = transform(image)
            input_batch = input_tensor.unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                
                
                
                if classification_approach == 1: # for Normal vs OsteoArthritis approach
                    
                    if split_ratio_called == 1:
                        outputs = self.pytorch_model_2C(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return  outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    elif split_ratio_called == 2:
                        outputs = self.pytorch_model_2C_70_30(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return  outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    elif split_ratio_called == 3:
                        outputs = self.pytorch_model_2C_60_40(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return  outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    else:
                        pass
                    
                elif classification_approach == 2: # for Normal vs Mild vs Severe approach
                    
                    if split_ratio_called == 1:
                        outputs = self.pytorch_model_3C(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return  outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    elif split_ratio_called == 2:
                        outputs = self.pytorch_model_3C_70_30(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return  outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    
                    elif split_ratio_called == 3:
                        outputs = self.pytorch_model_3C_60_40(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return  outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    else:
                        pass
                    
                    
                elif classification_approach == 3: # for 5 classes approach
                    
                    if split_ratio_called == 1:
                        outputs = self.pytorch_model_5C(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    elif split_ratio_called == 2:
                        outputs = self.pytorch_model_5C_70_30(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    
                    elif split_ratio_called == 3:
                        outputs = self.pytorch_model_5C_60_40(input_batch)
                        prediction = torch.max(outputs.data, 1)[1]
                        return outputs[0].cpu().numpy(),prediction.cpu().numpy()[0]
                    
                    else:
                        pass
                
                elif classification_approach == 4: # for  Fused Models approach
                    
                    if split_ratio_called == 1:
                        
                        outputs_0_1 = self.pytorch_model_0_1(input_batch)[0].cpu().numpy()
                        outputs_0_2 = self.pytorch_model_0_2(input_batch)[0].cpu().numpy()
                        outputs_0_3 = self.pytorch_model_0_3(input_batch)[0].cpu().numpy()
                        outputs_0_4 = self.pytorch_model_0_4(input_batch)[0].cpu().numpy()
                        outputs_1_2 = self.pytorch_model_1_2(input_batch)[0].cpu().numpy()
                        outputs_1_3 = self.pytorch_model_1_3(input_batch)[0].cpu().numpy()
                        outputs_1_4 = self.pytorch_model_1_4(input_batch)[0].cpu().numpy()
                        outputs_2_3 = self.pytorch_model_2_3(input_batch)[0].cpu().numpy()
                        outputs_2_4 = self.pytorch_model_2_4(input_batch)[0].cpu().numpy()
                        outputs_3_4 = self.pytorch_model_3_4(input_batch)[0].cpu().numpy()
                    
                    elif split_ratio_called == 2:
                        
                        outputs_0_1 = self.pytorch_model_0_1_70_30(input_batch)[0].cpu().numpy()
                        outputs_0_2 = self.pytorch_model_0_2_70_30(input_batch)[0].cpu().numpy()
                        outputs_0_3 = self.pytorch_model_0_3_70_30(input_batch)[0].cpu().numpy()
                        outputs_0_4 = self.pytorch_model_0_4_70_30(input_batch)[0].cpu().numpy()
                        outputs_1_2 = self.pytorch_model_1_2_70_30(input_batch)[0].cpu().numpy()
                        outputs_1_3 = self.pytorch_model_1_3_70_30(input_batch)[0].cpu().numpy()
                        outputs_1_4 = self.pytorch_model_1_4_70_30(input_batch)[0].cpu().numpy()
                        outputs_2_3 = self.pytorch_model_2_3_70_30(input_batch)[0].cpu().numpy()
                        outputs_2_4 = self.pytorch_model_2_4_70_30(input_batch)[0].cpu().numpy()
                        outputs_3_4 = self.pytorch_model_3_4_70_30(input_batch)[0].cpu().numpy()
                        
                        
                        
                    elif split_ratio_called == 3:
                        
                        outputs_0_1 = self.pytorch_model_0_1_60_40(input_batch)[0].cpu().numpy()
                        outputs_0_2 = self.pytorch_model_0_2_60_40(input_batch)[0].cpu().numpy()
                        outputs_0_3 = self.pytorch_model_0_3_60_40(input_batch)[0].cpu().numpy()
                        outputs_0_4 = self.pytorch_model_0_4_60_40(input_batch)[0].cpu().numpy()
                        outputs_1_2 = self.pytorch_model_1_2_60_40(input_batch)[0].cpu().numpy()
                        outputs_1_3 = self.pytorch_model_1_3_60_40(input_batch)[0].cpu().numpy()
                        outputs_1_4 = self.pytorch_model_1_4_60_40(input_batch)[0].cpu().numpy()
                        outputs_2_3 = self.pytorch_model_2_3_60_40(input_batch)[0].cpu().numpy()
                        outputs_2_4 = self.pytorch_model_2_4_60_40(input_batch)[0].cpu().numpy()
                        outputs_3_4 = self.pytorch_model_3_4_60_40(input_batch)[0].cpu().numpy()
                        
                    else:
                        pass
                        
                    
                        
                    votes = [] 
                    
                    votes_0 = (outputs_0_1[0]) + (outputs_0_2[0]) + (outputs_0_3[0]) + (outputs_0_4[0]) 
                    votes_1 = (outputs_0_1[1]) + (outputs_1_2[0]) + (outputs_1_3[0]) + (outputs_1_4[0])
                    votes_2 = (outputs_0_2[1]) + (outputs_1_2[1]) + (outputs_2_3[0]) + (outputs_2_4[0])
                    votes_3 = (outputs_0_3[1]) + (outputs_1_3[1]) + (outputs_2_3[1]) + (outputs_3_4[0])     
                    votes_4 = (outputs_0_4[1]) + (outputs_1_4[1]) + (outputs_2_4[1]) + (outputs_3_4[1])
                    
                    votes.append([votes_0,
                                votes_1,
                                votes_2,
                                votes_3,
                                votes_4])
                    
                    probabilities = self.softmax(votes[0])
                    prediction = np.argmax(probabilities)
                
                return np.array(probabilities), prediction
        else:
            self.not_detection_flag = 0
            return [], None
                
 
    def softmax(self, x):
        e_x = np.exp(x - np.max(x)).round(4)
        return e_x/e_x.sum()
    
    def on_button_click(self):
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.image_label.clear()
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        
        self.on_combo_clear()

    def on_combo_clear(self):
        self.output2 = None
        self.class2 = None
        self.output3 = None
        self.class3 = None
        self.output5 = None
        self.class5 = None
        self.output5_Tree = None
        self.class5_Tree = None
        self.score_Class2 = None
        self.score_Class3 = None
        self.score_Class5 = None
        self.score_Class5_Tree = None
        
        self.table.clearContents()
        self.table1.clearContents()
        self.table2.clearContents()
        self.table3.clearContents()
        self.table4.clearContents()
        
        self.table_layout_VBOX_Widget.setVisible(False)
        self.table.setVisible(False)
        self.table1.setVisible(False)
        self.table2.setVisible(False)
        self.table3.setVisible(False)
        self.table4.setVisible(False)
        
        self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
        self.image_label2.setVisible(True)
        self.image_label2.setText("AI Automated")
        
        
        
    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        image_path = "imgs/Feature_Extraction.png"
   
        if self.mouse_double_clicked == 1:
            pass
        else:
            self.set_background_image(self.image_label, image_path)
            

#  _________________________________ Conventional_CAD Screen ____________________________________________________

class Conventional_CAD_Screen(QWidget):
    variableChanged_Equalize_Feature_Visualization = pyqtSignal(bool)
    variableChanged_Auto_mode = pyqtSignal(bool)
    variableChanged_Visualization = pyqtSignal(bool)
    
    def __init__(self, main_window):
        super().__init__()
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        
        # self.resize(1500, 1000)
        self.resize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setMinimumSize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setMaximumSize(screen_width, screen_height)
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        
        
        self.main_window = main_window
        self.main_window.variableChanged.connect(self.handle_variable_changed)
        self.main_window.VariableChanged_Combo_Conventional.connect(self.handle_variable_changed_SR)
        self.main_window.VariableChanged_Combo_Fused_to_Conventional.connect(self.handle_variable_changed_SR)
        
        self.AI_Automated_CAD_Screen = AI_Automated_CAD_Screen(self.main_window)
        self.Fused_CVML_and_Ai_CAD_Screen = Fused_CVML_and_Ai_CAD_Screen(self.main_window, self, self.AI_Automated_CAD_Screen)
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen(self, None)

# ______________________________________________ initialization __________________________________________________
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.Features_input = []
        self.Features_input_LBP = []
        self.Features_input_LTP = []
        self.Features_input_HOG = []
        self.FirstSetVariable = 1
        
        self.load_cvml_Two_clases_models_Flag = 0    
        self.load_cvml_Three_clases_models_Flag = 0    
        self.load_cvml_Five_clases_models_Flag = 0    
        self.load_cvml_Five_clases_Binary_Tree_models_Flag = 0
        
        self.load_cvml_Two_clases_models_70_30_Flag = 0    
        self.load_cvml_Three_clases_models_70_30_Flag = 0    
        self.load_cvml_Five_clases_models_70_30_Flag = 0    
        self.load_cvml_Five_clases_Binary_Tree_models_70_30_Flag = 0
        
        self.load_cvml_Two_clases_models_60_40_Flag = 0    
        self.load_cvml_Three_clases_models_60_40_Flag = 0    
        self.load_cvml_Five_clases_models_60_40_Flag = 0    
        self.load_cvml_Five_clases_Binary_Tree_models_60_40_Flag = 0
        
        self.output2 = None
        self.class2 = None
        self.output3 = None
        self.class3 = None
        self.output5 = None
        self.class5 = None
        self.output5_Tree = None
        self.class5_Tree = None
        self.score_Class2 = None
        self.score_Class3 = None
        self.score_Class5 = None
        self.score_Class5_Tree = None
        
        self.SR = 1
# ______________________________________________ show_main_content __________________________________________________
        self.show_main_content()
    
    def handle_variable_changed_SR(self, new_value):
        self.SR = new_value
        print("SR", self.SR)
        
        if self.main_window.clasify_indicator == 1:
            self.show_Predictions()
            # self.show_Predictions(self.class2, self.score_Class2, self.output2, self.class3, self.score_Class3, self.output3, self.class5, self.score_Class5, self.output5, self.class5_Tree, self.score_Class5_Tree, self.output5_Tree)
            
        else:
            pass
            print(f"clasify_indicator {self.main_window.clasify_indicator}")
        
        
        
        
        
    def load_cvml_Five_clases_Binary_Tree_models(self):
        # 5 classes binary Tree classification models loading...
        # loading Scalers...
        try:
            if self.load_cvml_Five_clases_Binary_Tree_models_Flag == 1:
                return 
            
            if not hasattr(self, 'scaler_0_vs_1_HOG'):
                self.scaler_0_vs_1_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_0_vs_1_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_1_other_features'):
                self.scaler_0_vs_1_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_0_vs_1_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_2_HOG'):
                self.scaler_0_vs_2_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_0_vs_2_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_2_other_features'):
                self.scaler_0_vs_2_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_0_vs_2_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_3_HOG'):
                self.scaler_0_vs_3_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_0_vs_3_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_3_other_features'):
                self.scaler_0_vs_3_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_0_vs_3_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_4_HOG'):
                self.scaler_0_vs_4_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_0_vs_4_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_4_other_features'):
                self.scaler_0_vs_4_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_0_vs_4_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_2_HOG'):
                self.scaler_1_vs_2_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_1_vs_2_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_2_other_features'):
                self.scaler_1_vs_2_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_1_vs_2_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_3_HOG'):
                self.scaler_1_vs_3_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_1_vs_3_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_3_other_features'):
                self.scaler_1_vs_3_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_1_vs_3_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_4_HOG'):
                self.scaler_1_vs_4_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_1_vs_4_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_4_other_features'):
                self.scaler_1_vs_4_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_1_vs_4_other_features.pkl")
            if not hasattr(self, 'scaler_2_vs_3_HOG'):
                self.scaler_2_vs_3_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_2_vs_3_HOG.pkl")
            if not hasattr(self, 'scaler_2_vs_3_other_features'):
                self.scaler_2_vs_3_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_2_vs_3_other_features.pkl")
            if not hasattr(self, 'scaler_2_vs_4_HOG'):
                self.scaler_2_vs_4_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_2_vs_4_HOG.pkl")
            if not hasattr(self, 'scaler_2_vs_4_other_features'):
                self.scaler_2_vs_4_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_2_vs_4_other_features.pkl")
            if not hasattr(self, 'scaler_3_vs_4_HOG'):
                self.scaler_3_vs_4_HOG = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_3_vs_4_HOG.pkl")
            if not hasattr(self, 'scaler_3_vs_4_other_features'):
                self.scaler_3_vs_4_other_features = load("Models/New Conventional Models/Binary Tree/Scalers/scaler_3_vs_4_other_features.pkl")
            
            # loading Models...
            if not hasattr(self, 'LR_0_vs_4_HOG'):
                self.LR_0_vs_4_HOG = load("Models/New Conventional Models/Binary Tree/Models/LR_0_vs_4_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_1_HOG'):
                self.MLP_0_vs_1_HOG = load("Models/New Conventional Models/Binary Tree/Models/MLP_0_vs_1_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_2_HOG'):
                self.MLP_0_vs_2_HOG = load("Models/New Conventional Models/Binary Tree/Models/MLP_0_vs_2_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_2_other_features'):
                self.MLP_0_vs_2_other_features = load("Models/New Conventional Models/Binary Tree/Models/MLP_0_vs_2_other_features.pkl")
            if not hasattr(self, 'MLP_0_vs_3_HOG'):
                self.MLP_0_vs_3_HOG = load("Models/New Conventional Models/Binary Tree/Models/MLP_0_vs_3_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_3_other_features'):
                self.MLP_0_vs_3_other_features = load("Models/New Conventional Models/Binary Tree/Models/MLP_0_vs_3_other_features.pkl")
            if not hasattr(self, 'MLP_0_vs_4_other_features'):
                self.MLP_0_vs_4_other_features = load("Models/New Conventional Models/Binary Tree/Models/MLP_0_vs_4_other_features.pkl")
            if not hasattr(self, 'MLP_1_vs_3_other_features'):
                self.MLP_1_vs_3_other_features = load("Models/New Conventional Models/Binary Tree/Models/MLP_1_vs_3_other_features.pkl")
            if not hasattr(self, 'MLP_1_vs_4_HOG'):
                self.MLP_1_vs_4_HOG = load("Models/New Conventional Models/Binary Tree/Models/MLP_1_vs_4_HOG.pkl")
            if not hasattr(self, 'MLP_1_vs_4_other_features'):
                self.MLP_1_vs_4_other_features = load("Models/New Conventional Models/Binary Tree/Models/MLP_1_vs_4_other_features.pkl")
            if not hasattr(self, 'MLP_2_vs_3_HOG'):
                self.MLP_2_vs_3_HOG = load("Models/New Conventional Models/Binary Tree/Models/MLP_2_vs_3_HOG.pkl")
            if not hasattr(self, 'MLP_2_vs_4_HOG'):
                self.MLP_2_vs_4_HOG = load("Models/New Conventional Models/Binary Tree/Models/MLP_2_vs_4_HOG.pkl")
            if not hasattr(self, 'MLP_2_vs_4_other_features'):
                self.MLP_2_vs_4_other_features = load("Models/New Conventional Models/Binary Tree/Models/MLP_2_vs_4_other_features.pkl")
            if not hasattr(self, 'MLP_3_vs_4_other_features'):
                self.MLP_3_vs_4_other_features = load("Models/New Conventional Models/Binary Tree/Models/MLP_3_vs_4_other_features.pkl")
            if not hasattr(self, 'RF_0_vs_1_other_features'):
                self.RF_0_vs_1_other_features = load("Models/New Conventional Models/Binary Tree/Models/RF_0_vs_1_other_features.pkl")
            if not hasattr(self, 'RF_1_vs_2_other_features'):
                self.RF_1_vs_2_other_features = load("Models/New Conventional Models/Binary Tree/Models/RF_1_vs_2_other_features.pkl")
            if not hasattr(self, 'RF_2_vs_3_other_features'):
                self.RF_2_vs_3_other_features = load("Models/New Conventional Models/Binary Tree/Models/RF_2_vs_3_other_features.pkl")
            if not hasattr(self, 'SVM_1_vs_2_HOG'):
                self.SVM_1_vs_2_HOG = load("Models/New Conventional Models/Binary Tree/Models/SVM_1_vs_2_HOG.pkl")
            if not hasattr(self, 'SVM_1_vs_3_HOG'):
                self.SVM_1_vs_3_HOG = load("Models/New Conventional Models/Binary Tree/Models/SVM_1_vs_3_HOG.pkl")
            if not hasattr(self, 'SVM_3_vs_4_HOG'):
                self.SVM_3_vs_4_HOG = load("Models/New Conventional Models/Binary Tree/Models/SVM_3_vs_4_HOG.pkl")
            
            self.load_cvml_Five_clases_Binary_Tree_models_Flag = 1            
            
        except RuntimeError:
            pass
              
    def load_cvml_Two_clases_models(self):
        # 2 classes classification models loading...
        try:
            if self.load_cvml_Two_clases_models_Flag == 1:
                return
            if not hasattr(self, 'MLP_Normal_vs_osteo_other_features'):
                self.MLP_Normal_vs_osteo_other_features = load('Models/New Conventional Models/2 classes/MLP_Normal_vs_osteo_other_features.pkl')
            if not hasattr(self, 'MLP_Normal_vs_osteo_HOG'):
                self.MLP_Normal_vs_osteo_HOG = load('Models/New Conventional Models/2 classes/MLP_Normal_vs_osteo_HOG.pkl')
            if not hasattr(self, 'scaler_normal_vs_osteo_other_features'):
                self.scaler_normal_vs_osteo_other_features = load('Models/New Conventional Models/2 classes/scaler_normal_vs_osteo_other_features.pkl')
            if not hasattr(self, 'scaler_normal_vs_osteo_HOG'):
                self.scaler_normal_vs_osteo_HOG = load('Models/New Conventional Models/2 classes/scaler_normal_vs_osteo_HOG.pkl')
    
            self.load_cvml_Two_clases_models_Flag = 1
            
        except RuntimeError:
            pass
   

   
    def load_cvml_Three_clases_models(self):
        # 3 classes classification models loading...
        try:
            if self.load_cvml_Three_clases_models_Flag == 1:
                return 
            
            if not hasattr(self, 'MLP_3_Classes_other_features'):
                self.MLP_3_Classes_other_features = load('Models/New Conventional Models/3 classes/MLP_3_Classes_other_features.pkl')
            if not hasattr(self, 'MLP_3_Classes_HOG'):
                self.MLP_3_Classes_HOG = load('Models/New Conventional Models/3 classes/MLP_3_Classes_HOG.pkl')
            if not hasattr(self, 'scaler_3_Classes_other_features'):
                self.scaler_3_Classes_other_features = load('Models/New Conventional Models/3 classes/scaler_3_Classes_other_features.pkl')
            if not hasattr(self, 'scaler_3_Classes_HOG'):
                self.scaler_3_Classes_HOG = load('Models/New Conventional Models/3 classes/scaler_3_Classes_HOG.pkl')

            self.load_cvml_Three_clases_models_Flag = 1
                    
        except RuntimeError:
            pass    
            
    def load_cvml_Five_clases_models(self):
        # 5 classes classification models loading...
        try: 
            if self.load_cvml_Five_clases_models_Flag == 1:
                return 
            
            if not hasattr(self, 'Random_Forest_Chenn_and_Expert_Other_Features_5_Class'):
                self.Random_Forest_Chenn_and_Expert_Other_Features_5_Class = load('Models/New Conventional Models/5 classes/Random_Forest_Chenn_and_Expert_Other_Features_5_Class.pkl')
            if not hasattr(self, 'MLP_Chenn_and_Expert_HOG_5_Class'):
                self.MLP_Chenn_and_Expert_HOG_5_Class = load('Models/New Conventional Models/5 classes/MLP_Chenn_and_Expert_HOG_5_Class.pkl')
            if not hasattr(self, 'scaler_Chenn_and_Expert_Other_Features_5_Class'):
                self.scaler_Chenn_and_Expert_Other_Features_5_Class = load('Models/New Conventional Models/5 classes/scaler_Chenn_and_Expert_Other_Features_5_Class.pkl')
            if not hasattr(self, 'scaler_Chenn_and_Expert_HOG_5_Class'):
                self.scaler_Chenn_and_Expert_HOG_5_Class = load('Models/New Conventional Models/5 classes/scaler_Chenn_and_Expert_HOG_5_Class.pkl')
           
            self.load_cvml_Five_clases_models_Flag = 1 
                
        except RuntimeError:
            pass     














    def load_cvml_Five_clases_Binary_Tree_models_70_30(self):
        # 5 classes binary Tree classification models loading...
        # loading Scalers...
        try:
            if self.load_cvml_Five_clases_Binary_Tree_models_70_30_Flag == 1:
                return 
            
            if not hasattr(self, 'scaler_0_vs_1_HOG_70_30'):
                self.scaler_0_vs_1_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_0_vs_1_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_1_other_features_70_30'):
                self.scaler_0_vs_1_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_0_vs_1_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_2_HOG_70_30'):
                self.scaler_0_vs_2_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_0_vs_2_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_2_other_features_70_30'):
                self.scaler_0_vs_2_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_0_vs_2_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_3_HOG_70_30'):
                self.scaler_0_vs_3_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_0_vs_3_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_3_other_features_70_30'):
                self.scaler_0_vs_3_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_0_vs_3_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_4_HOG_70_30'):
                self.scaler_0_vs_4_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_0_vs_4_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_4_other_features_70_30'):
                self.scaler_0_vs_4_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_0_vs_4_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_2_HOG_70_30'):
                self.scaler_1_vs_2_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_1_vs_2_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_2_other_features_70_30'):
                self.scaler_1_vs_2_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_1_vs_2_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_3_HOG_70_30'):
                self.scaler_1_vs_3_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_1_vs_3_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_3_other_features_70_30'):
                self.scaler_1_vs_3_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_1_vs_3_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_4_HOG_70_30'):
                self.scaler_1_vs_4_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_1_vs_4_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_4_other_features_70_30'):
                self.scaler_1_vs_4_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_1_vs_4_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_2_vs_3_HOG_70_30'):
                self.scaler_2_vs_3_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_2_vs_3_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_2_vs_3_other_features_70_30'):
                self.scaler_2_vs_3_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_2_vs_3_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_2_vs_4_HOG_70_30'):
                self.scaler_2_vs_4_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_2_vs_4_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_2_vs_4_other_features_70_30'):
                self.scaler_2_vs_4_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_2_vs_4_70_30_other_features.pkl")
            if not hasattr(self, 'scaler_3_vs_4_HOG_70_30'):
                self.scaler_3_vs_4_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_3_vs_4_70_30_HOG.pkl")
            if not hasattr(self, 'scaler_3_vs_4_other_features_70_30'):
                self.scaler_3_vs_4_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Scalers/scaler_3_vs_4_70_30_other_features.pkl")
            
            # loading Models...
            if not hasattr(self, 'LR_0_vs_4_HOG_70_30'):
                self.LR_0_vs_4_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_0_vs_4_70_30_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_1_HOG_70_30'):
                self.MLP_0_vs_1_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_0_vs_1_70_30_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_2_HOG_70_30'):
                self.MLP_0_vs_2_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_0_vs_2_70_30_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_2_other_features_70_30'):
                self.MLP_0_vs_2_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/RF_0_vs_2_70_30_other_features.pkl")
            if not hasattr(self, 'MLP_0_vs_3_HOG_70_30'):
                self.MLP_0_vs_3_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/LR_0_vs_3_70_30_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_3_other_features_70_30'):
                self.MLP_0_vs_3_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/RF_0_vs_3_70_30_other_features.pkl")
            if not hasattr(self, 'MLP_0_vs_4_other_features_70_30'):
                self.MLP_0_vs_4_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_0_vs_4_70_30_other_features.pkl")
            if not hasattr(self, 'MLP_1_vs_3_other_features_70_30'):
                self.MLP_1_vs_3_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_1_vs_3_70_30_other_features.pkl")
            if not hasattr(self, 'MLP_1_vs_4_HOG_70_30'):
                self.MLP_1_vs_4_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/LR_1_vs_4_70_30_HOG.pkl")
            if not hasattr(self, 'MLP_1_vs_4_other_features_70_30'):
                self.MLP_1_vs_4_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_1_vs_4_70_30_other_features.pkl")
            if not hasattr(self, 'MLP_2_vs_3_HOG_70_30'):
                self.MLP_2_vs_3_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_2_vs_3_70_30_HOG.pkl")
            if not hasattr(self, 'MLP_2_vs_4_HOG_70_30'):
                self.MLP_2_vs_4_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_2_vs_4_70_30_HOG.pkl")
            if not hasattr(self, 'MLP_2_vs_4_other_features_70_30'):
                self.MLP_2_vs_4_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/RF_2_vs_4_70_30_other_features.pkl")
            if not hasattr(self, 'MLP_3_vs_4_other_features_70_30'):
                self.MLP_3_vs_4_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/RF_3_vs_4_70_30_other_features.pkl")
            if not hasattr(self, 'RF_0_vs_1_other_features_70_30'):
                self.RF_0_vs_1_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/RF_0_vs_1_70_30_other_features.pkl")
            if not hasattr(self, 'RF_1_vs_2_other_features_70_30'):
                self.RF_1_vs_2_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_1_vs_2_70_30_other_features.pkl")
            if not hasattr(self, 'RF_2_vs_3_other_features_70_30'):
                self.RF_2_vs_3_other_features_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/RF_2_vs_3_70_30_other_features.pkl")
            if not hasattr(self, 'SVM_1_vs_2_HOG_70_30'):
                self.SVM_1_vs_2_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_1_vs_2_70_30_HOG.pkl")
            if not hasattr(self, 'SVM_1_vs_3_HOG_70_30'):
                self.SVM_1_vs_3_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_1_vs_3_70_30_HOG.pkl")
            if not hasattr(self, 'SVM_3_vs_4_HOG_70_30'):
                self.SVM_3_vs_4_HOG_70_30 = load("Models/New Conventional Models/Binary Tree_70_30/Models/MLP_3_vs_4_70_30_HOG.pkl")
            
            self.load_cvml_Five_clases_Binary_Tree_models_70_30_Flag = 1            
            
        except RuntimeError:
            pass
              
    def load_cvml_Two_clases_models_70_30(self):
        # 2 classes classification models loading...
        try:
            if self.load_cvml_Two_clases_models_70_30_Flag == 1:
                return 
            
            if not hasattr(self, 'MLP_Normal_vs_osteo_other_features_70_30'):
                self.MLP_Normal_vs_osteo_other_features_70_30 = load('Models/New Conventional Models/2 classes_70_30/MLP_Normal_vs_osteo_70_30_other_features.pkl')
            if not hasattr(self, 'MLP_Normal_vs_osteo_HOG_70_30'):
                self.MLP_Normal_vs_osteo_HOG_70_30 = load('Models/New Conventional Models/2 classes_70_30/MLP_Normal_vs_osteo_70_30_HOG.pkl')
            if not hasattr(self, 'scaler_normal_vs_osteo_other_features_70_30'):
                self.scaler_normal_vs_osteo_other_features_70_30 = load('Models/New Conventional Models/2 classes_70_30/scaler_Normal_vs_osteo_70_30_other_features.pkl')
            if not hasattr(self, 'scaler_normal_vs_osteo_HOG_70_30'):
                self.scaler_normal_vs_osteo_HOG_70_30 = load('Models/New Conventional Models/2 classes_70_30/scaler_Normal_vs_osteo_70_30_HOG.pkl')
    
            self.load_cvml_Two_clases_models_70_30_Flag = 1
            
        except RuntimeError:
            pass
   

   
    def load_cvml_Three_clases_models_70_30(self):
        # 3 classes classification models loading...
        try:
            if self.load_cvml_Three_clases_models_70_30_Flag == 1:
                return
            
            if not hasattr(self, 'MLP_3_Classes_other_features_70_30'):
                self.MLP_3_Classes_other_features_70_30 = load('Models/New Conventional Models/3 classes_70_30/MLP_3_Classes_70_30_other_features.pkl')
            if not hasattr(self, 'MLP_3_Classes_HOG_70_30'):
                self.MLP_3_Classes_HOG_70_30 = load('Models/New Conventional Models/3 classes_70_30/MLP_3_Classes_70_30_HOG.pkl')
            if not hasattr(self, 'scaler_3_Classes_other_features_70_30'):
                self.scaler_3_Classes_other_features_70_30 = load('Models/New Conventional Models/3 classes_70_30/scaler_3_Classes_70_30_other_features.pkl')
            if not hasattr(self, 'scaler_3_Classes_HOG_70_30'):
                self.scaler_3_Classes_HOG_70_30 = load('Models/New Conventional Models/3 classes_70_30/scaler_3_Classes_70_30_HOG.pkl')

            self.load_cvml_Three_clases_models_70_30_Flag = 1
                    
        except RuntimeError:
            pass    
            
    def load_cvml_Five_clases_models_70_30(self):
        # 5 classes classification models loading...
        try:
            if self.load_cvml_Five_clases_models_70_30_Flag == 1:
                return 
            
            if not hasattr(self, 'Random_Forest_Chenn_and_Expert_Other_Features_5_Class_70_30'):
                self.Random_Forest_Chenn_and_Expert_Other_Features_5_Class_70_30 = load('Models/New Conventional Models/5 classes_70_30/MLP_5_Classes_70_30_other_features.pkl')
            if not hasattr(self, 'MLP_Chenn_and_Expert_HOG_5_Class_70_30'):
                self.MLP_Chenn_and_Expert_HOG_5_Class_70_30 = load('Models/New Conventional Models/5 classes_70_30/MLP_5_Classes_70_30_HOG.pkl')
            if not hasattr(self, 'scaler_Chenn_and_Expert_Other_Features_5_Class_70_30'):
                self.scaler_Chenn_and_Expert_Other_Features_5_Class_70_30 = load('Models/New Conventional Models/5 classes_70_30/scaler_5_Classes_70_30_other_features.pkl')
            if not hasattr(self, 'scaler_Chenn_and_Expert_HOG_5_Class_70_30'):
                self.scaler_Chenn_and_Expert_HOG_5_Class_70_30 = load('Models/New Conventional Models/5 classes_70_30/scaler_5_Classes_70_30_HOG.pkl')
           
            self.load_cvml_Five_clases_models_70_30_Flag = 1 
                
        except RuntimeError:
            pass 
        
        
        
        
        
        
        
        
        
        
        
    def load_cvml_Five_clases_Binary_Tree_models_60_40(self):
        # 5 classes binary Tree classification models loading...
        # loading Scalers...
        try:
            if self.load_cvml_Five_clases_Binary_Tree_models_60_40_Flag == 1:
                return
            
            if not hasattr(self, 'scaler_0_vs_1_HOG_60_40'):
                self.scaler_0_vs_1_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_0_vs_1_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_1_other_features_60_40'):
                self.scaler_0_vs_1_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_0_vs_1_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_2_HOG_60_40'):
                self.scaler_0_vs_2_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_0_vs_2_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_2_other_features_60_40'):
                self.scaler_0_vs_2_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_0_vs_2_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_3_HOG_60_40'):
                self.scaler_0_vs_3_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_0_vs_3_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_3_other_features_60_40'):
                self.scaler_0_vs_3_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_0_vs_3_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_0_vs_4_HOG_60_40'):
                self.scaler_0_vs_4_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_0_vs_4_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_0_vs_4_other_features_60_40'):
                self.scaler_0_vs_4_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_0_vs_4_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_2_HOG_60_40'):
                self.scaler_1_vs_2_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_1_vs_2_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_2_other_features_60_40'):
                self.scaler_1_vs_2_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_1_vs_2_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_3_HOG_60_40'):
                self.scaler_1_vs_3_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_1_vs_3_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_3_other_features_60_40'):
                self.scaler_1_vs_3_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_1_vs_3_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_1_vs_4_HOG_60_40'):
                self.scaler_1_vs_4_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_1_vs_4_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_1_vs_4_other_features_60_40'):
                self.scaler_1_vs_4_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_1_vs_4_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_2_vs_3_HOG_60_40'):
                self.scaler_2_vs_3_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_2_vs_3_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_2_vs_3_other_features_60_40'):
                self.scaler_2_vs_3_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_2_vs_3_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_2_vs_4_HOG_60_40'):
                self.scaler_2_vs_4_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_2_vs_4_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_2_vs_4_other_features_60_40'):
                self.scaler_2_vs_4_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_2_vs_4_60_40_other_features.pkl")
            if not hasattr(self, 'scaler_3_vs_4_HOG_60_40'):
                self.scaler_3_vs_4_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_3_vs_4_60_40_HOG.pkl")
            if not hasattr(self, 'scaler_3_vs_4_other_features_60_40'):
                self.scaler_3_vs_4_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Scalers/scaler_3_vs_4_60_40_other_features.pkl")
            
            # loading Models...
            if not hasattr(self, 'LR_0_vs_4_HOG_60_40'):
                self.LR_0_vs_4_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/LR_0_vs_4_60_40_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_1_HOG_60_40'):
                self.MLP_0_vs_1_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_0_vs_1_60_40_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_2_HOG_60_40'):
                self.MLP_0_vs_2_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_0_vs_2_60_40_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_2_other_features_60_40'):
                self.MLP_0_vs_2_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_0_vs_2_60_40_other_features.pkl")
            if not hasattr(self, 'MLP_0_vs_3_HOG_60_40'):
                self.MLP_0_vs_3_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_0_vs_3_60_40_HOG.pkl")
            if not hasattr(self, 'MLP_0_vs_3_other_features_60_40'):
                self.MLP_0_vs_3_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/RF_0_vs_3_60_40_other_features.pkl")
            if not hasattr(self, 'MLP_0_vs_4_other_features_60_40'):
                self.MLP_0_vs_4_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/RF_0_vs_4_60_40_other_features.pkl")
            if not hasattr(self, 'MLP_1_vs_3_other_features_60_40'):
                self.MLP_1_vs_3_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_1_vs_3_60_40_other_features.pkl")
            if not hasattr(self, 'MLP_1_vs_4_HOG_60_40'):
                self.MLP_1_vs_4_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/LR_1_vs_4_60_40_HOG.pkl")
            if not hasattr(self, 'MLP_1_vs_4_other_features_60_40'):
                self.MLP_1_vs_4_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_1_vs_4_60_40_other_features.pkl")
            if not hasattr(self, 'MLP_2_vs_3_HOG_60_40'):
                self.MLP_2_vs_3_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_2_vs_3_60_40_HOG.pkl")
            if not hasattr(self, 'MLP_2_vs_4_HOG_60_40'):
                self.MLP_2_vs_4_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_2_vs_4_60_40_HOG.pkl")
            if not hasattr(self, 'MLP_2_vs_4_other_features_60_40'):
                self.MLP_2_vs_4_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_2_vs_4_60_40_other_features.pkl")
            if not hasattr(self, 'MLP_3_vs_4_other_features_60_40'):
                self.MLP_3_vs_4_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/RF_3_vs_4_60_40_other_features.pkl")
            if not hasattr(self, 'RF_0_vs_1_other_features_60_40'):
                self.RF_0_vs_1_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/RF_0_vs_1_60_40_other_features.pkl")
            if not hasattr(self, 'RF_1_vs_2_other_features_60_40'):
                self.RF_1_vs_2_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_1_vs_2_60_40_other_features.pkl")
            if not hasattr(self, 'RF_2_vs_3_other_features_60_40'):
                self.RF_2_vs_3_other_features_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_2_vs_3_60_40_other_features.pkl")
            if not hasattr(self, 'SVM_1_vs_2_HOG_60_40'):
                self.SVM_1_vs_2_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_1_vs_2_60_40_HOG.pkl")
            if not hasattr(self, 'SVM_1_vs_3_HOG_60_40'):
                self.SVM_1_vs_3_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_1_vs_3_60_40_HOG.pkl")
            if not hasattr(self, 'SVM_3_vs_4_HOG_60_40'):
                self.SVM_3_vs_4_HOG_60_40 = load("Models/New Conventional Models/Binary Tree_60_40/Models/MLP_3_vs_4_60_40_HOG.pkl")
            
            self.load_cvml_Five_clases_Binary_Tree_models_60_40_Flag = 1            
            
        except RuntimeError:
            pass
              
    def load_cvml_Two_clases_models_60_40(self):
        # 2 classes classification models loading...
        try:
            if self.load_cvml_Two_clases_models_60_40_Flag == 1:
                return 
            
            if not hasattr(self, 'MLP_Normal_vs_osteo_other_features_60_40'):
                self.MLP_Normal_vs_osteo_other_features_60_40 = load('Models/New Conventional Models/2 classes_60_40/RF_Normal_vs_osteo_60_40_other_features.pkl')
            if not hasattr(self, 'MLP_Normal_vs_osteo_HOG_60_40'):
                self.MLP_Normal_vs_osteo_HOG_60_40 = load('Models/New Conventional Models/2 classes_60_40/MLP_Normal_vs_osteo_60_40_HOG.pkl')
            if not hasattr(self, 'scaler_normal_vs_osteo_other_features_60_40'):
                self.scaler_normal_vs_osteo_other_features_60_40 = load('Models/New Conventional Models/2 classes_60_40/scaler_Normal_vs_osteo_60_40_other_features.pkl')
            if not hasattr(self, 'scaler_normal_vs_osteo_HOG_60_40'):
                self.scaler_normal_vs_osteo_HOG_60_40 = load('Models/New Conventional Models/2 classes_60_40/scaler_Normal_vs_osteo_60_40_HOG.pkl')
    
            self.load_cvml_Two_clases_models_60_40_Flag = 1
            
        except RuntimeError:
            pass
   

   
    def load_cvml_Three_clases_models_60_40(self):
        # 3 classes classification models loading...
        try:
            if self.load_cvml_Three_clases_models_60_40_Flag == 1:
                return 
            
            if not hasattr(self, 'MLP_3_Classes_other_features_60_40'):
                self.MLP_3_Classes_other_features_60_40 = load('Models/New Conventional Models/3 classes_60_40/MLP_3_Classes_60_40_other_features.pkl')
            if not hasattr(self, 'MLP_3_Classes_HOG_60_40'):
                self.MLP_3_Classes_HOG_60_40 = load('Models/New Conventional Models/3 classes_60_40/MLP_3_Classes_60_40_HOG.pkl')
            if not hasattr(self, 'scaler_3_Classes_other_features_60_40'):
                self.scaler_3_Classes_other_features_60_40 = load('Models/New Conventional Models/3 classes_60_40/scaler_3_Classes_60_40_other_features.pkl')
            if not hasattr(self, 'scaler_3_Classes_HOG_60_40'):
                self.scaler_3_Classes_HOG_60_40 = load('Models/New Conventional Models/3 classes_60_40/scaler_3_Classes_60_40_HOG.pkl')

            self.load_cvml_Three_clases_models_60_40_Flag = 1
                    
        except RuntimeError:
            pass    
            
    def load_cvml_Five_clases_models_60_40(self):
        # 5 classes classification models loading...
        try:
            if self.load_cvml_Five_clases_models_60_40_Flag == 1:
                return 
            
            if not hasattr(self, 'Random_Forest_Chenn_and_Expert_Other_Features_5_Class_60_40'):
                self.Random_Forest_Chenn_and_Expert_Other_Features_5_Class_60_40 = load('Models/New Conventional Models/5 classes_60_40/RF_5_Classes_60_40_other_features.pkl')
            if not hasattr(self, 'MLP_Chenn_and_Expert_HOG_5_Class_60_40'):
                self.MLP_Chenn_and_Expert_HOG_5_Class_60_40 = load('Models/New Conventional Models/5 classes_60_40/MLP_5_Classes_60_40_HOG.pkl')
            if not hasattr(self, 'scaler_Chenn_and_Expert_Other_Features_5_Class_60_40'):
                self.scaler_Chenn_and_Expert_Other_Features_5_Class_60_40 = load('Models/New Conventional Models/5 classes_60_40/scaler_5_Classes_60_40_other_features.pkl')
            if not hasattr(self, 'scaler_Chenn_and_Expert_HOG_5_Class_60_40'):
                self.scaler_Chenn_and_Expert_HOG_5_Class_60_40 = load('Models/New Conventional Models/5 classes_60_40/scaler_5_Classes_60_40_HOG.pkl')
           
            self.load_cvml_Five_clases_models_60_40_Flag = 1 
                
        except RuntimeError:
            pass 
        
        
    def show_main_content(self):
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)
        self.image_label = ImageLabel(self, apply_effects = False)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setFixedWidth(int(0.3645833 * self.screen_width))
        self.image_label.setFixedHeight(int(0.3125 * self.screen_width))
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        self.image_label.updateEffectsBasedOnStyleAndPixmap()
        
        # Create the table
        self.table = QTableWidget()
        self.table.setFixedSize(int(0.4244791667 * self.screen_width), int(0.15625 * self.screen_width))
        # self.table.setFixedHeight(int(0.15625 * self.screen_width))
        
        # self.table.setFixedSize(825, 300)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.setRowCount(4)
        self.table.setColumnCount(2)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
        
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table's fixed size
        table_width = self.table.width()
        table_height = self.table.height()
        row_height = table_height // self.table.rowCount()
        column_width = table_width // self.table.columnCount()

        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, row_height)
        
        for column in range(self.table.columnCount()):
            self.table.setColumnWidth(column, column_width)
        
        self.table1 = QTableWidget()
        self.table1.setFixedHeight(int(0.25 * table_height))


        self.table1.verticalHeader().setVisible(False)
        self.table1.horizontalHeader().setVisible(False)
        self.table1.setRowCount(1)
        self.table1.setColumnCount(2)
        self.table1.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
                
                
        self.table1.setShowGrid(False)
        self.table1.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table1.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table1's fixed size
        table1_width = self.table1.width()
        table1_height = self.table1.height()
        row1_height = table1_height // self.table1.rowCount()
        column1_width = table1_width // self.table1.columnCount()

        for row1 in range(self.table1.rowCount()):
            self.table1.setRowHeight(row1, row1_height)
        
        for column1 in range(self.table1.columnCount()):
            self.table1.setColumnWidth(column1, column1_width)
        
        
        self.table2 = QTableWidget()
        self.table2.setFixedHeight(int(0.25 * table_height))
        self.table2.verticalHeader().setVisible(False)
        self.table2.horizontalHeader().setVisible(False)
        self.table2.setRowCount(1)
        self.table2.setColumnCount(3)
        self.table2.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
                
        self.table2.setShowGrid(False)
        self.table2.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table2.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table2's fixed size
        table2_width = self.table2.width()
        table2_height = self.table2.height()
        row2_height = table2_height // self.table2.rowCount()
        column2_width = table2_width // self.table2.columnCount()

        for row2 in range(self.table2.rowCount()):
            self.table2.setRowHeight(row2, row2_height)
        
        for column2 in range(self.table2.columnCount()):
            self.table2.setColumnWidth(column2, column2_width)
        
        self.table3 = QTableWidget()
        self.table3.setFixedHeight(int(0.25 * table_height))
        self.table3.verticalHeader().setVisible(False)
        self.table3.horizontalHeader().setVisible(False)
        self.table3.setRowCount(1)
        self.table3.setColumnCount(5)
        self.table3.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
        self.table3.setShowGrid(False)
        self.table3.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table3.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table3's fixed size
        table3_width = self.table3.width()
        table3_height = self.table3.height()
        row3_height = table3_height // self.table3.rowCount()
        column3_width = table3_width // self.table3.columnCount()

        for row3 in range(self.table3.rowCount()):
            self.table3.setRowHeight(row3, row3_height)
        
        for column3 in range(self.table3.columnCount()):
            self.table3.setColumnWidth(column3, column3_width)
        
        
        self.table4 = QTableWidget()
        self.table4.setFixedHeight(int(0.25 * table_height))
        self.table4.verticalHeader().setVisible(False)
        self.table4.horizontalHeader().setVisible(False)
        self.table4.setRowCount(1)
        self.table4.setColumnCount(5)
        self.table4.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                color: rgba(255,255,255,255);
                border: none;
                border-radius: 5px;
            }
        """)
        self.table4.setShowGrid(False)
        self.table4.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        self.table4.setSelectionMode(QAbstractItemView.NoSelection)   # Disable cell selection
        
        # Disable scroll bars
        self.table4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set row height and column width to fit the table4's fixed size
        table4_width = self.table4.width()
        table4_height = self.table4.height()
        row4_height = table4_height // self.table4.rowCount()
        column4_width = table4_width // self.table4.columnCount()

        for row4 in range(self.table4.rowCount()):
            self.table4.setRowHeight(row4, row4_height)
        
        for column4 in range(self.table4.columnCount()):
            self.table4.setColumnWidth(column4, column4_width)
                
        self.image_label2 = ImageLabel("Conventional CVML", apply_effects = False)
        self.image_label2.setFixedWidth(int(0.4244791667 * self.screen_width))
        self.image_label2.setFixedHeight(int(0.25 * table_height))
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setScaledContents(True)
        self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
        
        font = QFont()
        font.setPointSize(int(0.009895833 * self.screen_width))
        font.setBold(False)
        self.image_label2.setFont(font)
        self.table.setFont(font)
        self.table1.setFont(font)
        self.table2.setFont(font)
        self.table3.setFont(font)
        self.table4.setFont(font)
        
        self.table_layout_VBOX = QVBoxLayout()
        self.table_layout_VBOX.setSpacing(0)
        self.table_layout_VBOX.setContentsMargins(0, 0, 0, 0)
        
        self.table_layout_VBOX.addWidget(self.table1)
        self.table_layout_VBOX.addWidget(self.table2)
        self.table_layout_VBOX.addWidget(self.table3)
        self.table_layout_VBOX.addWidget(self.table4)

        self.table_layout_VBOX_Widget = QWidget()

        self.table_layout_VBOX_Widget.setFixedHeight(table_height)
        self.table_layout_VBOX_Widget.setFixedWidth(int(2.15 *table_height))
        self.table_layout_VBOX_Widget.setLayout(self.table_layout_VBOX)
        self.table_layout_VBOX_Widget.setStyleSheet("background-color: transparent; color: rgba(254,229,2,255);")
        
        self.table.setVisible(False)
        
        self.table_layout_VBOX_Widget.setVisible(False)
        self.table1.setVisible(False)
        self.table2.setVisible(False)
        self.table3.setVisible(False)
        self.table4.setVisible(False)
        
        spacerdashed1 = QSpacerItem(int(0.002604166 * self.screen_width), 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacerdashed2 = QSpacerItem(int(0.002604166 * self.screen_width), 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        self.table_layout = QHBoxLayout()
        self.table_layout.setSpacing(0)
        self.table_layout.setContentsMargins(0, 0, 0, 0)
        self.table_layout.addSpacerItem(spacerdashed1)
        self.table_layout.addWidget(self.image_label2)
        self.table_layout.addWidget(self.table)
        self.table_layout.addWidget(self.table_layout_VBOX_Widget)
        self.table_layout.addSpacerItem(spacerdashed2)

        self.table_layout_Widget = QWidget()
        self.table_layout_Widget.setLayout(self.table_layout)
        self.table_layout_Widget.setStyleSheet("background-color: transparent; color: rgba(254,229,2,255);")
        self.table_layout_Widget.setVisible(True)
        
        HBoxLayout1 = QHBoxLayout()
        HBoxLayout1.addWidget(self.image_label)
        VBoxLayout1 = QVBoxLayout()
        VBoxLayout1.addLayout(HBoxLayout1)
        VBoxLayout1.addWidget(self.table_layout_Widget)
        self.setLayout(VBoxLayout1)
# __________________________________________________ Functions _______________________________________________________
    def handle_variable_changed(self, new_value):
        self.FirstSetVariable = new_value
        # print("FirstSetVariable:", self.FirstSetVariable)
            
    def handle_variable_changed_Equalized(self, new_value):
        self.perform_intensity_normalization = new_value

    def mouseDoubleClickEvent(self, event):
        self.main_window.load_main_img()
    
    def on_button_click(self):
        self.file_path = None
        self.image = None
        self.pixmap = None
        self.rounded_pixmap = None
        self.image_label.clear()
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.image_label, image_path)
        
        self.Features_input = []
        self.Features_input_LBP = []
        self.Features_input_LTP = []
        self.Features_input_HOG = []
        self.set_image(None)
        self.set_path(None)
        
        self.on_combo_clear()
        
        
    def on_combo_clear(self):
        self.output2 = None
        self.class2 = None
        self.output3 = None
        self.class3 = None
        self.output5 = None
        self.class5 = None
        self.output5_Tree = None
        self.class5_Tree = None
        self.score_Class2 = None
        self.score_Class3 = None
        self.score_Class5 = None
        self.score_Class5_Tree = None
        
        self.table.clearContents()
        self.table1.clearContents()
        self.table2.clearContents()
        self.table3.clearContents()
        self.table4.clearContents()
        
        self.table_layout_VBOX_Widget.setVisible(False)
        self.table.setVisible(False)
        self.table1.setVisible(False)
        self.table2.setVisible(False)
        self.table3.setVisible(False)
        self.table4.setVisible(False)
        
        self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
        self.image_label2.setText("Conventional CVML")
        self.image_label2.setVisible(True)
        
        
    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")
    
    def set_image(self, img):
        self.Feature_Extraction_and_Visualization_Screen.Conventional_image = img
    
    def set_path(self, file_path):
        self.Feature_Extraction_and_Visualization_Screen.Conventional_image_path = file_path
            
            
            
    def carry_out(self):        
        self.set_image(self.image)
        self.set_path(self.file_path)
        
        self.Feature_Extraction_and_Visualization_Screen.load_Conventional_Image()
        
        self.Features_input = [
                               self.Feature_Extraction_and_Visualization_Screen.get_Medial_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_Central_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_Lateral_ratio(),
                               self.Feature_Extraction_and_Visualization_Screen.get_medial_area_Ratio_TWPA(),
                               self.Feature_Extraction_and_Visualization_Screen.get_central_area_Ratio_TWPA(),
                               self.Feature_Extraction_and_Visualization_Screen.get_lateral_area_Ratio_TWPA(),
                               self.Feature_Extraction_and_Visualization_Screen.get_intensity_mean(),
                               self.Feature_Extraction_and_Visualization_Screen.get_intensity_stddev(),
                               self.Feature_Extraction_and_Visualization_Screen.get_intensity_skewness(),
                               self.Feature_Extraction_and_Visualization_Screen.get_intensity_kurtosis(),
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['contrast'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['energy'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['correlation'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['homogeneity'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['dissimilarity'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['ASM'],
                               self.Feature_Extraction_and_Visualization_Screen.get_cooccurrence_properties()['max_probability']
        ]
        
        self.Features_input_LBP = self.Feature_Extraction_and_Visualization_Screen.get_lbp_features_Normalized()
        self.Features_input_LBP = self.Features_input_LBP.tolist()
        
        self.Features_input_LTP = self.Feature_Extraction_and_Visualization_Screen.get_ltp_features_Normalized()
        self.Features_input_LTP = self.Features_input_LTP.tolist()
        
        
        self.Features_input_Other_Features = [self.Features_input 
                                              
                                              + self.Features_input_LBP
                                              
                                              + [self.Feature_Extraction_and_Visualization_Screen.get_lbp_variance_Normalized(),
                                                self.Feature_Extraction_and_Visualization_Screen.get_lbp_entropy_Normalized()]
                                              
                                              + self.Features_input_LTP
                                              + [self.Feature_Extraction_and_Visualization_Screen.get_ltp_variance_Normalized(),
                                                self.Feature_Extraction_and_Visualization_Screen.get_ltp_entropy_Normalized()]
        
        ]
        
        
        self.Features_input_HOG = self.Feature_Extraction_and_Visualization_Screen.get_HOG_Normalized()
        self.Features_input_HOG = [self.Features_input_HOG.tolist()]
        
        
        
        del self.Features_input
        del self.Features_input_LBP
        del self.Features_input_LTP
        
        
        
        if self.SR == 1:
        
            if self.load_cvml_Two_clases_models_Flag == 0:
                self.load_cvml_Two_clases_models()
                
            if self.load_cvml_Three_clases_models_Flag == 0:
                self.load_cvml_Three_clases_models()
            
            if self.load_cvml_Five_clases_models_Flag == 0:
                self.load_cvml_Five_clases_models()
            
            if self.load_cvml_Five_clases_Binary_Tree_models_Flag == 0:
                self.load_cvml_Five_clases_Binary_Tree_models()
                # time.sleep(5)
                
                
            
            
            
            self.class2, self.output2 = self.get_Prediction_cvml_Two_clases_models()
            self.class3, self.output3 = self.get_Prediction_cvml_Three_clases_models()
            self.class5, self.output5 = self.get_Prediction_cvml_Five_clases_models(self.Features_input_Other_Features, self.Features_input_HOG)
            self.class5_Tree, self.output5_Tree = self.get_Prediction_cvml_Five_clases_Binary_Tree_models(self.Features_input_Other_Features, self.Features_input_HOG)
            
        if self.SR == 2:
        
        
            if self.load_cvml_Two_clases_models_70_30_Flag == 0:
                self.load_cvml_Two_clases_models_70_30()
                        
            if self.load_cvml_Three_clases_models_70_30_Flag == 0:
                self.load_cvml_Three_clases_models_70_30()
            
            if self.load_cvml_Five_clases_models_70_30_Flag == 0:
                self.load_cvml_Five_clases_models_70_30()
            
            if self.load_cvml_Five_clases_Binary_Tree_models_70_30_Flag == 0:
                self.load_cvml_Five_clases_Binary_Tree_models_70_30()
                # time.sleep(5)
        
           
            
            self.class2, self.output2 = self.get_Prediction_cvml_Two_clases_models_70_30()
            self.class3, self.output3 = self.get_Prediction_cvml_Three_clases_models_70_30()
            self.class5, self.output5 = self.get_Prediction_cvml_Five_clases_models_70_30(self.Features_input_Other_Features, self.Features_input_HOG)
            self.class5_Tree, self.output5_Tree = self.get_Prediction_cvml_Five_clases_Binary_Tree_models_70_30(self.Features_input_Other_Features, self.Features_input_HOG)
            
        if self.SR == 3:
        
            if self.load_cvml_Two_clases_models_60_40_Flag == 0:
                self.load_cvml_Two_clases_models_60_40()
                        
            if self.load_cvml_Three_clases_models_60_40_Flag == 0:
                self.load_cvml_Three_clases_models_60_40()
            
            if self.load_cvml_Five_clases_models_60_40_Flag == 0:
                self.load_cvml_Five_clases_models_60_40()
            
            if self.load_cvml_Five_clases_Binary_Tree_models_60_40_Flag == 0:
                self.load_cvml_Five_clases_Binary_Tree_models_60_40()
                # time.sleep(5)
                
                
            self.class2, self.output2 = self.get_Prediction_cvml_Two_clases_models_60_40()
            self.class3, self.output3 = self.get_Prediction_cvml_Three_clases_models_60_40()
            self.class5, self.output5 = self.get_Prediction_cvml_Five_clases_models_60_40(self.Features_input_Other_Features, self.Features_input_HOG)
            self.class5_Tree, self.output5_Tree = self.get_Prediction_cvml_Five_clases_Binary_Tree_models_60_40(self.Features_input_Other_Features, self.Features_input_HOG)
            
        
        print(f"self.class2: {self.class2}, self.output2: {self.output2}, self.class3: {self.class3}, self.output3: {self.output3}, self.class5: {self.class5}, self.output5: {self.output5}, self.class5_Tree: {self.class5_Tree}, self.output5_Tree: {self.output5_Tree}")
        
        self.score_Class2 = 100*self.output2[0][self.class2]
        self.score_Class3 = 100*self.output3[0][self.class3]
        self.score_Class5 = 100*self.output5[0][self.class5]
        self.score_Class5_Tree = 100*self.output5_Tree[self.class5_Tree]
        
        self.show_Predictions()
        
        # del self.Features_input_Other_Features
        # del self.Features_input_HOG
        
    
    def show_Predictions(self):
        # 2 Class
        if self.FirstSetVariable == 1:
            if self.class2 is not None:
                self.table.setVisible(False)
            
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(False)
                self.table3.setVisible(False)
                self.table4.setVisible(False)
                self.image_label2.setVisible(True)
            
                classes = [
                    ("Normal", QColor(0, 200, 0, 140)),
                    ("OsteoArthritis", QColor(255, 0, 0, 140))
                ]

                for idx, (label, color) in enumerate(classes):
                    if self.class2 == idx:
                        self.image_label2.setText(f"Normal Vs OsteoArthritis: <b>{label}</b>, with Highest score: <b>{self.score_Class2:0.1f}</b>%")
                        self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                        break

                self.item1 = QTableWidgetItem(f"{100*self.output2[0][0]:0.1f}%")
                self.item1.setTextAlignment(Qt.AlignCenter)
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item1.setBackground(QBrush(color))
                self.table1.setItem(0, 0, self.item1)
                
                self.item2 = QTableWidgetItem(f"{100*self.output2[0][1]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item2.setBackground(QBrush(color))
                self.item2.setTextAlignment(Qt.AlignCenter)
                self.table1.setItem(0, 1, self.item2)
                          
            else:
                self.table.setItem(0, 1, QTableWidgetItem(""))
                
                self.table1.setItem(0, 0, QTableWidgetItem(""))
                self.table1.setItem(0, 1, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table1.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table1.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                self.image_label2.setText("Conventional CVML")
                
        # 3 Class
        elif self.FirstSetVariable == 2:
            if self.class3 is not None:
                self.table.setVisible(False)
            
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(False)
                self.table2.setVisible(True)
                self.table3.setVisible(False)
                self.table4.setVisible(False)
                self.image_label2.setVisible(True)

                classes = [
                    ("Normal", QColor(0, 200, 0, 140)),
                    ("Mild", QColor(200, 200, 0, 140)),
                    ("Severe", QColor(255, 0, 0, 140))
                ]

                for idx, (label, color) in enumerate(classes):
                    if self.class3 == idx:
                        self.image_label2.setText(f"Normal Vs Mild Vs Severe: <b>{label}</b>, with Highest score: <b>{self.score_Class3:0.1f}</b>%")
                        self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                        break

                self.item3 = QTableWidgetItem(f"{100*self.output3[0][0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item3.setBackground(QBrush(color))
                self.item3.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 0, self.item3)
                
                
                self.item4 = QTableWidgetItem(f"{100*self.output3[0][1]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                self.item4.setBackground(QBrush(color))
                self.item4.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 1, self.item4)
                
                
                self.item5 = QTableWidgetItem(f"{100*self.output3[0][2]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item5.setBackground(QBrush(color))
                self.item5.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 2, self.item5)
                    
            else:
                self.table.setItem(1, 1, QTableWidgetItem(""))
                
                self.table2.setItem(0, 0, QTableWidgetItem(""))
                self.table2.setItem(0, 1, QTableWidgetItem(""))
                self.table2.setItem(0, 2, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table2.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table2.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                self.image_label2.clear()
                self.image_label2.setText("Conventional CVML")
                    

        # 5 Class
        elif self.FirstSetVariable == 3:
            if self.class5 is not None:
                self.table.setVisible(False)
                
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(False)
                self.table2.setVisible(False)
                self.table3.setVisible(True)
                self.table4.setVisible(False)
                self.image_label2.setVisible(True)
            
                classes = [
                    ("Normal", QColor(0, 200, 0, 140)),
                    ("Doubtful", QColor(100, 200, 0, 140)),
                    ("Mild", QColor(200, 200, 0, 140)),
                    ("Moderate", QColor(227, 100, 0, 140)),
                    ("Severe", QColor(255, 0, 0, 140))
                ]

                for idx, (label, color) in enumerate(classes):
                    if self.class5 == idx:
                        self.image_label2.setText(f"Kellgren-Lawrence 5-Classes: <b>{label}</b>, with Highest score: <b>{self.score_Class5:0.1f}</b>%")
                        self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                        break

                self.item6 = QTableWidgetItem(f"{100*self.output5[0][0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item6.setBackground(QBrush(color))
                self.item6.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 0, self.item6)
                
                self.item7 = QTableWidgetItem(f"{100*self.output5[0][1]:0.1f}%")
                color = QColor(*(100, 200, 0))
                color.setAlpha(140)
                self.item7.setBackground(QBrush(color))
                self.item7.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 1, self.item7)

                self.item8 = QTableWidgetItem(f"{100*self.output5[0][2]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                self.item8.setBackground(QBrush(color))
                self.item8.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 2, self.item8)
                
                self.item9 = QTableWidgetItem(f"{100*self.output5[0][3]:0.1f}%")
                color = QColor(*(227, 100, 0))
                color.setAlpha(140)
                self.item9.setBackground(QBrush(color))
                self.item9.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 3, self.item9)
                
                self.item10 = QTableWidgetItem(f"{100*self.output5[0][4]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item10.setBackground(QBrush(color))
                self.item10.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 4, self.item10)
            
            else:
                self.table.setItem(2, 1, QTableWidgetItem(""))
                
                self.table3.setItem(0, 0, QTableWidgetItem(""))
                self.table3.setItem(0, 1, QTableWidgetItem(""))
                self.table3.setItem(0, 2, QTableWidgetItem(""))
                self.table3.setItem(0, 3, QTableWidgetItem(""))
                self.table3.setItem(0, 4, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table3.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table3.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                self.image_label2.setText("Conventional CVML") 

        #  Binary Tree Fusion 
        elif self.FirstSetVariable == 4:
            if self.class5_Tree is not None:
                self.table.setVisible(False)
                
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(False)
                self.table2.setVisible(False)
                self.table3.setVisible(False)
                self.table4.setVisible(True)
                self.image_label2.setVisible(True)

                classes = [
                    ("Normal", QColor(0, 200, 0, 140)),
                    ("Doubtful", QColor(100, 200, 0, 140)),
                    ("Mild", QColor(200, 200, 0, 140)),
                    ("Moderate", QColor(227, 100, 0, 140)),
                    ("Severe", QColor(255, 0, 0, 140))
                ]

                for idx, (label, color) in enumerate(classes):
                    if self.class5_Tree == idx:
                        self.image_label2.setText(f"Probability voting 5-Classes: <b>{label}</b>, with Highest score: <b>{self.score_Class5_Tree:0.1f}</b>%")
                        self.image_label2.setStyleSheet(f"color: white; background-color: {color.name(QColor.HexArgb)};")
                        break

                self.item11 = QTableWidgetItem(f"{100*self.output5_Tree[0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                self.item11.setBackground(QBrush(color))
                self.item11.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 0, self.item11)
                
                
                self.item12 = QTableWidgetItem(f"{100*self.output5_Tree[1]:0.1f}%")
                color = QColor(*(100, 200, 0))
                color.setAlpha(140)
                self.item12.setBackground(QBrush(color))
                self.item12.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 1, self.item12)
                

                self.item13 = QTableWidgetItem(f"{100*self.output5_Tree[2]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                self.item13.setBackground(QBrush(color))
                self.item13.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 2, self.item13)
                
                
                self.item14 = QTableWidgetItem(f"{100*self.output5_Tree[3]:0.1f}%")
                color = QColor(*(227, 100, 0))
                color.setAlpha(140)
                self.item14.setBackground(QBrush(color))
                self.item14.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 3, self.item14)
                
                
                self.item15 = QTableWidgetItem(f"{100*self.output5_Tree[4]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                self.item15.setBackground(QBrush(color))
                self.item15.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 4, self.item15)
                    
            else:
                
                self.table4.setItem(0, 0, QTableWidgetItem(""))
                self.table4.setItem(0, 1, QTableWidgetItem(""))
                self.table4.setItem(0, 2, QTableWidgetItem(""))
                self.table4.setItem(0, 3, QTableWidgetItem(""))
                self.table4.setItem(0, 4, QTableWidgetItem(""))
                    
                pass
                self.table.setVisible(False)
                self.table4.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                
                self.table.clearContents()
                self.table4.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                self.image_label2.setText("Conventional CVML")
                    
                    
        
        elif self.FirstSetVariable == 5:
        
            self.item_main1 = QTableWidgetItem("Normal Vs OsteoArthritis")
            self.table.setItem(0, 0, self.item_main1)
            self.item_main2 = QTableWidgetItem("Normal Vs Mild Vs Severe")
            self.table.setItem(1, 0, self.item_main2)
            self.item_main3 = QTableWidgetItem("Kellgren-Lawrence 5-Classes")
            self.table.setItem(2, 0, self.item_main3)
            self.item_main4 = QTableWidgetItem("Probability voting 5-Classes")
            self.table.setItem(3, 0, self.item_main4)
    
            if (self.class5_Tree is not None):
                
                self.table.setVisible(True)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(True)
                self.image_label2.setVisible(False)
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
            
                colors = [(0, 200, 0), (100, 200, 0),(200, 200, 0), (227, 100, 0), (255, 0, 0)]
                descriptions = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
                
                if 0 <= self.class5_Tree < len(descriptions):
                    item_tree = QTableWidgetItem(f"{descriptions[self.class5_Tree]}, with Highest score: {self.score_Class5_Tree:0.1f}%")
                    color = QColor(*colors[self.class5_Tree])
                        
                color.setAlpha(140)
                item_tree.setBackground(QBrush(color))
                self.table.setItem(3, 1, item_tree)
                
                self.item_main4.setBackground(QBrush(color))
                self.item_main4.setTextAlignment(Qt.AlignCenter)

                item11 = QTableWidgetItem(f"{100*self.output5_Tree[0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                item11.setBackground(QBrush(color))
                item11.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 0, item11)
            
                item12 = QTableWidgetItem(f"{100*self.output5_Tree[1]:0.1f}%")
                color = QColor(*(100, 200, 0))
                color.setAlpha(140)
                item12.setBackground(QBrush(color))
                item12.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 1, item12)
            
                item13 = QTableWidgetItem(f"{100*self.output5_Tree[2]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                item13.setBackground(QBrush(color))
                item13.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 2, item13)
                
                item14 = QTableWidgetItem(f"{100*self.output5_Tree[3]:0.1f}%")
                color = QColor(*(227, 100, 0))
                color.setAlpha(140)
                item14.setBackground(QBrush(color))
                item14.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 3, item14)
            
                item15 = QTableWidgetItem(f"{100*self.output5_Tree[4]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                item15.setBackground(QBrush(color))
                item15.setTextAlignment(Qt.AlignCenter)
                self.table4.setItem(0, 4, item15)
                
            else:
                self.table.setItem(3, 1, QTableWidgetItem(""))
                
                self.table4.setItem(0, 0, QTableWidgetItem(""))
                self.table4.setItem(0, 1, QTableWidgetItem(""))
                self.table4.setItem(0, 2, QTableWidgetItem(""))
                self.table4.setItem(0, 3, QTableWidgetItem(""))
                self.table4.setItem(0, 4, QTableWidgetItem(""))
                    
                pass
                self.table.setVisible(False)
                self.table4.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                self.table.clearContents()
                self.table4.clearContents()
                self.image_label2.setText("Conventional CVML")
                
                
            if (self.class5 is not None):
                
                self.table.setVisible(True)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(True)
                
                self.image_label2.setVisible(False)
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                colors = [(0, 200, 0), (100, 200, 0),(200, 200, 0), (227, 100, 0), (255, 0, 0)]
                descriptions = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
                
                if self.class5 in range(len(descriptions)):
                    item_class5 = QTableWidgetItem(f"{descriptions[self.class5]}, with Highest score: {self.score_Class5:0.1f}%")
                    color = QColor(*colors[self.class5])
                        
                color.setAlpha(140)
                item_class5.setBackground(QBrush(color))
                self.table.setItem(2, 1, item_class5)
                
                self.item_main3.setBackground(QBrush(color))
                self.item_main3.setTextAlignment(Qt.AlignCenter)
                
                item6 = QTableWidgetItem(f"{100*self.output5[0][0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                item6.setBackground(QBrush(color))
                item6.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 0, item6)
            
            
                item7 = QTableWidgetItem(f"{100*self.output5[0][1]:0.1f}%")
                color = QColor(*(100, 200, 0))
                color.setAlpha(140)
                item7.setBackground(QBrush(color))
                item7.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 1, item7)
            
                item8 = QTableWidgetItem(f"{100*self.output5[0][2]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                item8.setBackground(QBrush(color))
                item8.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 2, item8)
            
                item9 = QTableWidgetItem(f"{100*self.output5[0][3]:0.1f}%")
                color = QColor(*(227, 100, 0))
                color.setAlpha(140)
                item9.setBackground(QBrush(color))
                item9.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 3, item9)
                
                item10 = QTableWidgetItem(f"{100*self.output5[0][4]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                item10.setBackground(QBrush(color))
                item10.setTextAlignment(Qt.AlignCenter)
                self.table3.setItem(0, 4, item10)
                
            else:
                self.table.setItem(2, 1, QTableWidgetItem(""))
                
                self.table3.setItem(0, 0, QTableWidgetItem(""))
                self.table3.setItem(0, 1, QTableWidgetItem(""))
                self.table3.setItem(0, 2, QTableWidgetItem(""))
                self.table3.setItem(0, 3, QTableWidgetItem(""))
                self.table3.setItem(0, 4, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table3.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table3.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                self.image_label2.setText("Conventional CVML")
                
            if (self.class3 is not None):
                
                self.table.setVisible(True)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(True)

                self.image_label2.setVisible(False)
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                colors = [(0, 200, 0), (200, 200, 0), (255, 0, 0)]
                descriptions = ["Normal", "Mild", "Severe"]
                
                if self.class3 in range(len(descriptions)):
                    item_class3 = QTableWidgetItem(f"{descriptions[self.class3]}, with Highest score: {self.score_Class3:0.1f}%")
                    color = QColor(*colors[self.class3])

                color.setAlpha(140)
                item_class3.setBackground(QBrush(color))
                self.table.setItem(1, 1, item_class3)
                
                self.item_main2.setBackground(QBrush(color))
                self.item_main2.setTextAlignment(Qt.AlignCenter)
                
                
                
                item3 = QTableWidgetItem(f"{100*self.output3[0][0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                item3.setBackground(QBrush(color))
                item3.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 0, item3)
                
                
                item4 = QTableWidgetItem(f"{100*self.output3[0][1]:0.1f}%")
                color = QColor(*(200, 200, 0))
                color.setAlpha(140)
                item4.setBackground(QBrush(color))
                item4.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 1, item4)
            
            
                item5 = QTableWidgetItem(f"{100*self.output3[0][2]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                item5.setBackground(QBrush(color))
                item5.setTextAlignment(Qt.AlignCenter)
                self.table2.setItem(0, 2, item5)
                
            else:
                self.table.setItem(1, 1, QTableWidgetItem(""))
                
                self.table2.setItem(0, 0, QTableWidgetItem(""))
                self.table2.setItem(0, 1, QTableWidgetItem(""))
                self.table2.setItem(0, 2, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table2.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table2.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                self.image_label2.setText("Conventional CVML")
                
                
            if (self.class2 is not None):
                
                self.table.setVisible(True)
                self.table_layout_VBOX_Widget.setVisible(True)
                self.table1.setVisible(True)
                self.table2.setVisible(True)
                self.table3.setVisible(True)
                self.table4.setVisible(True)
                
                self.image_label2.setVisible(False)
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                
                
                colors = [(0, 200, 0), (255, 0, 0)]
                descriptions = ["Normal", "Osteo"]
                
                if self.class2 in range(len(descriptions)):
                    item_clas2 = QTableWidgetItem(f"{descriptions[self.class2]}, with Highest score: {self.score_Class2:0.1f}%")
                    color = QColor(*colors[self.class2])
                
                color.setAlpha(140)
                item_clas2.setBackground(QBrush(color))
                self.table.setItem(0, 1, item_clas2)
                
                self.item_main1.setBackground(QBrush(color))
                self.item_main1.setTextAlignment(Qt.AlignCenter)
                
                item1 = QTableWidgetItem(f"{100*self.output2[0][0]:0.1f}%")
                color = QColor(*(0, 200, 0))
                color.setAlpha(140)
                item1.setBackground(QBrush(color))
                item1.setTextAlignment(Qt.AlignCenter)
                self.table1.setItem(0, 0, item1)
                
                item2 = QTableWidgetItem(f"{100*self.output2[0][1]:0.1f}%")
                color = QColor(*(255, 0, 0))
                color.setAlpha(140)
                item2.setBackground(QBrush(color))
                item2.setTextAlignment(Qt.AlignCenter)
                self.table1.setItem(0, 1, item2)
                    
            else:
                self.table.setItem(0, 1, QTableWidgetItem(""))
                
                self.table1.setItem(0, 0, QTableWidgetItem(""))
                self.table1.setItem(0, 1, QTableWidgetItem(""))
                
                pass
                self.table.setVisible(False)
                self.table1.setVisible(False)
                
                self.image_label2.setVisible(True)
                self.image_label2.clear()
                self.table.clearContents()
                self.table1.clearContents()
                self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
                self.image_label2.setText("Conventional CVML")

        else:
            pass    
            self.table.setVisible(False)
            
            self.table1.setVisible(False)
            self.table2.setVisible(False)
            self.table3.setVisible(False)
            self.table4.setVisible(False)
            
            self.image_label2.setVisible(True)
            self.table.clearContents()
            self.image_label2.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: rgba(255,255,255,255);")
            
            self.table1.clearContents()
            self.table2.clearContents()
            self.table3.clearContents()
            self.table4.clearContents()
            
            self.image_label2.clear()
            self.image_label2.setText("Conventional CVML")
            
            
    def get_Prediction_cvml_Two_clases_models(self):
        # 2 Classes Classification model with average method
        Features_input_Other_Features = self.scaler_normal_vs_osteo_other_features.transform(self.Features_input_Other_Features)
        Random_Forest_chen_and_expert_Other_2_Class_pred = self.MLP_Normal_vs_osteo_other_features.predict_proba(Features_input_Other_Features)
    
        Features_input_HOG =self.scaler_normal_vs_osteo_HOG.transform(self.Features_input_HOG)
        MLP_Normal_vs_osteo_HOG_2_Class_pred = self.MLP_Normal_vs_osteo_HOG.predict_proba(Features_input_HOG)
    
        output_prob = (Random_Forest_chen_and_expert_Other_2_Class_pred +
                        MLP_Normal_vs_osteo_HOG_2_Class_pred )/ 2
    
        Two_Class_Avg_pred = np.argmax(output_prob)
        
        return Two_Class_Avg_pred, output_prob

    def get_Prediction_cvml_Three_clases_models(self):
        # 3 Classes Classification model with average method
        
        Features_input_Other_Features = self.scaler_3_Classes_other_features.transform(self.Features_input_Other_Features)
        MLP_chen_and_expert_Other_3_Class_pred = self.MLP_3_Classes_other_features.predict_proba(Features_input_Other_Features)
    
        Features_input_HOG =self.scaler_3_Classes_HOG.transform(self.Features_input_HOG)
        MLP_3_Classes_HOG_pred = self.MLP_3_Classes_HOG.predict_proba(Features_input_HOG)
    
        output_prob = (MLP_chen_and_expert_Other_3_Class_pred +
                        MLP_3_Classes_HOG_pred )/ 2
    
        Three_Class_Avg_pred = np.argmax(output_prob)

        return Three_Class_Avg_pred, output_prob
    
    def get_Prediction_cvml_Five_clases_models(self, Features_input_Other_Features, Features_input_HOG):
        # 5 Classes Classification model with average method
        
        sample = self.scaler_Chenn_and_Expert_Other_Features_5_Class.transform(Features_input_Other_Features)
        Random_Forest_chen_and_expert_Other_5_Class_pred = self.Random_Forest_Chenn_and_Expert_Other_Features_5_Class.predict_proba(sample)
    
        sample =self.scaler_Chenn_and_Expert_HOG_5_Class.transform(Features_input_HOG)
        MLP_chen_and_expert_HOG_5_Class_pred = self.MLP_Chenn_and_Expert_HOG_5_Class.predict_proba(sample)
    
        output_prob = (Random_Forest_chen_and_expert_Other_5_Class_pred +
                        MLP_chen_and_expert_HOG_5_Class_pred )/ 2
    
        Five_Class_Avg_pred = np.argmax(output_prob)
        
        return Five_Class_Avg_pred, output_prob
    
    
    def get_Prediction_cvml_Five_clases_Binary_Tree_models(self, Features_input_Other_Features, Features_input_HOG):
        # 5 Classes Classification Binary Tree method with binary moddels using average method
        # predict Other Features
        
        sample = self.scaler_0_vs_1_other_features.transform(Features_input_Other_Features)
        predict_0_Vs_1_other_features = self.RF_0_vs_1_other_features.predict_proba(sample)

        sample = self.scaler_0_vs_2_other_features.transform(Features_input_Other_Features)
        predict_0_Vs_2_other_features = self.MLP_0_vs_2_other_features.predict_proba(sample)

        sample = self.scaler_0_vs_3_other_features.transform(Features_input_Other_Features)
        predict_0_Vs_3_other_features = self.MLP_0_vs_3_other_features.predict_proba(sample)

        sample = self.scaler_0_vs_4_other_features.transform(Features_input_Other_Features)
        predict_0_Vs_4_other_features = self.MLP_0_vs_4_other_features.predict_proba(sample)

        sample = self.scaler_1_vs_2_other_features.transform(Features_input_Other_Features)
        predict_1_Vs_2_other_features = self.RF_1_vs_2_other_features.predict_proba(sample)

        sample = self.scaler_1_vs_3_other_features.transform(Features_input_Other_Features)
        predict_1_Vs_3_other_features = self.MLP_1_vs_3_other_features.predict_proba(sample)

        sample = self.scaler_1_vs_4_other_features.transform(Features_input_Other_Features)
        predict_1_Vs_4_other_features = self.MLP_1_vs_4_other_features.predict_proba(sample)

        sample = self.scaler_2_vs_3_other_features.transform(Features_input_Other_Features)
        predict_2_Vs_3_other_features = self.RF_2_vs_3_other_features.predict_proba(sample)

        sample = self.scaler_2_vs_4_other_features.transform(Features_input_Other_Features)
        predict_2_Vs_4_other_features = self.MLP_2_vs_4_other_features.predict_proba(sample)

        sample = self.scaler_3_vs_4_other_features.transform(Features_input_Other_Features)
        predict_3_Vs_4_other_features = self.MLP_3_vs_4_other_features.predict_proba(sample)


        # predict HOG Features 
        sample = self.scaler_0_vs_1_HOG.transform(Features_input_HOG)
        predict_0_Vs_1_HOG = self.MLP_0_vs_1_HOG.predict_proba(sample)

        sample = self.scaler_0_vs_2_HOG.transform(Features_input_HOG)
        predict_0_Vs_2_HOG = self.MLP_0_vs_2_HOG.predict_proba(sample)

        sample = self.scaler_0_vs_3_HOG.transform(Features_input_HOG)
        predict_0_Vs_3_HOG = self.MLP_0_vs_3_HOG.predict_proba(sample)

        sample = self.scaler_0_vs_4_HOG.transform(Features_input_HOG)
        predict_0_Vs_4_HOG = self.LR_0_vs_4_HOG.predict_proba(sample)

        sample = self.scaler_1_vs_2_HOG.transform(Features_input_HOG)
        predict_1_Vs_2_HOG = self.SVM_1_vs_2_HOG.predict_proba(sample)

        sample = self.scaler_1_vs_3_HOG.transform(Features_input_HOG)
        predict_1_Vs_3_HOG = self.SVM_1_vs_3_HOG.predict_proba(sample)

        sample = self.scaler_1_vs_4_HOG.transform(Features_input_HOG)
        predict_1_Vs_4_HOG = self.MLP_1_vs_4_HOG.predict_proba(sample)

        sample = self.scaler_2_vs_3_HOG.transform(Features_input_HOG)
        predict_2_Vs_3_HOG = self.MLP_2_vs_3_HOG.predict_proba(sample)

        sample = self.scaler_2_vs_4_HOG.transform(Features_input_HOG)
        predict_2_Vs_4_HOG = self.MLP_2_vs_4_HOG.predict_proba(sample)

        sample = self.scaler_3_vs_4_HOG.transform(Features_input_HOG)
        predict_3_Vs_4_HOG = self.SVM_3_vs_4_HOG.predict_proba(sample)


        del sample

        # predict with avg method 
        predict_0_Vs_1_prob =  (predict_0_Vs_1_HOG + predict_0_Vs_1_other_features) / 2 
        predict_0_Vs_2_prob =  (predict_0_Vs_2_HOG + predict_0_Vs_2_other_features) / 2
        predict_0_Vs_3_prob =  (predict_0_Vs_3_HOG + predict_0_Vs_3_other_features) / 2
        predict_0_Vs_4_prob =  (predict_0_Vs_4_HOG + predict_0_Vs_4_other_features) / 2
        predict_1_Vs_2_prob =  (predict_1_Vs_2_HOG + predict_1_Vs_2_other_features) / 2 
        predict_1_Vs_3_prob =  (predict_1_Vs_3_HOG + predict_1_Vs_3_other_features) / 2 
        predict_1_Vs_4_prob =  (predict_1_Vs_4_HOG + predict_1_Vs_4_other_features) / 2 
        predict_2_Vs_3_prob =  (predict_2_Vs_3_HOG + predict_2_Vs_3_other_features) / 2
        predict_2_Vs_4_prob =  (predict_2_Vs_4_HOG + predict_2_Vs_4_other_features) / 2 
        predict_3_Vs_4_prob =  (predict_3_Vs_4_HOG + predict_3_Vs_4_other_features) / 2

        
        del predict_0_Vs_1_HOG
        del predict_0_Vs_1_other_features
        del predict_0_Vs_2_HOG
        del predict_0_Vs_2_other_features
        del predict_0_Vs_3_HOG
        del predict_0_Vs_3_other_features
        del predict_0_Vs_4_HOG
        del predict_0_Vs_4_other_features
        del predict_1_Vs_2_other_features
        del predict_1_Vs_2_HOG
        del predict_1_Vs_3_HOG
        del predict_1_Vs_3_other_features
        del predict_1_Vs_4_HOG
        del predict_1_Vs_4_other_features
        del predict_2_Vs_3_HOG
        del predict_2_Vs_3_other_features
        del predict_3_Vs_4_HOG
        del predict_2_Vs_4_other_features
        del predict_3_Vs_4_other_features
        del predict_2_Vs_4_HOG
        
        num_classes = 5

        P = np.zeros((num_classes, num_classes))
        # Populate the matrix with average probabilities
        P[0, 1] = predict_0_Vs_1_prob[0][0]
        P[1, 0] = predict_0_Vs_1_prob[0][1]

        P[0, 2] = predict_0_Vs_2_prob[0][0]
        P[2, 0] = predict_0_Vs_2_prob[0][1]

        P[0, 3] = predict_0_Vs_3_prob[0][0]
        P[3, 0] = predict_0_Vs_3_prob[0][1]

        P[0, 4] = predict_0_Vs_4_prob[0][0]
        P[4, 0] = predict_0_Vs_4_prob[0][1]

        P[1, 2] = predict_1_Vs_2_prob[0][0]
        P[2, 1] = predict_1_Vs_2_prob[0][1]

        P[1, 3] = predict_1_Vs_3_prob[0][0]
        P[3, 1] = predict_1_Vs_3_prob[0][1]

        P[1, 4] = predict_1_Vs_4_prob[0][0]
        P[4, 1] = predict_1_Vs_4_prob[0][1]

        P[2, 3] = predict_2_Vs_3_prob[0][0]
        P[3, 2] = predict_2_Vs_3_prob[0][1]

        P[2, 4] = predict_2_Vs_4_prob[0][0]
        P[4, 2] = predict_2_Vs_4_prob[0][1]

        P[3, 4] = predict_3_Vs_4_prob[0][0]
        P[4, 3] = predict_3_Vs_4_prob[0][1]

        # Aggregating votes for each class
        
        
        del predict_0_Vs_1_prob
        del predict_0_Vs_2_prob
        del predict_0_Vs_3_prob
        del predict_0_Vs_4_prob
        del predict_1_Vs_2_prob
        del predict_1_Vs_3_prob
        del predict_1_Vs_4_prob
        del predict_2_Vs_3_prob
        del predict_2_Vs_4_prob
        del predict_3_Vs_4_prob
        
        
        
        votes = []
        l1 = P[0, 1] + P[0, 2] + P[0, 3] + P[0, 4]
        l2 = P[1, 0] + P[1, 2] + P[1, 3] + P[1, 4]
        l3 = P[2, 0] + P[2, 1] + P[2, 3] + P[2, 4]
        l4 = P[3, 0] + P[3, 1] + P[3, 2] + P[3, 4]
        l5 = P[4, 0] + P[4, 1] + P[4, 2] + P[4, 3]
        
        del P
        

        votes.append([l1,l2,l3,l4,l5])
                            
        probabilities = self.AI_Automated_CAD_Screen.softmax(votes[0])
        prediction = np.argmax(probabilities)
    
        return prediction, probabilities


    def get_Prediction_cvml_Two_clases_models_70_30(self):
        # 2 Classes Classification model with average method
        Features_input_Other_Features = self.scaler_normal_vs_osteo_other_features_70_30.transform(self.Features_input_Other_Features)
        Random_Forest_chen_and_expert_Other_2_Class_pred = self.MLP_Normal_vs_osteo_other_features_70_30.predict_proba(Features_input_Other_Features)
    
        Features_input_HOG =self.scaler_normal_vs_osteo_HOG_70_30.transform(self.Features_input_HOG)
        MLP_Normal_vs_osteo_HOG_2_Class_pred = self.MLP_Normal_vs_osteo_HOG_70_30.predict_proba(Features_input_HOG)
    
        output_prob = (Random_Forest_chen_and_expert_Other_2_Class_pred +
                        MLP_Normal_vs_osteo_HOG_2_Class_pred )/ 2
    
        Two_Class_Avg_pred = np.argmax(output_prob)
        
        return Two_Class_Avg_pred, output_prob

    def get_Prediction_cvml_Three_clases_models_70_30(self):
        # 3 Classes Classification model with average method
        
        Features_input_Other_Features = self.scaler_3_Classes_other_features_70_30.transform(self.Features_input_Other_Features)
        MLP_chen_and_expert_Other_3_Class_pred = self.MLP_3_Classes_other_features_70_30.predict_proba(Features_input_Other_Features)
    
        Features_input_HOG =self.scaler_3_Classes_HOG_70_30.transform(self.Features_input_HOG)
        MLP_3_Classes_HOG_pred = self.MLP_3_Classes_HOG_70_30.predict_proba(Features_input_HOG)
    
        output_prob = (MLP_chen_and_expert_Other_3_Class_pred +
                        MLP_3_Classes_HOG_pred )/ 2
    
        Three_Class_Avg_pred = np.argmax(output_prob)

        return Three_Class_Avg_pred, output_prob
    
    
    
    
    def get_Prediction_cvml_Five_clases_models_70_30(self, Features_input_Other_Features, Features_input_HOG):
        # 5 Classes Classification model with average method
        
        sample = self.scaler_Chenn_and_Expert_Other_Features_5_Class_70_30.transform(Features_input_Other_Features)
        Random_Forest_chen_and_expert_Other_5_Class_pred = self.Random_Forest_Chenn_and_Expert_Other_Features_5_Class_70_30.predict_proba(sample)
    
        sample =self.scaler_Chenn_and_Expert_HOG_5_Class_70_30.transform(Features_input_HOG)
        MLP_chen_and_expert_HOG_5_Class_pred = self.MLP_Chenn_and_Expert_HOG_5_Class_70_30.predict_proba(sample)
        
                
        output_prob = (Random_Forest_chen_and_expert_Other_5_Class_pred +
                        MLP_chen_and_expert_HOG_5_Class_pred )/ 2
    
        Five_Class_Avg_pred = np.argmax(output_prob)
        
        return Five_Class_Avg_pred, output_prob
    
    
    
    def get_Prediction_cvml_Five_clases_Binary_Tree_models_70_30(self, Features_input_Other_Features, Features_input_HOG):
        # 5 Classes Classification Binary Tree method with binary moddels using average method
        # predict Other Features
        
        sample = self.scaler_0_vs_1_other_features_70_30.transform(Features_input_Other_Features)
        predict_0_Vs_1_other_features = self.RF_0_vs_1_other_features_70_30.predict_proba(sample)

        sample = self.scaler_0_vs_2_other_features_70_30.transform(Features_input_Other_Features)
        predict_0_Vs_2_other_features = self.MLP_0_vs_2_other_features_70_30.predict_proba(sample)

        sample = self.scaler_0_vs_3_other_features_70_30.transform(Features_input_Other_Features)
        predict_0_Vs_3_other_features = self.MLP_0_vs_3_other_features_70_30.predict_proba(sample)

        sample = self.scaler_0_vs_4_other_features_70_30.transform(Features_input_Other_Features)
        predict_0_Vs_4_other_features = self.MLP_0_vs_4_other_features_70_30.predict_proba(sample)

        sample = self.scaler_1_vs_2_other_features_70_30.transform(Features_input_Other_Features)
        predict_1_Vs_2_other_features = self.RF_1_vs_2_other_features_70_30.predict_proba(sample)

        sample = self.scaler_1_vs_3_other_features_70_30.transform(Features_input_Other_Features)
        predict_1_Vs_3_other_features = self.MLP_1_vs_3_other_features_70_30.predict_proba(sample)

        sample = self.scaler_1_vs_4_other_features_70_30.transform(Features_input_Other_Features)
        predict_1_Vs_4_other_features = self.MLP_1_vs_4_other_features_70_30.predict_proba(sample)

        sample = self.scaler_2_vs_3_other_features_70_30.transform(Features_input_Other_Features)
        predict_2_Vs_3_other_features = self.RF_2_vs_3_other_features_70_30.predict_proba(sample)

        sample = self.scaler_2_vs_4_other_features_70_30.transform(Features_input_Other_Features)
        predict_2_Vs_4_other_features = self.MLP_2_vs_4_other_features_70_30.predict_proba(sample)

        sample = self.scaler_3_vs_4_other_features_70_30.transform(Features_input_Other_Features)
        predict_3_Vs_4_other_features = self.MLP_3_vs_4_other_features_70_30.predict_proba(sample)


        # predict HOG Features 
        sample = self.scaler_0_vs_1_HOG_70_30.transform(Features_input_HOG)
        predict_0_Vs_1_HOG = self.MLP_0_vs_1_HOG_70_30.predict_proba(sample)

        sample = self.scaler_0_vs_2_HOG_70_30.transform(Features_input_HOG)
        predict_0_Vs_2_HOG = self.MLP_0_vs_2_HOG_70_30.predict_proba(sample)

        sample = self.scaler_0_vs_3_HOG_70_30.transform(Features_input_HOG)
        predict_0_Vs_3_HOG = self.MLP_0_vs_3_HOG_70_30.predict_proba(sample)

        sample = self.scaler_0_vs_4_HOG_70_30.transform(Features_input_HOG)
        predict_0_Vs_4_HOG = self.LR_0_vs_4_HOG_70_30.predict_proba(sample)

        sample = self.scaler_1_vs_2_HOG_70_30.transform(Features_input_HOG)
        predict_1_Vs_2_HOG = self.SVM_1_vs_2_HOG_70_30.predict_proba(sample)

        sample = self.scaler_1_vs_3_HOG_70_30.transform(Features_input_HOG)
        predict_1_Vs_3_HOG = self.SVM_1_vs_3_HOG_70_30.predict_proba(sample)

        sample = self.scaler_1_vs_4_HOG_70_30.transform(Features_input_HOG)
        predict_1_Vs_4_HOG = self.MLP_1_vs_4_HOG_70_30.predict_proba(sample)

        sample = self.scaler_2_vs_3_HOG_70_30.transform(Features_input_HOG)
        predict_2_Vs_3_HOG = self.MLP_2_vs_3_HOG_70_30.predict_proba(sample)

        sample = self.scaler_2_vs_4_HOG_70_30.transform(Features_input_HOG)
        predict_2_Vs_4_HOG = self.MLP_2_vs_4_HOG_70_30.predict_proba(sample)

        sample = self.scaler_3_vs_4_HOG_70_30.transform(Features_input_HOG)
        predict_3_Vs_4_HOG = self.SVM_3_vs_4_HOG_70_30.predict_proba(sample)


        # predict with avg method 
        predict_0_Vs_1_prob =  (predict_0_Vs_1_HOG + predict_0_Vs_1_other_features) / 2 
        predict_0_Vs_2_prob =  (predict_0_Vs_2_HOG + predict_0_Vs_2_other_features) / 2
        predict_0_Vs_3_prob =  (predict_0_Vs_3_HOG + predict_0_Vs_3_other_features) / 2
        predict_0_Vs_4_prob =  (predict_0_Vs_4_HOG + predict_0_Vs_4_other_features) / 2
        predict_1_Vs_2_prob =  (predict_1_Vs_2_HOG + predict_1_Vs_2_other_features) / 2 
        predict_1_Vs_3_prob =  (predict_1_Vs_3_HOG + predict_1_Vs_3_other_features) / 2 
        predict_1_Vs_4_prob =  (predict_1_Vs_4_HOG + predict_1_Vs_4_other_features) / 2 
        predict_2_Vs_3_prob =  (predict_2_Vs_3_HOG + predict_2_Vs_3_other_features) / 2
        predict_2_Vs_4_prob =  (predict_2_Vs_4_HOG + predict_2_Vs_4_other_features) / 2 
        predict_3_Vs_4_prob =  (predict_3_Vs_4_HOG + predict_3_Vs_4_other_features) / 2

        del predict_0_Vs_1_HOG
        del predict_0_Vs_1_other_features
        del predict_0_Vs_2_HOG
        del predict_0_Vs_2_other_features
        del predict_0_Vs_3_HOG
        del predict_0_Vs_3_other_features
        del predict_0_Vs_4_HOG
        del predict_0_Vs_4_other_features
        del predict_1_Vs_2_other_features
        del predict_1_Vs_2_HOG
        del predict_1_Vs_3_HOG
        del predict_1_Vs_3_other_features
        del predict_1_Vs_4_HOG
        del predict_1_Vs_4_other_features
        del predict_2_Vs_3_HOG
        del predict_2_Vs_3_other_features
        del predict_3_Vs_4_HOG
        del predict_2_Vs_4_other_features
        del predict_3_Vs_4_other_features
        del predict_2_Vs_4_HOG
        
        
        
        num_classes = 5

        P = np.zeros((num_classes, num_classes))
        # Populate the matrix with average probabilities
        P[0, 1] = predict_0_Vs_1_prob[0][0]
        P[1, 0] = predict_0_Vs_1_prob[0][1]

        P[0, 2] = predict_0_Vs_2_prob[0][0]
        P[2, 0] = predict_0_Vs_2_prob[0][1]

        P[0, 3] = predict_0_Vs_3_prob[0][0]
        P[3, 0] = predict_0_Vs_3_prob[0][1]

        P[0, 4] = predict_0_Vs_4_prob[0][0]
        P[4, 0] = predict_0_Vs_4_prob[0][1]

        P[1, 2] = predict_1_Vs_2_prob[0][0]
        P[2, 1] = predict_1_Vs_2_prob[0][1]

        P[1, 3] = predict_1_Vs_3_prob[0][0]
        P[3, 1] = predict_1_Vs_3_prob[0][1]

        P[1, 4] = predict_1_Vs_4_prob[0][0]
        P[4, 1] = predict_1_Vs_4_prob[0][1]

        P[2, 3] = predict_2_Vs_3_prob[0][0]
        P[3, 2] = predict_2_Vs_3_prob[0][1]

        P[2, 4] = predict_2_Vs_4_prob[0][0]
        P[4, 2] = predict_2_Vs_4_prob[0][1]

        P[3, 4] = predict_3_Vs_4_prob[0][0]
        P[4, 3] = predict_3_Vs_4_prob[0][1]

        # Aggregating votes for each class
        
        
        
        del predict_0_Vs_1_prob
        del predict_0_Vs_2_prob
        del predict_0_Vs_3_prob
        del predict_0_Vs_4_prob
        del predict_1_Vs_2_prob
        del predict_1_Vs_3_prob
        del predict_1_Vs_4_prob
        del predict_2_Vs_3_prob
        del predict_2_Vs_4_prob
        del predict_3_Vs_4_prob
        
        
        
        votes = []
        l1 = P[0, 1] + P[0, 2] + P[0, 3] + P[0, 4]
        l2 = P[1, 0] + P[1, 2] + P[1, 3] + P[1, 4]
        l3 = P[2, 0] + P[2, 1] + P[2, 3] + P[2, 4]
        l4 = P[3, 0] + P[3, 1] + P[3, 2] + P[3, 4]
        l5 = P[4, 0] + P[4, 1] + P[4, 2] + P[4, 3]
        
        del P
        

        votes.append([l1,l2,l3,l4,l5])
                            
        probabilities = self.AI_Automated_CAD_Screen.softmax(votes[0])
        prediction = np.argmax(probabilities)
    
        return prediction, probabilities
    
    
    
    
    
    
    
    
    
    
    def get_Prediction_cvml_Two_clases_models_60_40(self):
        # 2 Classes Classification model with average method
        Features_input_Other_Features = self.scaler_normal_vs_osteo_other_features_60_40.transform(self.Features_input_Other_Features)
        Random_Forest_chen_and_expert_Other_2_Class_pred = self.MLP_Normal_vs_osteo_other_features_60_40.predict_proba(Features_input_Other_Features)
    
        Features_input_HOG =self.scaler_normal_vs_osteo_HOG_60_40.transform(self.Features_input_HOG)
        MLP_Normal_vs_osteo_HOG_2_Class_pred = self.MLP_Normal_vs_osteo_HOG_60_40.predict_proba(Features_input_HOG)
    
        output_prob = (Random_Forest_chen_and_expert_Other_2_Class_pred +
                        MLP_Normal_vs_osteo_HOG_2_Class_pred )/ 2
    
        Two_Class_Avg_pred = np.argmax(output_prob)
        
        return Two_Class_Avg_pred, output_prob

    def get_Prediction_cvml_Three_clases_models_60_40(self):
        # 3 Classes Classification model with average method
        
        Features_input_Other_Features = self.scaler_3_Classes_other_features_60_40.transform(self.Features_input_Other_Features)
        MLP_chen_and_expert_Other_3_Class_pred = self.MLP_3_Classes_other_features_60_40.predict_proba(Features_input_Other_Features)
    
        Features_input_HOG =self.scaler_3_Classes_HOG_60_40.transform(self.Features_input_HOG)
        MLP_3_Classes_HOG_pred = self.MLP_3_Classes_HOG_60_40.predict_proba(Features_input_HOG)
    
        output_prob = (MLP_chen_and_expert_Other_3_Class_pred +
                        MLP_3_Classes_HOG_pred )/ 2
    
        Three_Class_Avg_pred = np.argmax(output_prob)

        return Three_Class_Avg_pred, output_prob
    
    def get_Prediction_cvml_Five_clases_models_60_40(self, Features_input_Other_Features, Features_input_HOG):
        # 5 Classes Classification model with average method
        
        sample = self.scaler_Chenn_and_Expert_Other_Features_5_Class_60_40.transform(Features_input_Other_Features)
        Random_Forest_chen_and_expert_Other_5_Class_pred = self.Random_Forest_Chenn_and_Expert_Other_Features_5_Class_60_40.predict_proba(sample)
    
        sample =self.scaler_Chenn_and_Expert_HOG_5_Class_60_40.transform(Features_input_HOG)
        MLP_chen_and_expert_HOG_5_Class_pred = self.MLP_Chenn_and_Expert_HOG_5_Class_60_40.predict_proba(sample)
    
        output_prob = (Random_Forest_chen_and_expert_Other_5_Class_pred +
                        MLP_chen_and_expert_HOG_5_Class_pred )/ 2
    
        Five_Class_Avg_pred = np.argmax(output_prob)
        
        return Five_Class_Avg_pred, output_prob
    
    
    def get_Prediction_cvml_Five_clases_Binary_Tree_models_60_40(self, Features_input_Other_Features, Features_input_HOG):
        # 5 Classes Classification Binary Tree method with binary moddels using average method
        # predict Other Features
        
        sample = self.scaler_0_vs_1_other_features_60_40.transform(Features_input_Other_Features)
        predict_0_Vs_1_other_features = self.RF_0_vs_1_other_features_60_40.predict_proba(sample)

        sample = self.scaler_0_vs_2_other_features_60_40.transform(Features_input_Other_Features)
        predict_0_Vs_2_other_features = self.MLP_0_vs_2_other_features_60_40.predict_proba(sample)

        sample = self.scaler_0_vs_3_other_features_60_40.transform(Features_input_Other_Features)
        predict_0_Vs_3_other_features = self.MLP_0_vs_3_other_features_60_40.predict_proba(sample)

        sample = self.scaler_0_vs_4_other_features_60_40.transform(Features_input_Other_Features)
        predict_0_Vs_4_other_features = self.MLP_0_vs_4_other_features_60_40.predict_proba(sample)

        sample = self.scaler_1_vs_2_other_features_60_40.transform(Features_input_Other_Features)
        predict_1_Vs_2_other_features = self.RF_1_vs_2_other_features_60_40.predict_proba(sample)

        sample = self.scaler_1_vs_3_other_features_60_40.transform(Features_input_Other_Features)
        predict_1_Vs_3_other_features = self.MLP_1_vs_3_other_features_60_40.predict_proba(sample)

        sample = self.scaler_1_vs_4_other_features_60_40.transform(Features_input_Other_Features)
        predict_1_Vs_4_other_features = self.MLP_1_vs_4_other_features_60_40.predict_proba(sample)

        sample = self.scaler_2_vs_3_other_features_60_40.transform(Features_input_Other_Features)
        predict_2_Vs_3_other_features = self.RF_2_vs_3_other_features_60_40.predict_proba(sample)

        sample = self.scaler_2_vs_4_other_features_60_40.transform(Features_input_Other_Features)
        predict_2_Vs_4_other_features = self.MLP_2_vs_4_other_features_60_40.predict_proba(sample)

        sample = self.scaler_3_vs_4_other_features_60_40.transform(Features_input_Other_Features)
        predict_3_Vs_4_other_features = self.MLP_3_vs_4_other_features_60_40.predict_proba(sample)


        # predict HOG Features 
        sample = self.scaler_0_vs_1_HOG_60_40.transform(Features_input_HOG)
        predict_0_Vs_1_HOG = self.MLP_0_vs_1_HOG_60_40.predict_proba(sample)

        sample = self.scaler_0_vs_2_HOG_60_40.transform(Features_input_HOG)
        predict_0_Vs_2_HOG = self.MLP_0_vs_2_HOG_60_40.predict_proba(sample)

        sample = self.scaler_0_vs_3_HOG_60_40.transform(Features_input_HOG)
        predict_0_Vs_3_HOG = self.MLP_0_vs_3_HOG_60_40.predict_proba(sample)

        sample = self.scaler_0_vs_4_HOG_60_40.transform(Features_input_HOG)
        predict_0_Vs_4_HOG = self.LR_0_vs_4_HOG_60_40.predict_proba(sample)

        sample = self.scaler_1_vs_2_HOG_60_40.transform(Features_input_HOG)
        predict_1_Vs_2_HOG = self.SVM_1_vs_2_HOG_60_40.predict_proba(sample)

        sample = self.scaler_1_vs_3_HOG_60_40.transform(Features_input_HOG)
        predict_1_Vs_3_HOG = self.SVM_1_vs_3_HOG_60_40.predict_proba(sample)

        sample = self.scaler_1_vs_4_HOG_60_40.transform(Features_input_HOG)
        predict_1_Vs_4_HOG = self.MLP_1_vs_4_HOG_60_40.predict_proba(sample)

        sample = self.scaler_2_vs_3_HOG_60_40.transform(Features_input_HOG)
        predict_2_Vs_3_HOG = self.MLP_2_vs_3_HOG_60_40.predict_proba(sample)

        sample = self.scaler_2_vs_4_HOG_60_40.transform(Features_input_HOG)
        predict_2_Vs_4_HOG = self.MLP_2_vs_4_HOG_60_40.predict_proba(sample)

        sample = self.scaler_3_vs_4_HOG_60_40.transform(Features_input_HOG)
        predict_3_Vs_4_HOG = self.SVM_3_vs_4_HOG_60_40.predict_proba(sample)


        # predict with avg method 
        predict_0_Vs_1_prob =  (predict_0_Vs_1_HOG + predict_0_Vs_1_other_features) / 2 
        predict_0_Vs_2_prob =  (predict_0_Vs_2_HOG + predict_0_Vs_2_other_features) / 2
        predict_0_Vs_3_prob =  (predict_0_Vs_3_HOG + predict_0_Vs_3_other_features) / 2
        predict_0_Vs_4_prob =  (predict_0_Vs_4_HOG + predict_0_Vs_4_other_features) / 2
        predict_1_Vs_2_prob =  (predict_1_Vs_2_HOG + predict_1_Vs_2_other_features) / 2 
        predict_1_Vs_3_prob =  (predict_1_Vs_3_HOG + predict_1_Vs_3_other_features) / 2 
        predict_1_Vs_4_prob =  (predict_1_Vs_4_HOG + predict_1_Vs_4_other_features) / 2 
        predict_2_Vs_3_prob =  (predict_2_Vs_3_HOG + predict_2_Vs_3_other_features) / 2
        predict_2_Vs_4_prob =  (predict_2_Vs_4_HOG + predict_2_Vs_4_other_features) / 2 
        predict_3_Vs_4_prob =  (predict_3_Vs_4_HOG + predict_3_Vs_4_other_features) / 2

        del predict_0_Vs_1_HOG
        del predict_0_Vs_1_other_features
        del predict_0_Vs_2_HOG
        del predict_0_Vs_2_other_features
        del predict_0_Vs_3_HOG
        del predict_0_Vs_3_other_features
        del predict_0_Vs_4_HOG
        del predict_0_Vs_4_other_features
        del predict_1_Vs_2_other_features
        del predict_1_Vs_2_HOG
        del predict_1_Vs_3_HOG
        del predict_1_Vs_3_other_features
        del predict_1_Vs_4_HOG
        del predict_1_Vs_4_other_features
        del predict_2_Vs_3_HOG
        del predict_2_Vs_3_other_features
        del predict_3_Vs_4_HOG
        del predict_2_Vs_4_other_features
        del predict_3_Vs_4_other_features
        del predict_2_Vs_4_HOG
        
        
        
        num_classes = 5

        P = np.zeros((num_classes, num_classes))
        # Populate the matrix with average probabilities
        P[0, 1] = predict_0_Vs_1_prob[0][0]
        P[1, 0] = predict_0_Vs_1_prob[0][1]

        P[0, 2] = predict_0_Vs_2_prob[0][0]
        P[2, 0] = predict_0_Vs_2_prob[0][1]

        P[0, 3] = predict_0_Vs_3_prob[0][0]
        P[3, 0] = predict_0_Vs_3_prob[0][1]

        P[0, 4] = predict_0_Vs_4_prob[0][0]
        P[4, 0] = predict_0_Vs_4_prob[0][1]

        P[1, 2] = predict_1_Vs_2_prob[0][0]
        P[2, 1] = predict_1_Vs_2_prob[0][1]

        P[1, 3] = predict_1_Vs_3_prob[0][0]
        P[3, 1] = predict_1_Vs_3_prob[0][1]

        P[1, 4] = predict_1_Vs_4_prob[0][0]
        P[4, 1] = predict_1_Vs_4_prob[0][1]

        P[2, 3] = predict_2_Vs_3_prob[0][0]
        P[3, 2] = predict_2_Vs_3_prob[0][1]

        P[2, 4] = predict_2_Vs_4_prob[0][0]
        P[4, 2] = predict_2_Vs_4_prob[0][1]

        P[3, 4] = predict_3_Vs_4_prob[0][0]
        P[4, 3] = predict_3_Vs_4_prob[0][1]

        # Aggregating votes for each class
        
        
        del predict_0_Vs_1_prob
        del predict_0_Vs_2_prob
        del predict_0_Vs_3_prob
        del predict_0_Vs_4_prob
        del predict_1_Vs_2_prob
        del predict_1_Vs_3_prob
        del predict_1_Vs_4_prob
        del predict_2_Vs_3_prob
        del predict_2_Vs_4_prob
        del predict_3_Vs_4_prob
        
        
        votes = []
        l1 = P[0, 1] + P[0, 2] + P[0, 3] + P[0, 4]
        l2 = P[1, 0] + P[1, 2] + P[1, 3] + P[1, 4]
        l3 = P[2, 0] + P[2, 1] + P[2, 3] + P[2, 4]
        l4 = P[3, 0] + P[3, 1] + P[3, 2] + P[3, 4]
        l5 = P[4, 0] + P[4, 1] + P[4, 2] + P[4, 3]
        
        del P

        votes.append([l1,l2,l3,l4,l5])
                            
        probabilities = self.AI_Automated_CAD_Screen.softmax(votes[0])
        prediction = np.argmax(probabilities)
    
        return prediction, probabilities
    
#  _______________________________________________ HOG Histogram _____________________________________________________
class HOGHistogram(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HOG Histogram")
        self.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.setGeometry(int(0.546875 * self.screen_width), int(0.20833 * self.screen_width), int(0.4167 * self.screen_width), int(0.20833 * self.screen_width))
        self.HOGHistogram_label = QLabel("HOG Histogram")
        self.HOGHistogram_label.setStyleSheet("color: white;font:25px")
        self.HOGHistogram_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.HOGHistogram_label)
        self.setLayout(layout)

    def set_HOG_Histogram(self, histogram):
        if histogram is not None:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(histogram)), histogram, align="center")
            plt.title(" Histogram of Oriented Gradients")
            plt.xlabel("HOG Bins")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("Temp/Saved Figures/HOG_Histogram.png")
            HOG_Histogram_img = QImage("Temp/Saved Figures/HOG_Histogram.png")
            self.HOGHistogram_label.setPixmap(QPixmap.fromImage(HOG_Histogram_img))
            self.HOGHistogram_label.setScaledContents(True)
            plt.close()

    def closeEvent(self, event):
        options = ["Save", "Discard", "Cancel"]
        msg = QMessageBox()
        msg.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Close HOG Histogram Window")
        msg.setWindowIcon(QIcon("imgs/LogoSplashScreen_Original.png"))
        msg.setText("Do you want to save changes before closing?")
        for option in options:
            msg.addButton(option, QMessageBox.AcceptRole)
        result = msg.exec_()
        if result == 0:
            pixmap = self.HOGHistogram_label.grab()
            pixmap.save("Temp/Saved Figures/HOG_Histogram.png")
            event.accept()
        elif result == 1:
            event.accept()
        else:
            event.ignore()
#  _______________________________________________ Settings _____________________________________________________
class SettingsWindow(QWidget):
    def __init__(self, shared_data):
        super().__init__()
        self.setWindowTitle("Settings - Select the Features sets!")
        self.setWindowIcon(QIcon("imgs/setting_blacks.svg"))
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.setGeometry(int(0.078125 * self.screen_width), int(0.104167 * self.screen_width), int(0.28645833 * self.screen_width), int(0.3645833 * self.screen_width))
        self.setFixedSize(int(0.28645833 * self.screen_width), int(0.3645833 * self.screen_width))
        self.pixmap = QPixmap("imgs/4settings_Background.png")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Window | Qt.WindowTitleHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
#  _______________________________________________ initializations _____________________________________________________
        
        self.shared_data = shared_data
#  ____________________________________________________ Layout ____________________________________________________
        
        vertical_layout_Features_selection = QVBoxLayout()
        vertical_layout_Features_selection.setContentsMargins(0,0,0,0)
        
        self.histogram_layout = QHBoxLayout(self)
        self.histogram_Widget = QWidget(self)
        self.histogram_Widget.setLayout(self.histogram_layout)
        
        self.histogram_label_Name = QLabel("""• Histogram Properties:         • Mean
                                            • Standard Deviation
                                            • Skewness
                                            • Kurtosis""")
        self.histogram_label_Name.setStyleSheet("color: rgba(10, 10,10, 255)")
        
        font2 = QFont()
        font2.setPointSize(int(0.0078125 * self.screen_width))
        font2.setBold(False)
        font2.setFamily("Helvetica")
        self.histogram_label_Name.setFont(font2)
        
        
        self.histogram_switch = QPushButton()
        self.histogram_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.histogram_switch.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.histogram_switch.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.histogram_switch.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
        self.histogram_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
        self.histogram_switch.clicked.connect(self.histogram_switch_func)
        self.histogram_layout.addWidget(self.histogram_label_Name)
        self.histogram_layout.addWidget(self.histogram_switch)
        vertical_layout_Features_selection.addWidget(self.histogram_Widget)
        
        self.cooccurrence_layout = QHBoxLayout(self)
        self.cooccurrence_Widget = QWidget(self)
        self.cooccurrence_Widget.setLayout(self.cooccurrence_layout)
        
        self.cooccurrence_label_Name = QLabel("""• Cooccurrence Properties:
                                            • Contrast
                                            • Energy
                                            • Correlation
                                            • Homogeneity
                                            • Dissimilarity
                                            • ASM
                                            • Max Probability""")
        self.cooccurrence_label_Name.setStyleSheet("color: rgba(10, 10,10, 255)")
        
        font2 = QFont()
        font2.setPointSize(int(0.0078125 * self.screen_width))
        font2.setBold(False)
        font2.setFamily("Helvetica")
        self.cooccurrence_label_Name.setFont(font2)
        
        self.cooccurrence_switch = QPushButton()
        self.cooccurrence_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.cooccurrence_switch.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.cooccurrence_switch.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.cooccurrence_switch.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
        self.cooccurrence_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
        self.cooccurrence_switch.clicked.connect(self.cooccurrence_switch_func)
        self.cooccurrence_layout.addWidget(self.cooccurrence_label_Name)
        self.cooccurrence_layout.addWidget(self.cooccurrence_switch)
        vertical_layout_Features_selection.addWidget(self.cooccurrence_Widget)
        
        self.lbp_layout = QHBoxLayout(self)
        self.lbp_Widget = QWidget(self)
        self.lbp_Widget.setLayout(self.lbp_layout)
        
        self.lbp_label_Name = QLabel("""• Local Binary Pattern Properties:
                                            • LBP Histogram Bins
                                            • LBP Variance
                                            • LBP Entropy""")
        self.lbp_label_Name.setStyleSheet("color: rgba(10, 10,10, 255)")
        
        font2 = QFont()
        font2.setPointSize(int(0.0078125 * self.screen_width))
        font2.setBold(False)
        font2.setFamily("Helvetica")
        self.lbp_label_Name.setFont(font2)
        
        self.lbp_switch = QPushButton()
        self.lbp_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.lbp_switch.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.lbp_switch.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.lbp_switch.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
        self.lbp_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
        
        self.lbp_switch.clicked.connect(self.lbp_switch_func)
        self.lbp_layout.addWidget(self.lbp_label_Name)
        self.lbp_layout.addWidget(self.lbp_switch)
        vertical_layout_Features_selection.addWidget(self.lbp_Widget)
        
        self.ltp_layout = QHBoxLayout(self)
        self.ltp_Widget = QWidget(self)
        self.ltp_Widget.setLayout(self.ltp_layout)
        
        self.ltp_label_Name = QLabel("""• Local Ternary Pattern Properties:
                                            • LTP Histogram Bins
                                            • LTP Variance
                                            • LTP Entropy""")
        self.ltp_label_Name.setStyleSheet("color: rgba(10, 10,10, 255)")
        
        font2 = QFont()
        font2.setPointSize(int(0.0078125 * self.screen_width))
        font2.setBold(False)
        font2.setFamily("Helvetica")
        self.ltp_label_Name.setFont(font2)
        
        self.ltp_switch = QPushButton()
        self.ltp_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.ltp_switch.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.ltp_switch.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.ltp_switch.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
        self.ltp_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
        self.ltp_switch.clicked.connect(self.ltp_switch_func)
        self.ltp_layout.addWidget(self.ltp_label_Name)
        self.ltp_layout.addWidget(self.ltp_switch)
        vertical_layout_Features_selection.addWidget(self.ltp_Widget)
        
        self.hog_layout = QHBoxLayout(self)
        self.hog_Widget = QWidget(self)
        self.hog_Widget.setLayout(self.hog_layout)
        
        self.hog_label_Name = QLabel("• Histogram of Oriented Gradient Bins.")
        self.hog_label_Name.setStyleSheet("color: rgba(10, 10,10, 255)")
        
        font2 = QFont()
        font2.setPointSize(int(0.0078125 * self.screen_width))
        font2.setBold(False)
        font2.setFamily("Helvetica")
        self.hog_label_Name.setFont(font2)
        
        self.hog_switch = QPushButton()
        self.hog_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.hog_switch.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.hog_switch.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.hog_switch.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
        self.hog_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
        self.hog_switch.clicked.connect(self.hog_switch_func)
        self.hog_layout.addWidget(self.hog_label_Name)
        self.hog_layout.addWidget(self.hog_switch)
        vertical_layout_Features_selection.addWidget(self.hog_Widget)
        
        buttons_layout = QHBoxLayout(self)
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        vertical_layout_Features_selection.addWidget(buttons_widget)
        
        font3 = QFont()
        font3.setPointSize(int(0.00625 * self.screen_width))
        font3.setBold(False)
        font3.setFamily("Helvetica")
        
        self.OK_Button = QPushButton("Apply Changes")
        self.OK_Button.setMinimumHeight(int(0.0140625 * self.screen_width))
        self.OK_Button.setFont(font3)
        self.OK_Button.clicked.connect(self.Apply_changes_Button_func)
        buttons_layout.addWidget(self.OK_Button)
        
        self.Restore_to_defaults_Button = QPushButton("Restore to defaults")
        self.Restore_to_defaults_Button.setMinimumHeight(int(0.0140625 * self.screen_width))
        self.Restore_to_defaults_Button.setFont(font3)
        self.Restore_to_defaults_Button.clicked.connect(self.Restore_to_defaults_Button_func)
        buttons_layout.addWidget(self.Restore_to_defaults_Button)
        
        centralLayout = QVBoxLayout()
        centralLayout.setContentsMargins(int(0.005208333 * self.screen_width),int(0.005208333 * self.screen_width),int(0.005208333 * self.screen_width),int(0.005208333 * self.screen_width))
        vertical_layout_Features_selection_widget = QWidget(self)
        vertical_layout_Features_selection_widget.setLayout(vertical_layout_Features_selection)
        centralLayout.addWidget(vertical_layout_Features_selection_widget)
        self.setLayout(centralLayout)
    
        self.setMouseTracking(True)  # Enable mouse tracking
        self.draggable = False
        self.offset = QPoint()
    
    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() == Qt.WindowMinimized:
                self.showNormal()
            elif self.windowState() == Qt.WindowMaximized:
                self.showNormal()
        super().changeEvent(event)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        painter.drawPixmap(0, 0, scaled_pixmap)
        
        painter.setRenderHint(QPainter.Antialiasing)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.draggable = True
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if self.draggable:
            self.move(self.mapToGlobal(event.pos() - self.offset))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.draggable = False
            
    def Restore_to_defaults_Button_func(self):
        self.shared_data["Histogram_properties_switch_parameter"] = True
        self.shared_data["cooccurrence_properties_switch_parameter"] = True
        self.shared_data["lbp_switch_parameter"] = True
        self.shared_data["ltp_switch_parameter"] = True
        self.shared_data["hog_switch_parameter"] = True
        
        self.histogram_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.histogram_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.histogram_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.histogram_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
        self.histogram_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        self.cooccurrence_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.cooccurrence_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.cooccurrence_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.cooccurrence_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
        self.cooccurrence_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        self.lbp_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.lbp_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.lbp_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.lbp_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
        self.lbp_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        self.ltp_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.ltp_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.ltp_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.ltp_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
        self.ltp_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        self.hog_switch.setIcon(QIcon('imgs/enable-mode.png'))
        self.hog_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.hog_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
        self.hog_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
        self.hog_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
                
    def Apply_changes_Button_func(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)")

        if not file_path:
            return

        directory, file_name = os.path.split(file_path)
        csv_file_path = f'{directory}/{file_name}'
        self.createCSV(csv_file_path)
        self.hide()

    def createCSV(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.shared_data["csv"] = self.csv_file_path
        print("File dialog successful, CSV file created at:", csv_file_path)
            
    def histogram_switch_func(self):
        if  self.shared_data["Histogram_properties_switch_parameter"] == True:
            self.shared_data["Histogram_properties_switch_parameter"] = not self.shared_data["Histogram_properties_switch_parameter"]
            
            self.histogram_switch.setIcon(QIcon('imgs/disable-mode.svg'))
            self.histogram_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.histogram_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.histogram_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.histogram_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        else:
            self.shared_data["Histogram_properties_switch_parameter"] = not self.shared_data["Histogram_properties_switch_parameter"]
        
            self.histogram_switch.setIcon(QIcon('imgs/enable-mode.png'))
            self.histogram_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.histogram_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.histogram_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.histogram_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
                
    def cooccurrence_switch_func(self):
        if  self.shared_data["cooccurrence_properties_switch_parameter"] == True:
            self.shared_data["cooccurrence_properties_switch_parameter"] = not self.shared_data["cooccurrence_properties_switch_parameter"]
            
            self.cooccurrence_switch.setIcon(QIcon('imgs/disable-mode.svg'))
            self.cooccurrence_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.cooccurrence_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.cooccurrence_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.cooccurrence_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        else:
            self.shared_data["cooccurrence_properties_switch_parameter"] = not self.shared_data["cooccurrence_properties_switch_parameter"]
        
            self.cooccurrence_switch.setIcon(QIcon('imgs/enable-mode.png'))
            self.cooccurrence_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.cooccurrence_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.cooccurrence_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.cooccurrence_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
    def lbp_switch_func(self):
        if  self.shared_data["lbp_switch_parameter"] == True:
            self.shared_data["lbp_switch_parameter"] = not self.shared_data["lbp_switch_parameter"]
            self.lbp_switch.setIcon(QIcon('imgs/disable-mode.svg'))
            self.lbp_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.lbp_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.lbp_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.lbp_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        else:
            self.shared_data["lbp_switch_parameter"] = not self.shared_data["lbp_switch_parameter"]
            self.lbp_switch.setIcon(QIcon('imgs/enable-mode.png'))
            self.lbp_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.lbp_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.lbp_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.lbp_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
            
    def ltp_switch_func(self):
        if  self.shared_data["ltp_switch_parameter"] == True:
            self.shared_data["ltp_switch_parameter"] = not self.shared_data["ltp_switch_parameter"]
            self.ltp_switch.setIcon(QIcon('imgs/disable-mode.svg'))
            self.ltp_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.ltp_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.ltp_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.ltp_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        else:
            self.shared_data["ltp_switch_parameter"] = not self.shared_data["ltp_switch_parameter"]
            self.ltp_switch.setIcon(QIcon('imgs/enable-mode.png'))
            self.ltp_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.ltp_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.ltp_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.ltp_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
    def hog_switch_func(self):
        if  self.shared_data["hog_switch_parameter"] == True:
            self.shared_data["hog_switch_parameter"] = not self.shared_data["hog_switch_parameter"]
            self.hog_switch.setIcon(QIcon('imgs/disable-mode.svg'))
            self.hog_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.hog_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.hog_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.hog_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        else:
            self.shared_data["hog_switch_parameter"] = not self.shared_data["hog_switch_parameter"]
            self.hog_switch.setIcon(QIcon('imgs/enable-mode.png'))
            self.hog_switch.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.hog_switch.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.hog_switch.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.hog_switch.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")

#  _________________________________________________ Help ___________________________________________________
class HelpWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Help")
        self.setWindowIcon(QIcon("imgs/help-circle.svg"))
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.setFixedSize(int(self.screen_width), int(self.screen_height))
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        self.Engineering_insights_label_Flag = True
        self.x = -int(0.46875 * self.screen_width)
        self.y = -int(0.2380208333 * self.screen_width)
        self.F = 0.5530550291791955

        layout2 = QVBoxLayout()
        layout2.setContentsMargins(0,0,0,0)
        widget2 = QWidget()
        widget2.setLayout(layout2)
        
        help1 = """ 
                    Geometrical based Features: (Exported in Automatic CSV File)

                •	Medial Region: The inner side of the knee joint.
                •	Central Region: The middle section of the knee joint.
                •	Lateral Region: The outer side of the knee joint.
                •	Tibial Width (mm): The horizontal measurement of the tibial plateau (in millimeters), which is the top surface of the tibia.
                •	JSN Avg V. Distance (Pixel): The average vertical distance of the Joint Space Narrowing (JSN) across the medial, central, and lateral regions measured in pixels.
                •	JSN Avg V. Distance (mm): The average vertical distance of the Joint Space Narrowing (JSN) across the medial, central, and lateral regions measured in millimeters.
                •	Medial_distance (Pixel): The average vertical distance of the joint space in the medial region measured in pixels.
                •	Central_distance (Pixel) The average   vertical distance of the joint space in the central region measured in pixels.
                •	Lateral_distance (Pixel): The average   vertical distance of the joint space in the lateral region measured in pixels.
                •	Medial_distance (mm): The average vertical distance of the joint space in the medial region measured in millimeters.
                •	Central_distance (mm): The average   vertical distance of the joint space in the central region measured in millimeters.
                •	Lateral_distance (mm): The average   vertical distance of the joint space in the lateral region measured in millimeters.
                •	Tibial_Medial_ratio: The ratio of the medial joint space distance to the tibial width.
                •	Tibial_Central_ratio: The ratio of the central joint space distance to the tibial width.
                •	Tibial_Lateral_ratio: The ratio of the lateral joint space distance to the tibial width.
                •	JSN Area (Squared Pixel): The total area of joint space narrowing (JSN) across the medial, central, and lateral regions measured in squared pixels.
                •	JSN Area (Squared mm): The total area of joint space narrowing (JSN) across the medial, central, and lateral regions measured in squared millimeters. 
                •	Medial Area (Squared Pixel): The area of joint space in the medial region measured in squared pixels. 
                •	Central Area (Squared Pixel): The area of joint space in the central region measured in squared pixels. 
                •	Lateral Area (Squared Pixel): The area of joint space in the Lateral region measured in squared pixels.
                •	Medial Area (Squared mm): The area of joint space in the medial region measured in squared millimeters.
                •	Central Area (Squared mm): The area of joint space in the Central region measured in squared millimeters 
                •	Lateral Area (Squared mm): The area of joint space in the Lateral region measured in squared millimeters 
                •	Medial Area (JSN Ratio): Ratio of medial area affected by joint space narrowing.
                •	Central Area (JSN Ratio): Ratio of central area affected by joint space narrowing.
                •	Lateral Area (JSN Ratio): Ratio of lateral area affected by joint space narrowing.
                •	Medial Area Ratio TWPA (%): Percentage of medial area affected by tibial width per area.
                •	Central Area Ratio TWPA (%): Percentage of central area affected by tibial width per area.
                •	Lateral Area Ratio TWPA (%): Percentage of lateral area affected by tibial width per area.

                """
        self.inform_label1 = QLabel(help1)
        self.inform_label1.setStyleSheet("""background-color: white;
                                          color: rgba(0,0,0,255);
                                          border: none;
                                          text-align: left;
                                          """)
        self.inform_label1.setAlignment(Qt.AlignLeft)
        
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(int(0.0078125 * self.screen_width))
        font.setBold(False)
        self.inform_label1.setFont(font)
        self.inform_label1.setContentsMargins(int(0.0078125 * self.screen_width),int(0.0078125 * self.screen_width),int(0.0078125 * self.screen_width),int(0.0078125 * self.screen_width))
        
        scroll_area1 = QScrollArea(self)
        scroll_area1.setWidgetResizable(True)
        scroll_area1.setMinimumHeight(int(0.09114583333 * self.screen_width))
        scroll_area1.setWidget(self.inform_label1)
        
        help2 = """
                          Image based Features: (Exported in Automatic and Manual CSV Files)

                •	Mean: The average pixel intensity value within the image. It gives a sense of the overall brightness of the image.
                •	Sigma (Standard Deviation): A measure of the spread or dispersion of pixel intensity values from the mean. Higher sigma indicates greater variation in pixel intensities.
                •	Skewness: A measure of the asymmetry of the distribution of pixel intensities. 
                •	Kurtosis: A measure of the "tailedness" of the distribution of pixel intensities; it indicates the presence of outliers.
                •	Contrast: Measures the local variations in pixel intensities. Higher contrast indicates larger differences between adjacent pixel values.
                •	Energy: Represents the uniformity or smoothness of the image texture. High energy values indicate more texture variation.
                •	Correlation: A measure of how correlated a pixel is to its neighbors over the entire image.
                •	Homogeneity: Reflects the closeness of pixel intensity values in the image. High homogeneity indicates that neighboring pixel intensities are similar.
                •	Dissimilarity: Measures the average absolute difference in pixel intensity values between neighboring pixels.
                •	ASM (Angular Second Moment): Also known as Energy, it is a measure of texture uniformity and homogeneity, calculated as the sum of squares of pixel intensities in the Gray Level Co-occurrence Matrix (GLCM).
                •	Max Probability: The maximum probability of occurrence of a certain texture pattern in the image.
                •	LBP (Local Binary Pattern) features: Descriptors that capture the local texture patterns by comparing each pixel with its neighboring pixels and encoding the result as a binary number.
                •	LTP (Local Ternary Pattern) features:  Descriptors that but capture local ternary patterns.
                •	HOG (Histogram of Oriented Gradients) features: Descriptors that capture the distribution of gradient orientations in different parts of the image.

                """

        self.inform_label2 = QLabel(help2)
        self.inform_label2.setStyleSheet("""background-color: white;
                                          color: rgba(0,0,0,255);
                                          border: none;
                                          text-align: left;
                                          """)
        self.inform_label2.setAlignment(Qt.AlignLeft)
        font2 = QFont("Segoe UI")
        font2.setFamily("Arial")
        font2.setPointSize(int(0.0078125 * self.screen_width))
        font2.setBold(False)
        self.inform_label2.setFont(font2)
        self.inform_label2.setContentsMargins(int(0.0078125 * self.screen_width),int(0.0078125 * self.screen_width),int(0.0078125 * self.screen_width),int(0.0078125 * self.screen_width))

        scroll_area2 = QScrollArea(self)
        scroll_area2.setWidgetResizable(True)
        scroll_area2.setWidget(self.inform_label2)
        scroll_area2.setMinimumHeight(int(0.07291667 * self.screen_width))
        
            
        manual_layout = QHBoxLayout(self)
        manual_layout.setContentsMargins(int(0.015625 * self.screen_width), int(0.0078125 * self.screen_width), 0, 0)
        manual_widget = QWidget()
        manual_widget.setLayout(manual_layout)
        manual_widget.setMaximumHeight(int(0.046875 * self.screen_width))
        
        
        self.exit_help_button = QPushButton(self)
        self.exit_help_button.setIcon(QIcon("imgs/return.png"))
        self.exit_help_button.setIconSize(QSize(int(0.015625 * self.screen_width),int(0.015625 * self.screen_width)))
        self.exit_help_button.setFixedSize(int(0.015625 * self.screen_width), int(0.015625 * self.screen_width))
        self.exit_help_button.setToolTip("Return")
        self.exit_help_button.clicked.connect(self.exit_help_function)
        self.exit_help_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
        """)     
                
        
        manual_layout.addWidget(self.exit_help_button)
        heading_label = QLabel(
            """
    To know how to use the " Manual image-based Feature Extractor "
            """
        )

        font = QFont()
        font.setPointSize(int(0.007291666 * self.screen_width))
        font.setFamily("Helvetica")
        heading_label.setFont(font)
        heading_label.setAlignment(Qt.AlignCenter)
        manual_layout.addWidget(heading_label)

        video_path = "Help/Manual Feature Extractor.mp4"
        watch_video_button = QPushButton("Click here")
        watch_video_button.setCursor(Qt.PointingHandCursor)
        font4 = QFont()
        font4.setPointSize(int(0.007291666 * self.screen_width))
        watch_video_button.setFont(font4)
        watch_video_button.setStyleSheet("""
                                        border: none;
                                        background-color: transparent;
                                        color: rgb(63, 127, 166);
                                        text-align: left;
                                        """)

        watch_video_button.clicked.connect(lambda: self.openVideoWindow(video_path))
        manual_layout.addWidget(watch_video_button)
        
        spacer2 = QSpacerItem(int(0.02604167 * self.screen_width), 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        manual_layout.addItem(spacer2)
        layout2.addWidget(manual_widget)
        
        
        Engineering_insights_layout = QVBoxLayout(self)
        Engineering_insights_layout_widget = QWidget()
        Engineering_insights_layout_widget.setLayout(Engineering_insights_layout)
        
        Engineering_insights_label_note = QLabel("The Geometric insight for knee OsteoArthritis classification throught the Kellgren-Lawrence grading system")
        Engineering_insights_label_note.setAlignment(Qt.AlignLeft)        
        
        self.toggle_note = AnimatedButton("imgs/reverse_ref.png")
        self.toggle_note.setIconSize(QSize(int(0.0078125 * self.screen_width),int(0.0078125 * self.screen_width)))
        self.toggle_note.setFixedSize(int(0.0078125 * self.screen_width),int(0.0078125 * self.screen_width))
        
        self.toggle_note.clicked.connect(self.toggle_Note)
        
        hori = QHBoxLayout()
        hori.setContentsMargins(int(0.036458333 * self.screen_width), 0, int(0.0104167 * self.screen_width) ,0)
        hori_Widget = QWidget()
        hori_Widget.setLayout(hori)
        
        hori.addWidget(Engineering_insights_label_note)
        hori.addWidget(self.toggle_note, alignment=Qt.AlignLeft)
        
        self.Engineering_insights_label = ZoomingLabel(self.x, self.y, self.F)
        self.scroll_area_Engineering_insights = QScrollArea()
        self.scroll_area_Engineering_insights.setWidget(self.Engineering_insights_label)
        self.scroll_area_Engineering_insights.setWidgetResizable(True)
        
        self.scroll_area_Engineering_insights.setFixedSize(int(self.screen_width), int(0.416667 * self.screen_width))
        self.Engineering_insights_label.setScaledContents(True)
        self.pixmap = QPixmap("Help/engineering-insights.png")
        self.Engineering_insights_label.setPixmap(self.pixmap)
        self.Engineering_insights_label.setAlignment(Qt.AlignCenter)
        

        font = QFont()
        font.setPointSize(int(0.009375 * self.screen_width))
        font.setFamily("Helvetica")
        Engineering_insights_label_note.setFont(font)
        Engineering_insights_label_note.setAlignment(Qt.AlignCenter)
        Engineering_insights_layout.addWidget(hori_Widget)
        Engineering_insights_layout.addWidget(self.scroll_area_Engineering_insights)
        
        self.Default_Pos = AnimatedButton("imgs/initial_pos.png")
        self.Default_Pos.setToolTip("Reset to default Position")
        self.Default_Pos.clicked.connect(self.reset_to_initial_Pos)
        self.Default_Pos.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
        """)


        button_size = self.Default_Pos.sizeHint()
        self.overlay = QWidget(self)
        self.overlay.setGeometry(
            self.width() - button_size.width() - int(0.005208333 * self.screen_width),  # 10 px margin from the right
            self.height() - button_size.height() - int(0.005208333 * self.screen_width),  # 10 px margin from the bottom
            button_size.width(),
            button_size.height()
        )
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0);")  # Fully transparent
        self.overlay.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        self.Default_Pos.setParent(self.overlay)
        self.Default_Pos.setGeometry(0, 0, button_size.width(), button_size.height())
        self.overlay.setVisible(True)

        
        vertical_layout_Features_selection = QHBoxLayout()
        vertical_layout_Features_selection.setContentsMargins(0,0,0,0)
        layout2.addLayout(vertical_layout_Features_selection)
        vertical_layout_Features_selection.addWidget(scroll_area1)
        vertical_layout_Features_selection.addWidget(scroll_area2)
        
        layout2.addWidget(Engineering_insights_layout_widget)
        
        centralLayout = QVBoxLayout()
        centralLayout.setContentsMargins(0,0,0,0)
        centralLayout.addWidget(widget2)
        self.setLayout(centralLayout)
   
    
    def resizeEvent(self, event):
        button_size = self.Default_Pos.sizeHint()
        self.overlay.setGeometry(
            self.width() - button_size.width() - int(0.005208333 * self.screen_width),  # 10 px margin from the right
            self.height() - button_size.height() - int(0.005208333 * self.screen_width),  # 10 px margin from the bottom
            button_size.width(),
            button_size.height()
        )
        self.overlay.raise_()
        
    def reset_to_initial_Pos(self):
        
        self.Engineering_insights_label.x = -(int(0.46875 * self.screen_width))
        self.Engineering_insights_label.y = -(int(0.238020833 * self.screen_width))
        self.Engineering_insights_label.F = 0.5530550291791955
        self.Engineering_insights_label.setPixmap(QPixmap("Help/engineering-insights.png"))
        self.Engineering_insights_label.offset = QPoint(self.Engineering_insights_label.x, self.Engineering_insights_label.y)
        self.Engineering_insights_label.zoom_factor = self.Engineering_insights_label.F
        self.Engineering_insights_label.update()
        
    def toggle_Note(self):
        self.scroll_area_Engineering_insights.setVisible(not self.scroll_area_Engineering_insights.isVisible())
        self.Default_Pos.setVisible(not self.Default_Pos.isVisible())
        self.Engineering_insights_label_Flag = not self.Engineering_insights_label_Flag
        
        if self.Engineering_insights_label_Flag == True:
            self.toggle_note.setIcon(QIcon("imgs/ref.png"))
            self.Engineering_insights_label_Flag = False
            
        else:
            self.toggle_note.setIcon(QIcon("imgs/reverse_ref.png"))
            self.Engineering_insights_label_Flag = True
            
    def exit_help_function(self):
        self.close()
    
    def openVideoWindow(self, video_path):
        video_player = VideoPlayer(video_path)
        video_player.setGeometry(0, 0, int(0.4167 * self.screen_width), int(0.3125 * self.screen_width))
        video_player.show()

class VisualizationWindow(QWidget):   
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Feature analysis and Visualizations")
        self.setWindowIcon(QIcon("imgs/Visualization.png"))
        
        
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint) 
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: transparent;")
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        
        self.setMinimumSize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setGeometry(int(0.20833 * self.screen_width), int(0.02604167 * self.screen_width), int(0.78125 * screen_width), int(0.925 * screen_height))

        
        
        
        self.Histogram_Intensity_label = QLabel("Intensity Histogram Window")
        self.Histogram_Intensity_label.setStyleSheet("color: red;font:25px")
        self.Histogram_Intensity_label.setAlignment(Qt.AlignCenter)
        self.histogram = None

        
        
        self.LBPHistogramWindow_label = QLabel("LBP Histogram Window")
        self.LBPHistogramWindow_label.setStyleSheet("color: white;font:25px")
        self.LBPHistogramWindow_label.setAlignment(Qt.AlignCenter)

        self.LTPHistogramWindow_label = QLabel("LTP Histogram Window")
        self.LTPHistogramWindow_label.setStyleSheet("color: white;font:25px")
        self.LTPHistogramWindow_label.setAlignment(Qt.AlignCenter)

            
            
            
            
        self.LBPImgWindow_label = QLabel("LBP Image")
        self.LTPImgWindow_label = QLabel("LTP Image")
        font = QFont()
        font.setPointSize(int(0.0104167 * self.screen_width))
        self.LBPImgWindow_label.setFont(font)
        self.LTPImgWindow_label.setFont(font)
        self.LBPImgWindow_label.setAlignment(Qt.AlignCenter)
        self.LTPImgWindow_label.setAlignment(Qt.AlignCenter)
        self.LBPImgWindow_label.setStyleSheet("color: white;")
        self.LTPImgWindow_label.setStyleSheet("color: white;")
    
        self.LBPandLTPImgsWindow = QHBoxLayout()
        self.LBPandLTPImgsWindow.addWidget(self.LBPImgWindow_label)
        self.LBPandLTPImgsWindow.addWidget(self.LTPImgWindow_label)
            
            
        
        
        self.CRegImgWindow_label = QLabel("Centeral Region Width Image")
        self.TibImgWindow_label = QLabel("Tibial Width Image")
        font = QFont()
        font.setPointSize(int(0.0104167 * self.screen_width))
        self.CRegImgWindow_label.setFont(font)
        self.TibImgWindow_label.setFont(font)
        self.CRegImgWindow_label.setAlignment(Qt.AlignCenter)
        self.TibImgWindow_label.setAlignment(Qt.AlignCenter)
        self.CRegImgWindow_label.setStyleSheet("color: white;")
        self.TibImgWindow_label.setStyleSheet("color: white;")

        self.CenteralRegandTibialWidthImgsWindow = QHBoxLayout()
        self.CenteralRegandTibialWidthImgsWindow.addWidget(self.CRegImgWindow_label)
        self.CenteralRegandTibialWidthImgsWindow.addWidget(self.TibImgWindow_label)
        
        self.Predicted_JSN_label = QLabel("Predicted JSN Original Image")
        font = QFont()
        font.setPointSize(int(0.0104167 * self.screen_width))
        self.Predicted_JSN_label.setFont(font)
        self.Predicted_JSN_label.setAlignment(Qt.AlignCenter)
        self.Predicted_JSN_label.setStyleSheet("color: white;")


        self.HOG_image_label = QLabel("HOG Image")
        font = QFont()
        font.setPointSize(int(0.0104167 * self.screen_width))
        self.HOG_image_label.setFont(font)
        self.HOG_image_label.setAlignment(Qt.AlignCenter)
        self.HOG_image_label.setStyleSheet("color: white;")
            
        self.HOGHistogram = HOGHistogram()

        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(64, 164, 64, 50), stop:1 rgba(64, 164, 64, 0));
        """
        self.setStyleSheet(gradient_style)
        
        centralLayout = QVBoxLayout()
        centralLayout.setContentsMargins(0,0,0,0)
        self.setLayout(centralLayout)

        self.Vertical_Layout1 = QVBoxLayout()
        self.Vertical_Layout1.addWidget(self.Histogram_Intensity_label)
        self.Vertical_Layout1_Widget = QWidget()
        self.Vertical_Layout1_Widget.setLayout(self.Vertical_Layout1)
        
        self.Vertical_Layout2 = QVBoxLayout()
        self.Vertical_Layout2.addLayout(self.LBPandLTPImgsWindow)
        self.Vertical_Layout2_Widget = QWidget()
        self.Vertical_Layout2_Widget.setLayout(self.Vertical_Layout2)
        
        self.Vertical_Layout3 = QVBoxLayout()
        self.Vertical_Layout3.addWidget(self.LBPHistogramWindow_label)
        self.Vertical_Layout3_Widget = QWidget()
        self.Vertical_Layout3_Widget.setLayout(self.Vertical_Layout3)
        
        self.Vertical_Layout4 = QVBoxLayout()
        self.Vertical_Layout4.addWidget(self.LTPHistogramWindow_label)
        self.Vertical_Layout4_Widget = QWidget()
        self.Vertical_Layout4_Widget.setLayout(self.Vertical_Layout4)
        
        self.Vertical_Layout5 = QVBoxLayout()
        self.Vertical_Layout5.addLayout(self.CenteralRegandTibialWidthImgsWindow)
        self.Vertical_Layout5_Widget = QWidget()
        self.Vertical_Layout5_Widget.setLayout(self.Vertical_Layout5)
        
        self.Vertical_Layout6 = QVBoxLayout()
        self.Vertical_Layout6_Widget = QWidget()
        self.Vertical_Layout6_Widget.setLayout(self.Vertical_Layout6)
        
        self.Horizontal_Layout_V5 = QHBoxLayout()
        self.Horizontal_Layout_V5.addWidget(self.Predicted_JSN_label)
        self.Horizontal_Layout_V5.addWidget(self.HOG_image_label)
        self.Horizontal_Layout_V5_Widget = QWidget()
        self.Horizontal_Layout_V5_Widget.setLayout(self.Horizontal_Layout_V5)
        
        self.Vertical_Layout6.addWidget(self.Horizontal_Layout_V5_Widget)
        
        self.Horizonatal1 = QHBoxLayout()        
        self.Horizonatal1.addWidget(self.Vertical_Layout1_Widget)
        self.Horizonatal1.addWidget(self.Vertical_Layout2_Widget)
        self.Horizonatal1_Widget = QWidget()
        self.Horizonatal1_Widget.setLayout(self.Horizonatal1)

        self.Horizonatal2 = QHBoxLayout()
        self.Horizonatal2.addWidget(self.Vertical_Layout3_Widget)
        self.Horizonatal2.addWidget(self.Vertical_Layout4_Widget)
        self.Horizonatal2_Widget = QWidget()
        self.Horizonatal2_Widget.setLayout(self.Horizonatal2)

        self.Horizonatal3 = QHBoxLayout()
        self.Horizonatal3.addWidget(self.Vertical_Layout5_Widget)
        self.Horizonatal3.addWidget(self.Horizontal_Layout_V5_Widget)
        self.Horizonatal3_Widget = QWidget()
        self.Horizonatal3_Widget.setLayout(self.Horizonatal3)

        centralLayout.addWidget(self.Horizonatal1_Widget)
        centralLayout.addWidget(self.Horizonatal2_Widget)
        centralLayout.addWidget(self.Horizonatal3_Widget)
        
        
        
    
    def set_histogram(self, histogram):
        self.histogram = histogram
        self.display_histogram()

    def display_histogram(self):
        if self.histogram is not None:
            max_value = max(self.histogram)
            hist_image = np.zeros((256, 256, 3), dtype=np.uint8)

            for i in range(256):
                if max_value != 0:
                    normalized_value = int(self.histogram[i] * 255 / max_value)
                else:
                    normalized_value = 0
                color = QColor(255, 255 - normalized_value, 255 - normalized_value)
                hist_image[255 - normalized_value:, i, :] = color.getRgb()[:3]

            height, width, channel = hist_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(hist_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.Histogram_Intensity_label.setPixmap(QPixmap.fromImage(q_img))
            self.Histogram_Intensity_label.setScaledContents(True)
            
            
    def set_lbp_histogram(self, histogram):
        if histogram is not None:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(histogram)), histogram, align="center")
            plt.title("Local Binary Pattern Histogram")
            plt.xlabel("LBP Patterns")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("Temp/Saved Figures/lbp_histogram.png")
            lbp_histogram_image = QImage("Temp/Saved Figures/lbp_histogram.png")
            self.LBPHistogramWindow_label.setPixmap(QPixmap.fromImage(lbp_histogram_image))
            self.LBPHistogramWindow_label.setScaledContents(True)
            plt.close()
            
        
    def set_LTPHistogramWindow_histogram(self, histogram):
        if histogram is not None:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(histogram)), histogram, align="center")
            plt.title("Local Ternary Pattern Histogram")
            plt.xlabel("LTP Patterns")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("Temp/Saved Figures/ltp_histogram.png")
            ltp_histogram_image = QImage("Temp/Saved Figures/ltp_histogram.png")
            self.LTPHistogramWindow_label.setPixmap(QPixmap.fromImage(ltp_histogram_image))
            self.LTPHistogramWindow_label.setScaledContents(True)
            plt.close()
    
    
    def set_lbp_image(self, lbp_image):
        if lbp_image is not None:
            height, width = lbp_image.shape
            bytes_per_line = width
            q_img = QImage(lbp_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.LBPImgWindow_label.setPixmap(QPixmap.fromImage(q_img))
            self.LBPImgWindow_label.setScaledContents(True)


    def set_ltp_image(self, ltp_image):
        if ltp_image is not None:
            height, width = ltp_image.shape
            bytes_per_line = width
            q_img = QImage(ltp_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.LTPImgWindow_label.setPixmap(QPixmap.fromImage(q_img))
            self.LTPImgWindow_label.setScaledContents(True)  
            
            
    def set_CRegWidth_Image(self, image):
        if image is not None:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.CRegImgWindow_label.setPixmap(QPixmap.fromImage(q_img))
            self.CRegImgWindow_label.setScaledContents(True)

    def set_TibialWidth_Image(self, image):
        if image is not None:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.TibImgWindow_label.setPixmap(QPixmap.fromImage(q_img))
            self.TibImgWindow_label.setScaledContents(True)
            
    def set_Predicted_JSN_Image(self, image):
        if image is not None:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.Predicted_JSN_label.setPixmap(QPixmap.fromImage(q_img))
            self.Predicted_JSN_label.setScaledContents(True)
    
    def set_HOG_Image(self, image):
        if image is not None:
            if image.dtype != np.uint8:
                image = (image - image.min()) / (image.max() - image.min()) * 255.0
                image = image.astype(np.uint8)

            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.HOG_image_label.setPixmap(QPixmap.fromImage(q_img))
            self.HOG_image_label.setScaledContents(True)
            
            
            
#  _____________________________________________ Feature Extraction and Visualization Screen ____________________________________________________________
class Feature_Extraction_and_Visualization_Screen(QWidget):
    def __init__(self, main_window, shared_data):
        super().__init__()
        
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        self.resize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setMinimumSize(int(0.78125 * screen_width), int(0.925 * screen_height))
        self.setMaximumSize(screen_width, screen_height)
        
        self.screen_width = screen_width
        self.screen_height = screen_height       
        
        self.main_window = main_window
        self.main_window.variableChanged_Equalize_Feature_Visualization.connect(self.handle_variable_changed_Equalized)
        self.main_window.variableChanged_Auto_mode.connect(self.handle_variable_changed_Auto_mode)
        self.main_window.variableChanged_Visualization.connect(self.handle_variable_changed_Visualization)
        
        self.shared_data = shared_data
        self.setMouseTracking(True)
        self.show_main_content()
        
    def show_main_content(self):
        self.HOGHistogram = HOGHistogram()
        self.HOGHistogram.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.VisualizationWindow = VisualizationWindow()
        self.VisualizationWindow.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)
        self.HOGHistogram.setStyleSheet(gradient_style)
        
        self.image_label = ImageLabel("Manual Feature Extractor")
        self.image_label.setToolTip("Manually Extract the image-based Features")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setMouseTracking(True)
        self.JSN_label = ImageLabel(self, apply_effects = False)
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.JSN_label, image_path)
        self.JSN_label.setAlignment(Qt.AlignCenter)
        self.JSN_label.setScaledContents(True)
        self.JSN_label.setMouseTracking(True)
        self.JSN_label.mouseMoveEvent = self.showToolTip
        self.YOLO_label = ImageLabel("ROI Detection")
        self.YOLO_label.setAlignment(Qt.AlignCenter)
        self.YOLO_label.setScaledContents(True)
        self.YOLO_label.setMouseTracking(True)
        self.YOLO_label.mouseMoveEvent = self.showToolTip_ROI_Detector
        self.Intensity_label = ImageLabel("Equalization")
        self.Intensity_label.setAlignment(Qt.AlignCenter)
        self.Intensity_label.setStyleSheet("color: white;")
        self.Binarization_label = ImageLabel("Binarization")
        self.Binarization_label.setAlignment(Qt.AlignCenter)
        self.Binarization_label.setStyleSheet("color: white;")
        self.edge_label = ImageLabel("Edge Detection")
        self.edge_label.setAlignment(Qt.AlignCenter)
        self.edge_label.setStyleSheet("color: white;")
        font = QFont()
        font.setPointSize(int(0.0104167 * self.screen_width))
        self.image_label.setFont(font)
        self.JSN_label.setFont(font)
        self.YOLO_label.setFont(font)
        self.Intensity_label.setFont(font)
        self.Binarization_label.setFont(font)
        self.edge_label.setFont(font)
        
        self.image_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")
        # self.JSN_label.setStyleSheet(" border-radius: 50px; background-image: url(imgs/Feature_Extraction.png); background-repeat: no-repeat; background-color: rgba(0, 0, 0, 0); color: darkgray;")
        self.YOLO_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")
        self.Intensity_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")
        self.Binarization_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")
        self.edge_label.setStyleSheet(" border-radius: 100px; background-color: rgba(64, 64, 64, 100); color: darkgray;")

        self.image_label.updateEffectsBasedOnStyleAndPixmap()        
        self.JSN_label.updateEffectsBasedOnStyleAndPixmap()        
        self.YOLO_label.updateEffectsBasedOnStyleAndPixmap()        
        self.Intensity_label.updateEffectsBasedOnStyleAndPixmap()        
        self.Binarization_label.updateEffectsBasedOnStyleAndPixmap()        
        self.edge_label.updateEffectsBasedOnStyleAndPixmap()        

        layoutH2_space1 = QHBoxLayout()
        layoutH2_space1_Widget = QWidget()
        layoutH2_space1_Widget.setFixedHeight(int(0.0104167 * self.screen_width))
        layoutH2_space1_Widget.setLayout(layoutH2_space1)
        layoutH2_space1_Widget.setStyleSheet("""
                                             background-color: transparent;
                                             border: none;
                                             """)
        
        layoutH2_space2 = QHBoxLayout()
        layoutH2_space2_Widget = QWidget()
        layoutH2_space2_Widget.setFixedHeight(int(0.02604167 * self.screen_width))
        layoutH2_space2_Widget.setLayout(layoutH2_space2)
        layoutH2_space2_Widget.setStyleSheet("""
                                             background-color: transparent;
                                             border: none;
                                             """)
        
        
        layout = QGridLayout()
        layoutH1 = QHBoxLayout()
        layout.addWidget(layoutH2_space1_Widget, 0, 0, 1, 1)
        
        layout.addLayout(layoutH1, 1, 0, 1, 1)
        layoutH1.addWidget(self.image_label)
        layoutH1.addWidget(self.JSN_label)
        layoutH1.addWidget(self.YOLO_label)
        layoutH = QHBoxLayout()
        layout.addLayout(layoutH, 2, 0, 1, 1)
  
        layout.addWidget(layoutH2_space2_Widget, 3, 0, 1, 1)

        layoutH.addWidget(self.Intensity_label)
        layoutH.addWidget(self.Binarization_label)
        layoutH.addWidget(self.edge_label)
        
        self.setLayout(layout)
        
        
        # self.histogram_window = HistogramWindow()
        # self.histogram_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.VisualizationWindow.set_histogram(None)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_histogram_display)
        self.timer.start(1)
        
# _____________________________________________ Iitialization _______________________________________________
        self.folder_name = "Temp"
        self.mouse_double_clicked = 0
        self.first_itr = 1
        self.interval = 0
        self.duration = 0
        self.switch = True
        self.switch2 = False
        self.Equalize_button_switch = True
        self.mode = True
        self.padding_indicator = 0    
        self.image = None
        self.image_path = None
        self.Conventional_image = None
        self.Conventional_image_path = None
        self.cropped_images = []
        self.user_cropped = False        
        self.setAcceptDrops(True)
        self.center_crop = ()
        self.perform_intensity_normalization = True
        self.perform_canny_edge = True
        self.end_Automation = 0
        self.conventional_image_Indicator = 0
        self.check_knee_literality = False
        self.percentage = 0
        self.prob = 0.0
        self.Centeral_Left_X = 0
        self.Centeral_Right_X = 0
        self.Centeral_Left_Y  = 0
        self.Centeral_Right_Y = 0
        self.Centeral_Region_Width = 0
        self.Centeral_Region_Height = 0
        self.Centeral_Region_Center = ()
        self.Xc =  0 
        self.Yc = 0
        self.Tibial_width = 0
        self.start_Tibial_width = 0
        self.end_Tibial_width = 0
        self.average_medial_distance = 0
        self.average_central_distance = 0
        self.average_lateral_distance = 0
        self.average_medial_distance_mm = 0
        self.average_central_distance_mm = 0
        self.average_lateral_distance_mm = 0
        self.Medial_ratio = 0
        self.Central_ratio = 0
        self.Lateral_ratio = 0
        self.average_distance = 0
        self.average_distance_mm = 0
        self.red_area = 0
        self.green_area = 0
        self.blue_area = 0
        self.medial_area = 0
        self.central_area = 0
        self.lateral_area = 0
        self.medial_area_Squaredmm = 0
        self.central_area_Squaredmm = 0
        self.lateral_area_Squaredmm = 0
        self.medial_area_Ratio = 0
        self.central_area_Ratio = 0
        self.lateral_area_Ratio = 0
        self.JSN_Area_Total = 0
        self.JSN_Area_Total_Squared_mm = 0
        self.Tibial_width_Predicted_Area = 0 
        self.Tibial_width_Predicted_Area_mm = 0
        self.medial_area_Ratio_TWPA = 0
        self.central_area_Ratio_TWPA = 0
        self.lateral_area_Ratio_TWPA = 0
        self.intensity_mean = 0
        self.intensity_stddev = 0
        self.intensity_skewness = 0
        self.intensity_kurtosis = 0
        self.cooccurrence_properties = {}
        self.lbp_features = []
        self.lbp_variance = 0
        self.lbp_entropy = 0
        self.lbp_features_Normalized = []
        self.lbp_variance_Normalized = 0
        self.lbp_entropy_Normalized = 0
        self.ltp_features = []
        self.ltp_variance = 0
        self.ltp_entropy = 0
        self.ltp_features_Normalized = []
        self.ltp_variance_Normalized = 0
        self.ltp_entropy_Normalized = 0
        self.hog_bins = []
        self.hog_bins_Normalized = []
        self.progress_Flag = False      
#                        ____________________ Calculate Area Ratio______________________________
        self.screw_thickness1 = 4.5
        self.screw_thickness2 = 3.0
        self.screw_length = 20
        self.screw_area_standard = self.screw_thickness1* 0.75* self.screw_length + self.screw_thickness2* 0.25* self.screw_length
        self.screw_area_virtual = 471 #Squared pixel
        self.area_ratio = self.screw_area_virtual /  self.screw_area_standard
#                        ____________________ Calculate Length Ratio______________________________
        self.virtual_length = 140 #unit pixel
        self.length_ratio = self.virtual_length / self.screw_length
#  __________________________________________________ Functions _______________________________________________________    
    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")
        
    def handle_variable_changed_Equalized(self, new_value):
        self.perform_intensity_normalization = new_value
       
    def handle_variable_changed_Auto_mode(self, new_value):
        self.switch = new_value
    
    def handle_variable_changed_Visualization(self, new_value):
        self.switch2 = new_value
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        image_path = "imgs/Feature_Extraction.png"
        
        if self.mouse_double_clicked == 1:
            pass
        else:
            self.set_background_image(self.JSN_label, image_path)
          
    def showToolTip(self, event):
        if self.image is not None:            
            if (self.check_knee_literality):
                subregions_area = ["Medial Area (Squared mm)", "Central Area (Squared mm)", "Lateral Area (Squared mm)"]
                subregions_distance = ["Medial Distance (mm)", "Central Distance (mm)", "Lateral Distance (mm)"]
                areas = [self.medial_area_Squaredmm, self.central_area_Squaredmm, self.lateral_area_Squaredmm]
                distances = [self.average_medial_distance_mm, self.average_central_distance_mm, self.average_lateral_distance_mm]
            else:
                subregions_area = ["Lateral Area (Squared mm)", "Central Area (Squared mm)", "Medial Area (Squared mm)"]
                subregions_distance = ["Lateral Distance (mm)", "Central Distance (mm)", "Medial Distance (mm)"]
                areas = [self.lateral_area_Squaredmm, self.central_area_Squaredmm, self.medial_area_Squaredmm]
                distances = [self.average_lateral_distance_mm, self.average_central_distance_mm, self.average_medial_distance_mm]
                
            subregion_width = self.JSN_label.width() / 3
            subregion_boundaries = [(i * subregion_width, (i + 1) * subregion_width) for i in range(3)]
            for i, (start, end) in enumerate(subregion_boundaries):
                if start <= event.pos().x() < end:
                    pos = self.mapToGlobal(self.JSN_label.pos())
                    rect = self.JSN_label.rect()
                    QToolTip.showText(event.globalPos(), f"{subregions_area[i]}: {areas[i]}\n{subregions_distance[i]}: {distances[i]} ", self, QRect(pos.x(), pos.y(), rect.width(), rect.height()))
                    return

            QToolTip.hideText()    
        else:
            pass
    
    def showToolTip_ROI_Detector(self, event):
        if self.YOLO_label.text() == "ROI Detection":
            pass
        else:
            if self.prob is not None:
                QToolTip.showText(event.globalPos(), f"Probability is {self.prob}")
            else:
                QToolTip.showText(event.globalPos(), "")
        
    def switch2_toggle(self):
        if(self.switch2):
            self.VisualizationWindow.show()
            self.HOGHistogram.show()
        else:
            self.VisualizationWindow.hide()
            self.HOGHistogram.hide()

    def load_Conventional_Image(self):
        
        self.conventional_image_Indicator = 2
        
        print(f"self.padding_indicator: {self.padding_indicator}")
        
        self.Conventional_image = self.equalize_Original_image(self.Conventional_image)
        
        self.display_image(self.Conventional_image, target_label=self.image_label)
        self.display_image(self.Conventional_image, target_label=self.YOLO_label)
        self.display_image(self.Conventional_image, target_label=self.JSN_label)
        self.display_image(self.Conventional_image, target_label=self.VisualizationWindow.CRegImgWindow_label)
        self.display_image(self.Conventional_image, target_label=self.VisualizationWindow.TibImgWindow_label)
        self.display_image(self.Conventional_image, target_label=self.VisualizationWindow.Predicted_JSN_label)
        
        
        self.check_knee_literality = self.set_knee_literality(self.Conventional_image)
            
        self.YOLO_CReg_predict(self.Conventional_image)
        self.YOLO_TibialW_predict(self.Conventional_image)

        self.predicted_polygon = self.YOLO_predict_Segmented(self.Conventional_image)
        self.draw_predicted_polygon(self.predicted_polygon, self.JSN_label)                
        self.draw_predicted_polygon_Original(self.predicted_polygon, self.VisualizationWindow.Predicted_JSN_label)
        
        self.red_area, self.green_area, self.blue_area = self.calculate_three_subregions_areas()

        if (self.check_knee_literality):
            self.medial_area = self.red_area
            self.central_area = self.green_area
            self.lateral_area = self.blue_area
            
        else:
            self.medial_area = self.blue_area
            self.central_area = self.green_area
            self.lateral_area = self.red_area
                            
        
        if self.Tibial_width_Predicted_Area != 0:
            self.medial_area_Ratio_TWPA = self.medial_area / self.Tibial_width_Predicted_Area
            self.central_area_Ratio_TWPA = self.central_area / self.Tibial_width_Predicted_Area
            self.lateral_area_Ratio_TWPA = self.lateral_area / self.Tibial_width_Predicted_Area
            
            self.medial_area_Ratio_TWPA *= 100 
            self.central_area_Ratio_TWPA *= 100
            self.lateral_area_Ratio_TWPA *= 100


        yolo_results = self.YOLO_predict(self.Conventional_image, self.padding_indicator)
        self.calculate_and_save_features_YOLO_Conventional(yolo_results, self.Conventional_image_path,self.Conventional_image)
        # print(f"\033[92myolo_results = {yolo_results}.\033[0m")
        self.save_PNGs("Temp/Saved Figures")
        print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        self.on_button_click()

    def load_image(self, image_path):
        
        self.image_label.disableEffects()        
        self.JSN_label.disableEffects()        
        self.YOLO_label.disableEffects()        
        self.Intensity_label.disableEffects()        
        self.Binarization_label.disableEffects()        
        self.edge_label.disableEffects() 
                        
        self.user_cropped = False
        self.padding_indicator = 0
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        self.image = self.Apply_Padding(self.image)
            
        if (self.perform_intensity_normalization):
            self.image = self.equalize_Original_image(self.image)

        else:
            pass
        
        self.display_image(self.image, target_label=self.image_label)
        self.display_image(self.image, target_label=self.YOLO_label)
        self.display_image(self.image, target_label=self.JSN_label)
        self.display_image(self.image, target_label=self.VisualizationWindow.CRegImgWindow_label)
        self.display_image(self.image, target_label=self.VisualizationWindow.TibImgWindow_label)
        self.display_image(self.image, target_label=self.VisualizationWindow.Predicted_JSN_label)
          
        if (self.switch):
            directory_path = os.path.dirname(image_path)            
            self.folder_name = os.path.basename(directory_path)
            
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            self.check_knee_literality = self.set_knee_literality(self.image)       
        
            self.YOLO_CReg_predict(self.image)
            self.YOLO_TibialW_predict(self.image)

            self.predicted_polygon = self.YOLO_predict_Segmented(self.image)
            self.draw_predicted_polygon(self.predicted_polygon, self.JSN_label)                
            self.draw_predicted_polygon_Original(self.predicted_polygon, self.VisualizationWindow.Predicted_JSN_label)

            self.red_area, self.green_area, self.blue_area = self.calculate_three_subregions_areas()

            if (self.check_knee_literality):
                self.medial_area = self.red_area
                self.central_area = self.green_area
                self.lateral_area = self.blue_area
                
            else:
                self.medial_area = self.blue_area
                self.central_area = self.green_area
                self.lateral_area = self.red_area
                                
            print(f"\033[92mMedial Area = {self.medial_area}.Squared Pixel\033[0m")
            print(f"\033[92mCentral Area = {self.central_area}.Squared Pixel\033[0m")
            print(f"\033[92mLateral Area = {self.lateral_area}.Squared Pixel\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            
            self.medial_area_Squaredmm = self.medial_area / self.area_ratio
            self.central_area_Squaredmm = self.central_area / self.area_ratio
            self.lateral_area_Squaredmm = self.lateral_area / self.area_ratio
            
            print(f"\033[92mMedial Area = {self.medial_area_Squaredmm}.Squared mm\033[0m")
            print(f"\033[92mCentral Area = {self.central_area_Squaredmm}.Squared mm\033[0m")
            print(f"\033[92mLateral Area = {self.lateral_area_Squaredmm}.Squared mm\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            if self.JSN_Area_Total != 0:
                self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
                self.central_area_Ratio = self.central_area / self.JSN_Area_Total
                self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total
                
                print(f"\033[92mMedial Area Ratio = {self.medial_area_Ratio}.\033[0m")
                print(f"\033[92mCentral Area Ratio = {self.central_area_Ratio}.\033[0m")
                print(f"\033[92mLateral Area Ratio = {self.lateral_area_Ratio}.\033[0m")
                
                
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
            if self.Tibial_width_Predicted_Area != 0:
                self.medial_area_Ratio_TWPA = self.medial_area / self.Tibial_width_Predicted_Area
                self.central_area_Ratio_TWPA = self.central_area / self.Tibial_width_Predicted_Area
                self.lateral_area_Ratio_TWPA = self.lateral_area / self.Tibial_width_Predicted_Area
                
                self.medial_area_Ratio_TWPA *= 100 
                self.central_area_Ratio_TWPA *= 100
                self.lateral_area_Ratio_TWPA *= 100
            
                print(f"\033[92mMedial Area Ratio TWPA (%) = {self.medial_area_Ratio_TWPA}.\033[0m")
                print(f"\033[92mCentral Area Ratio TWPA (%) = {self.central_area_Ratio_TWPA}.\033[0m")
                print(f"\033[92mLateral Area Ratio TWPA (%) = {self.lateral_area_Ratio_TWPA}.\033[0m")
        
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

            yolo_results = self.YOLO_predict(self.image, self.padding_indicator)
            self.calculate_and_save_features_YOLO(yolo_results, image_path, self.image)
            print(f"\033[92myolo_results = {yolo_results}.\033[0m")
            print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
        
        else:
            
            self.current = image_path
                
            directory_path = os.path.dirname(image_path)
            
            if image_path.lower().endswith(('.dcm')):
                image_files = glob.glob(os.path.join(directory_path, "*.dcm"))
            
            elif image_path.lower().endswith(('.ima')):
                image_files = glob.glob(os.path.join(directory_path, "*.ima"))
                
            elif image_path.lower().endswith(('.jpeg')):
                image_files = glob.glob(os.path.join(directory_path, "*.jpeg"))
            
            elif image_path.lower().endswith(('.jpg')):
                image_files = glob.glob(os.path.join(directory_path, "*.jpg"))
                
            elif image_path.lower().endswith(('.png')):
                image_files = glob.glob(os.path.join(directory_path, "*.png"))
                self.total_image_files = len(image_files)
            else:
                print("\033[91mUnsupported image format.\033[0m")
                pass
            
            self.folder_name = os.path.basename(directory_path)
            
            counter = 0
            for image_path in image_files:
                QApplication.processEvents()  # Process UI events to keep it responsive
                
                if self.first_itr == 1:
                    self.main_window.overlay.setVisible(True)
                    QApplication.processEvents()  # Process UI events to keep it responsive
                    st_itr = time.time()
                
                if self.first_itr == 0:
                    self.main_window.overlay.setVisible(False)
                
                counter+=1
                self.padding_indicator = 0

                self.image_path = image_path
                self.image = cv2.imread(self.image_path)

                self.image = self.Apply_Padding(self.image)
            
                if (self.perform_intensity_normalization):
                    self.image = self.equalize_Original_image(self.image)
                else:
                    pass
                
                # print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")
                self.check_knee_literality = self.set_knee_literality(self.image)
                
                self.YOLO_CReg_predict(self.image)
                self.YOLO_TibialW_predict(self.image)
                
                self.predicted_polygon = self.YOLO_predict_Segmented(self.image)
                self.draw_predicted_polygon(self.predicted_polygon, self.JSN_label)
                self.draw_predicted_polygon_Original(self.predicted_polygon, self.VisualizationWindow.Predicted_JSN_label)

                self.red_area, self.green_area, self.blue_area = self.calculate_three_subregions_areas()

                if (self.check_knee_literality):
                    self.medial_area = self.red_area
                    self.central_area = self.green_area
                    self.lateral_area = self.blue_area
                    
                else:
                    self.medial_area = self.blue_area
                    self.central_area = self.green_area
                    self.lateral_area = self.red_area
                                    
                self.medial_area_Squaredmm = self.medial_area / self.area_ratio
                self.central_area_Squaredmm = self.central_area / self.area_ratio
                self.lateral_area_Squaredmm = self.lateral_area / self.area_ratio

                if self.JSN_Area_Total != 0:
            
                    self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
                    self.central_area_Ratio = self.central_area / self.JSN_Area_Total
                    self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total


                if self.Tibial_width_Predicted_Area != 0:
                    self.medial_area_Ratio_TWPA = self.medial_area / self.Tibial_width_Predicted_Area
                    self.central_area_Ratio_TWPA = self.central_area / self.Tibial_width_Predicted_Area
                    self.lateral_area_Ratio_TWPA = self.lateral_area / self.Tibial_width_Predicted_Area
                    
                    self.medial_area_Ratio_TWPA *= 100 
                    self.central_area_Ratio_TWPA *= 100
                    self.lateral_area_Ratio_TWPA *= 100
            
                
                yolo_results = self.YOLO_predict(self.image, self.padding_indicator)
                self.calculate_and_save_features_YOLO(yolo_results, self.image_path, self.image)
                print(f"\033[92mCounter = {counter}.\033[0m")
        
                QApplication.processEvents()  # Process UI events to keep it responsive
                
                if self.first_itr == 1:
                    end_itr = time.time()
                    
                    self.first_itr = 0
                    self.interval = (end_itr - st_itr)  # Interval in seconds
                    print(f"Interval for one iteration: {self.interval} seconds")

                    self.duration = self.interval * (len(image_files) - 1)  # Duration in seconds
                    print(f"Total estimated duration: {self.duration} seconds for remaining {(len(image_files) - 1)} images")

                    self.main_window.overlay.setVisible(False)
                    
                    self.main_window.toggleMinimized()

                    self.timer_window = CountdownTimer(self.duration//10)
                    self.timer_thread = threading.Thread(target=self.timer_window.show())
                    self.timer_thread.start()

                QApplication.processEvents()  # Process UI events to keep it responsive
                
            self.main_window.toggleMinimized()
            self.first_itr = 1
            st_itr = 0
            end_itr = 0
            self.interval = 0
            self.duration = 0
                    

            
            self.end_Automation = 2
            self.on_button_click()
            self.image = cv2.imread(self.current)
                   
            self.padding_indicator = 0
            self.image = self.Apply_Padding(self.image)
            
            if (self.perform_intensity_normalization):
                self.image = self.equalize_Original_image(self.image)
            else:
                pass
        
            self.display_image(self.image, target_label=self.image_label)
            self.display_image(self.image, target_label=self.YOLO_label)
            self.display_image(self.image, target_label=self.JSN_label)
            self.display_image(self.image, target_label=self.VisualizationWindow.CRegImgWindow_label)
            self.display_image(self.image, target_label=self.VisualizationWindow.TibImgWindow_label)
            self.display_image(self.image, target_label=self.VisualizationWindow.Predicted_JSN_label)

            # print(f"\033[92m____________________________________________________________________________________________________________________\033[0m")

            self.check_knee_literality = self.set_knee_literality(self.image)

            self.YOLO_CReg_predict(self.image)
            self.YOLO_TibialW_predict(self.image)
            
            self.predicted_polygon = self.YOLO_predict_Segmented(self.image)
            self.draw_predicted_polygon(self.predicted_polygon, self.JSN_label)
            self.draw_predicted_polygon_Original(self.predicted_polygon, self.VisualizationWindow.Predicted_JSN_label)

            self.red_area, self.green_area, self.blue_area = self.calculate_three_subregions_areas()

            if (self.check_knee_literality):
                self.medial_area = self.red_area
                self.central_area = self.green_area
                self.lateral_area = self.blue_area
                
            else:
                self.medial_area = self.blue_area
                self.central_area = self.green_area
                self.lateral_area = self.red_area
                
            self.medial_area_Squaredmm = self.medial_area / self.area_ratio
            self.central_area_Squaredmm = self.central_area / self.area_ratio
            self.lateral_area_Squaredmm = self.lateral_area / self.area_ratio
                
            if self.JSN_Area_Total != 0:
                self.medial_area_Ratio = self.medial_area / self.JSN_Area_Total
                self.central_area_Ratio = self.central_area / self.JSN_Area_Total
                self.lateral_area_Ratio = self.lateral_area / self.JSN_Area_Total
                
            if self.Tibial_width_Predicted_Area != 0:
                self.medial_area_Ratio_TWPA = self.medial_area / self.Tibial_width_Predicted_Area
                self.central_area_Ratio_TWPA = self.central_area / self.Tibial_width_Predicted_Area
                self.lateral_area_Ratio_TWPA = self.lateral_area / self.Tibial_width_Predicted_Area
                
                self.medial_area_Ratio_TWPA *= 100 
                self.central_area_Ratio_TWPA *= 100
                self.lateral_area_Ratio_TWPA *= 100
            
            yolo_results = self.YOLO_predict(self.image, self.padding_indicator)

        if self.perform_intensity_normalization:
            self.equalize_histogram()
            self.binarization()

        if self.perform_canny_edge:
            self.save_PNGs("Temp/Saved Figures")
            img = cv2.imread("Temp/Saved Figures/JSN-Region.png")
            self.perform_edge_detection(img)

        self.calculate_histogram()
        self.save_PNGs("Temp/Saved Figures")
     
    def Apply_Padding(self, image):
        height, width, _ = image.shape
        
        if height != width:
            if height >  width : 
                padding = (height - width) //2
                padded_image = cv2.copyMakeBorder(image, 0 , 0, padding, padding, cv2.BORDER_CONSTANT, value=[0 , 0 , 0])
                image = padded_image
                image = cv2.resize(image, (224, 224))    
                self.padding_indicator = 1

                return image

            else:

                padding = np.abs((height - width) //2)
                padded_image = cv2.copyMakeBorder(image, padding , padding, 0, 0, cv2.BORDER_CONSTANT, value=[0 , 0 , 0])
                image = padded_image
                image = cv2.resize(image, (224, 224))    
                self.padding_indicator = 2
                
                return image
                    
        elif height == width:
            image = cv2.resize(image, (224, 224))  
            self.padding_indicator = 0
            
            return image
            
    def equalize_Original_image(self, img):
        original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_img = cv2.equalizeHist(original_gray)
        self.equalized_image_rgb = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB)
        
        return self.equalized_image_rgb

    def display_image(self, img, target_label=None):
        if img is not None:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 2:
                    height, width = img.shape
                    bytes_per_line = width
                    format = QImage.Format_Grayscale8
                else:
                    height, width, channel = img.shape
                    bytes_per_line = 3 * width
                    format = QImage.Format_RGB888
                
                img_bytes = img.tobytes()
                q_img = QImage(img_bytes, width, height, bytes_per_line, format)
            elif isinstance(img, QImage):
                q_img = img
            else:
                print("\033[91mUnsupported image format.\033[0m")
                return
            if target_label is not None:
                target_label.setPixmap(QPixmap.fromImage(q_img))
                target_label.setScaledContents(True)
            else:
                self.image_label.setPixmap(QPixmap.fromImage(q_img))
                self.image_label.setScaledContents(True)

    
    def set_knee_literality(self, img):
        resize = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
        device = torch.device("cpu")
        resize_tensor = torch.from_numpy(np.expand_dims(np.transpose(resize, (2, 0, 1)), axis=0)).to(device)
        
        if not hasattr(self, 'knee_Literality_model'):
            self.knee_Literality_model = SimpleCNN()
            self.knee_Literality_model.load_state_dict(torch.load('Models/NewFeatureExtractionAndVisualizationModels/Knee_Literality.pth'))
            self.knee_Literality_model.eval()
                
        output = self.knee_Literality_model(resize_tensor)
        if output.item() > 0.5:
            print(f"\033[92mLeft Knee.\033[0m")
            return True
        else:
            print(f"\033[92mRight Knee.\033[0m")
            return False
  
    def YOLO_predict_Segmented(self, img):
        if not hasattr(self, 'model_instance_Segmentation'):
            self.model_instance_Segmentation =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_Instance Segmentation.pt")
        
        if img is not None and self.model_instance_Segmentation is not None:
            output = self.model_instance_Segmentation.predict(img,
                                                                            save=False,
                                                                            show=False,
                                                                            show_labels=False,
                                                                            show_conf=True,
                                                                            save_txt=False)
            masks = output[0].masks

            if masks is not None:
                masks = masks.xy
                self.predicted_polygon = [mask.tolist() for mask in masks]
                return self.predicted_polygon
            else:
                print("\033[91mMasks are None. No segmentation data available.\033[0m")
                return []
        else:
            print("\033[91mImage or model not available.\033[0m")
            return []

    def draw_predicted_polygon_Original(self, polygon, target_label):
        if polygon and target_label is not None:
            pixmap = target_label.pixmap()
            painter = QPainter(pixmap)
            color = QColor(0, 0, 0)
            color.setAlpha(255)
            painter.setPen(QPen(color))
            painter.setBrush(QBrush(color))
            scaled_polygon = [QPointF(float(x), float(y)) for point in polygon for x, y in point]
            painter.drawPolygon(QPolygonF(scaled_polygon))

    def draw_predicted_polygon(self, polygon, target_label):
        if polygon and target_label is not None:
                pixmap = target_label.pixmap()
                painter = QPainter(pixmap)
                color = QColor(50, 255, 50)
                color.setAlpha(255)
                painter.setPen(QPen(color))
                painter.setBrush(QBrush(color))
                path = QPainterPath()
                path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for point in polygon for x, y in point]))
                painter.setClipPath(path)
                painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for point in polygon for x, y in point]))

                if len(polygon) > 0:
                    
                    left_x = min(point[0] for point in polygon[0])
                    right_x = max(point[0] for point in polygon[0])

                    if self.yolo_results_CReg == [] or self.yolo_results_TibialW == []:
                        self.Medial_ratio = 0
                        self.Central_ratio = 0
                        self.Lateral_ratio = 0
                            
                        return


                    medial_region_start = left_x
                    medial_region_end = (self.Tibial_width / 2 - 0.5 * self.Centeral_Region_Width) + self.start_Tibial_width
                    central_region_start = medial_region_end
                    central_region_end = (self.Tibial_width / 2 + 0.5 * self.Centeral_Region_Width) + self.start_Tibial_width
                    lateral_region_start = central_region_end
                    lateral_region_end = right_x
                
                    self.color_red = QColor(255, 0, 0)
                    self.color_green = QColor(0, 255, 0)
                    self.color_blue = QColor(0, 0, 255)
                    self.color_red.setAlpha(255)
                    self.color_green.setAlpha(255)
                    self.color_blue.setAlpha(255)
                    self.draw_subregion(painter, pixmap, polygon, medial_region_start, medial_region_end, self.color_red)
                    self.draw_subregion(painter, pixmap, polygon, central_region_start, central_region_end, self.color_green)
                    self.draw_subregion(painter, pixmap, polygon, lateral_region_start, lateral_region_end, self.color_blue)
                    
                    polygon = [[[int(coord) for coord in point] for point in sublist] for sublist in polygon]
                    left_x = min(point[0] for  point in polygon[0])
                    right_x = max(point[0] for point in polygon[0])
                                
                    if self.Tibial_width != 0:
                        self.percentage = (right_x - left_x)/self.Tibial_width
                    self.Automatic_JSW_tibial_ratio = 0.926155
                    self.JSW_width = self.Automatic_JSW_tibial_ratio * self.Tibial_width
                    assumed_left = self.start_Tibial_width + 0.5 * self.Tibial_width - 0.5 * self.JSW_width
                    assumed_right = assumed_left + self.JSW_width
                    
                    polygon = [[[int(coord) for coord in point] for point in sublist] for sublist in polygon]
                    two_dim_array = np.array([point for sublist in polygon for point in sublist])

                    if self.end_Automation == 2:
                        self.image_path = self.current
                        self.end_Automation = 0
                    else:
                        pass
                    
                    if self.conventional_image_Indicator == 2:
                        img = cv2.imread(self.Conventional_image_path)
                        self.conventional_image_Indicator = 0                 
                    else:
                        img = cv2.imread(self.image_path)
                        self.padding_indicator = 0
                        img = self.Apply_Padding(img)
                        
                    self.fill_colored_polygon(img, two_dim_array)
                    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255)

    # ______________________________  Vertical Distance and Area _________________________________ #

                    column_index = int(self.Xc)
                    row_index  = int (self.Yc) 

                    Upper_index= row_index
                    down_index = row_index

                    distace_Central = [] 
                    distace_Right = []
            
                    for move_right in range (int(assumed_right - column_index)):
                        if  ((column_index +move_right)>right_x)  and((column_index +move_right)< assumed_right):
                            distace_Right.append(0)

                        else :
                            try: 
                                if edges[row_index] [column_index+ move_right] > 0  :
                                    row_index = int((Upper_index + down_index)/2)

                                upper_point_flag = 0
                                upper_point = 0 
                                Upper_index = row_index
                                down_point_flag = 0
                                down_point = 0
                                
                                down_index = row_index
                            #  get upper point to get vertical distance 
                                while upper_point_flag == 0 :
                                    if  edges[Upper_index] [column_index +move_right] >0:
                                        upper_point_flag = 1
                                        upper_point = Upper_index   
                                    Upper_index -= 1
                            #  get dwon point to get vertical distance  
                                while down_point_flag == 0:
                                    if  edges[down_index] [column_index +move_right] >0:
                                        down_point_flag = 1
                                        down_point = down_index
                                    down_index += 1

                                if ((column_index +move_right) >= lateral_region_start) and ((column_index +move_right)<=right_x):
                                    distace_Right.append(down_point -upper_point)

                                elif  ((column_index +move_right) < lateral_region_start) and ((column_index + move_right) >  medial_region_end):
                                    distace_Central.append(down_point -upper_point)
                            
                                else:
                                    pass
                            except:
                                continue 

                    column_index = int(self.Xc)
                    row_index  = int (self.Yc)
                    
                    Upper_index= row_index
                    down_index = row_index

                    distace_Left = []

                    for move_left in range (int(column_index - assumed_left)):
                        if  ((column_index - move_left) < left_x)  and((column_index - move_left)> assumed_left):
                            distace_Left.append(0)

                        else : 
                            try:
                                if edges[row_index] [column_index- move_left] >0  :
                                    row_index = int((Upper_index +down_index)/2)

                                upper_point_flag = 0
                                upper_point = 0
                                Upper_index= row_index

                                down_point_flag = 0
                                down_point = 0
                                down_index = row_index

                                # get upper point to get vertical distance 
                                while upper_point_flag == 0 :
                                    if  edges[Upper_index] [column_index  - move_left] >0:
                                        upper_point_flag = 1
                                        upper_point = Upper_index   
                                    Upper_index -=1

                                # get down point to get vertical distance 
                                while down_point_flag == 0:
                                    if  edges[down_index] [column_index  - move_left] >0:
                                        down_point_flag = 1
                                        down_point = down_index
                                    down_index += 1

                                if ((column_index  - move_left) >= left_x ) and ((column_index  - move_left) <= medial_region_end):
                                    distace_Left.append(down_point -upper_point)

                                elif  ((column_index - move_left) < lateral_region_start) and ((column_index - move_left) >  medial_region_end):
                                    distace_Central.append(down_point -upper_point)
                            
                                else:
                                    pass
                            except:
                                continue
                    
                    if distace_Central == []:
                        pass
                    
                    else:
                        distace_Central.pop(0)

                    self.average_medial_distance =  np.mean(distace_Left)
                    self.average_central_distance =  np.mean (distace_Central)
                    self.average_lateral_distance = np.mean(distace_Right)
                    
                    if (self.average_medial_distance is None):
                        self.average_medial_distance = 0
                        
                    if (self.average_central_distance is None):
                        self.average_central_distance = 0
                        
                    if (self.average_lateral_distance is None):
                        self.average_lateral_distance = 0
                    
                        
                    self.average_distance = (self.average_medial_distance + self.average_central_distance + self.average_lateral_distance) / 3
                    self.average_distance_mm = self.average_distance / self.length_ratio
                    
                    self.medial_area = np.sum(distace_Left)
                    self.central_area = np.sum(distace_Central)
                    self.lateral_area = np.sum(distace_Right)
                    
                    if (self.medial_area is None):
                        self.medial_area = 0
                        
                    if (self.central_area is None):
                        self.central_area = 0
                        
                    if (self.lateral_area is None):
                        self.lateral_area = 0
                        
                    self.JSN_Area_Total =  self.medial_area + self.central_area + self.lateral_area
                    self.JSN_Area_Total_Squared_mm = self.JSN_Area_Total / self.area_ratio
                    
                    
                    
                    del polygon
                    del left_x
                    del right_x
                    del self.JSW_width
                    del assumed_left
                    del assumed_right
                    del two_dim_array
                    del edges
                    del img
                    del column_index
                    del row_index 
                    del Upper_index
                    del down_index
                    del distace_Central
                    del distace_Right
                    del distace_Left
                    
                    # self.Xc = 0
                    # self.Yc = 0
    #__________________________________________  end ______________________________________________#
                    
                    if (self.check_knee_literality):
                        pass
                    else:
                        self.spare = self.average_medial_distance
                        self. average_medial_distance = self.average_lateral_distance
                        self.average_lateral_distance = self.spare
                        
                    # print(f"\033[92m               ___________________________________________________________       \033[0m")

                    # print(f"\033[92mAvg V.Distance (Medial): {self.average_medial_distance} Pixel.\033[0m")
                    # print(f"\033[92mAvg V.Distance (Central): {self.average_central_distance} Pixel.\033[0m")
                    # print(f"\033[92mAvg V.Distance (Lateral): {self.average_lateral_distance} Pixel.\033[0m")
                    
                    # print(f"\033[92m               ___________________________________________________________       \033[0m")

                    self.average_medial_distance_mm = self.average_medial_distance / self.length_ratio
                    self.average_central_distance_mm = self.average_central_distance / self.length_ratio
                    self.average_lateral_distance_mm = self.average_lateral_distance / self.length_ratio
                    
                    # print(f"\033[92mAvg V.Distance (Medial): {self.average_medial_distance_mm} mm.\033[0m")
                    # print(f"\033[92mAvg V.Distance (Central): {self.average_central_distance_mm} mm.\033[0m")
                    # print(f"\033[92mAvg V.Distance (Lateral): {self.average_lateral_distance_mm} mm.\033[0m")
                    
                    
                    # print(f"\033[92m               ___________________________________________________________       \033[0m")
                    
                
                    if (self.Tibial_width != 0):
                        
                        self.Medial_ratio = self.average_medial_distance/self.Tibial_width
                        self.Central_ratio = self.average_central_distance/self.Tibial_width
                        self.Lateral_ratio = self.average_lateral_distance/self.Tibial_width
                    

                        # print(f"\033[92mAvg Tibial V.Distance ratio (Medial): {self.Medial_ratio}.\033[0m")
                        # print(f"\033[92mAvg Tibial V.Distance ratio (Central): {self.Central_ratio}.\033[0m")
                        # print(f"\033[92mAvg Tibial V.Distance ratio (Lateral): {self.Lateral_ratio}.\033[0m")
                        # print(f"\033[92m               ___________________________________________________________       \033[0m")
                        
                    else:
                        pass
                        print("\033[91mTibial Width: {self.Tibial_width} Pixel.\033[0m")
                    
                    painter.end()
                    target_label.setPixmap(pixmap)
                    target_label.setScaledContents(True)    
                else:
                    print("\033[91mNo points in the polygon to calculate distances from.\033[0m")

        else:
            pass


    def draw_subregion(self, painter, pixmap, polygon, start_x, end_x, color):
        subregion_path = QPainterPath()
        subregion_path.addRect(start_x, 0, end_x - start_x, pixmap.height())
        painter.setClipPath(subregion_path)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for point in polygon for x, y in point]))
  
    def calculate_three_subregions_areas(self):
        return self.medial_area, self.central_area, self.lateral_area
  
    def fill_colored_polygon(self , image, two_dim_array, color=(0, 0, 0)):
            two_dim_array = np.array([two_dim_array], dtype=np.int32)
            cv2.fillPoly(image, [two_dim_array], color=color)

    def YOLO_CReg_predict(self, img):
            if not hasattr(self, 'model_YOLO_CenteralReg'):
                self.model_YOLO_CenteralReg =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_CenteralReg.pt")
            
            if self.model_YOLO_CenteralReg is not None and img is not None:
                predict_imageCReg = self.model_YOLO_CenteralReg.predict(img)
                self.yolo_results_CReg = self.get_coordinate_Predict_image(predict_imageCReg)
                self.draw_yolo_roi_box_CReg(self.yolo_results_CReg, img)
            else:
                return "\033[91mNo image or model available.\033[0m"
    
    def draw_yolo_roi_box_CReg(self, yolo_results, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                self.Centeral_Left_X, self.Centeral_Left_Y, self.Centeral_Right_X, self.Centeral_Right_Y, class_name, prob = result
                
                if prob > 0.1:
                    top_left = QPoint(self.Centeral_Left_X, self.Centeral_Left_Y)
                    bottom_right = QPoint(self.Centeral_Right_X, self.Centeral_Right_Y)
                    self.Centeral_Region_Width = self.Centeral_Right_X - self.Centeral_Left_X
                    self.Centeral_Region_Height = self.Centeral_Right_Y - self.Centeral_Left_Y
                    self.Xc = self.Centeral_Left_X + self.Centeral_Region_Width/2
                    self.Yc = self.Centeral_Left_Y + self.Centeral_Region_Height/2
                    self.Centeral_Region_Center = (self.Xc , self.Yc)
                    painter = QPainter(self.VisualizationWindow.CRegImgWindow_label.pixmap())
                    color = QColor(0, 255, 0)
                    color.setAlpha(255)
                    pen = QPen(color)
                    pen.setWidth(2)
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                    painter.drawRect(QRect(top_left, bottom_right))
                    painter.end()
                    self.VisualizationWindow.CRegImgWindow_label.update()
                    print(f"\033[92mCenteral Region Width: = {self.Centeral_Region_Width} Pixel.\033[0m")
                    print(f"\033[92mCenteral Region Height: = {self.Centeral_Region_Height} Pixel.\033[0m")
                    print(f"\033[92mCenteral Region Center: = {self.Centeral_Region_Center}.\033[0m")
                    
                    if self.Centeral_Region_Width is None:
                        self.Centeral_Region_Width = 0
                        self.Centeral_Region_Height = 0
                        
                        
    def YOLO_TibialW_predict(self, img):
                if not hasattr(self, 'model_YOLO_TibialWidth'):
                    self.model_YOLO_TibialWidth =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_TibialWidth.pt")
            
                if self.model_YOLO_TibialWidth is not None and img is not None:
                    predict_imageTibialW = self.model_YOLO_TibialWidth.predict(img)
                    self.yolo_results_TibialW = self.get_coordinate_Predict_image(predict_imageTibialW)
                    self.draw_yolo_roi_box_TibialW(self.yolo_results_TibialW, img)
                else:
                    return "\033[91mNo image or model available.\033[0m"
        
    def draw_yolo_roi_box_TibialW(self, yolo_results, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                self.start_Tibial_width, y1, self.end_Tibial_width, y2, class_name, prob = result
                if prob > 0.1:
                    top_left = QPoint(self.start_Tibial_width, y1)
                    bottom_right = QPoint(self.end_Tibial_width, y2)
                    self.Tibial_width = self.end_Tibial_width-self.start_Tibial_width
                    self.Tibial_width_Predicted_Area = self.Tibial_width * (y2-y1)
                    self.Tibial_width_Predicted_Area_mm = self.Tibial_width_Predicted_Area / self.area_ratio
                    painter = QPainter(self.VisualizationWindow.TibImgWindow_label.pixmap())
                    color = QColor(0, 255, 0)
                    color.setAlpha(255)
                    pen = QPen(color)
                    pen.setWidth(2)
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                    painter.drawRect(QRect(top_left, bottom_right))
                    painter.end()
                    self.VisualizationWindow.TibImgWindow_label.update()
                    print(f"\033[92mTibial Width: = {self.Tibial_width} Pixel.\033[0m")

                    if self.Tibial_width is None:
                        self.Tibial_width = 0
    
    def YOLO_predict(self, img, padding_indicator):
        if not hasattr(self, 'old_model_detection'):
            self.old_model_detection =YOLO("Models/NewFeatureExtractionAndVisualizationModels/YOLO_ROI detection.pt")
        if not hasattr(self, 'new_model_detection'):
            self.new_model_detection =YOLO("Models/NewFeatureExtractionAndVisualizationModels/best.pt")
        
        if self.old_model_detection is not None and self.new_model_detection is not None and img is not None:
            if padding_indicator == 0:
                img = cv2.resize(img , (224 ,224))
                predict_image = self.old_model_detection.predict(img)
                output = self.get_coordinate_Predict_image(predict_image)

                if  len(output) == 0 : 
                    predict_image = self.new_model_detection.predict(img)
               
            else:
                img = cv2.resize(img , (224 ,224))
                predict_image = self.new_model_detection.predict(img)
            
            yolo_results = self.get_coordinate_Predict_image(predict_image)
            self.draw_yolo_roi_box(yolo_results, img)
            return yolo_results
        else:
            return "\033[91mNo image or model available.\033[0m"
        
 
 
 
    def get_coordinate_Predict_image(self, predict_image):
        rlts = predict_image [0]
        output = [] 
        prob_flag = 0.0
        
        for box  in rlts.boxes:
            x1 ,y1,x2,y2 = [
                round(x) for x in box.xyxy[0].tolist()
            ]
            class_id = box.cls[0].item()
            prob =round(box.conf[0].item(),2)
            if prob > prob_flag :
                output.append([
                x1 ,y1,x2,y2 ,rlts.names[class_id] , prob
                                ])
                prob_flag = prob
        
        if len(output) > 1:
            output = output[-1]
    
        return  output
    
    
    def draw_yolo_roi_box(self, yolo_results, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                x1, y1, x2, y2, class_name, prob = result
                if prob > 0.1:
                    top_left = QPoint(x1, y1)
                    bottom_right = QPoint(x2, y2)
                    painter = QPainter(self.YOLO_label.pixmap())
                    color = QColor(0, 255, 0)
                    color.setAlpha(255)
                    pen = QPen(color)
                    pen.setWidth(4)
                    pen.setStyle(Qt.SolidLine)
                    painter.setPen(pen)
                    painter.drawRect(QRect(top_left, bottom_right))
                    painter.end()
                    self.YOLO_label.update()

    
    def Save_ALL(self):
        save_directory = QFileDialog.getExistingDirectory(None, "Select Directory to Save Files", QDir.homePath())

        if not save_directory:
            return
        
        save_as_dcm = self.ask_save_as_dcm_dialog()

        text = "All figures have been saved successfully.\n"
        msgBox = CustomMessageBox("Custom message for logo", "imgs/LogoSplashScreen_Original.png", "OK", 0)
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setWindowIcon(QIcon("imgs/save.svg"))
        msgBox.setWindowTitle("save")
        msgBox.setText(text)
        # msgBox.addButton(QMessageBox.Ok)
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        msgBox.exec_()
        
        if save_as_dcm:
            self.save_PNGs_as_DICOM(save_directory)
        else:
            self.save_PNGs(save_directory)
        
    def ask_save_as_dcm_dialog(self):
        dialog = QMessageBox()
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle("Save As")
        dialog.setText("Do you want to save as DICOM or PNG?")
        dialog.addButton("DICOM", QMessageBox.YesRole)
        dialog.addButton("PNG", QMessageBox.NoRole)
        result = dialog.exec_()

        return result == 0
    
    def save_PNGs(self, save_directory):
        labels = [
            (self.image_label, "Manual Feature Extractor image.png"),
            (self.JSN_label, "JSN_label_Sub-Regions.png"),
            (self.YOLO_label, "YOLO_ROI.png"),
            (self.Intensity_label, "Intensity_label.png"),
            (self.Binarization_label, "Binarization_label.png"),
            (self.edge_label, "canny-edge-label.png"),
            (self.VisualizationWindow.Predicted_JSN_label, "JSN-Region.png"),
            # (self.HistogramOfOrientedGradientsImage.HOG_image_label, "HOG Image.png"),
            (self.VisualizationWindow.HOG_image_label, "HOG Image.png"),
            (self.HOGHistogram.HOGHistogram_label, "HOG_Histogram.png"),
            (self.VisualizationWindow.LBPImgWindow_label, "lbp_image.png"),
            (self.VisualizationWindow.LTPImgWindow_label, "ltp_image.png"),
            (self.VisualizationWindow.TibImgWindow_label, "Tibial Width.png"),
            (self.VisualizationWindow.CRegImgWindow_label, "Centeral Region Width.png"),
            # (self.histogram_window.Histogram_Intensity_label, "histogram.png"),
            (self.VisualizationWindow.Histogram_Intensity_label, "histogram.png"),
            (self.VisualizationWindow.LBPHistogramWindow_label, "lbp_histogram.png"),
            # (self.LTPHistogramWindow.LTPHistogramWindow_label, "ltp_histogram.png")            
            (self.VisualizationWindow.LTPHistogramWindow_label, "ltp_histogram.png")            
        ]

        for label, filename in labels:
            filepath = os.path.join(save_directory, filename)
            pixmap = label.grab()
            img = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            width, height = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4)
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            resized_image = cv2.resize(image, (224, 224))
            cv2.imwrite(filepath, resized_image)

        save_directory = None
    
    def save_PNGs_as_DICOM(self, save_directory):
        labels = [
            (self.image_label, "Manual Feature Extractor image.png"),
            (self.JSN_label, "JSN_label_Sub-Regions.png"),
            (self.YOLO_label, "YOLO_ROI.png"),
            (self.Intensity_label, "Intensity_label.png"),
            (self.Binarization_label, "Binarization_label.png"),
            (self.edge_label, "canny-edge-label.png"),
            (self.VisualizationWindow.Predicted_JSN_label, "JSN-Region.png"),
            (self.VisualizationWindow.HOG_image_label, "HOG Image.png"),
            (self.HOGHistogram.HOGHistogram_label, "HOG_Histogram.png"),
            (self.VisualizationWindow.LBPImgWindow_label, "lbp_image.png"),
            (self.VisualizationWindow.LTPImgWindow_label, "ltp_image.png"),
            (self.VisualizationWindow.TibImgWindow_label, "Tibial Width.png"),
            (self.VisualizationWindow.CRegImgWindow_label, "Centeral Region Width.png"),
            (self.VisualizationWindow.Histogram_Intensity_label, "histogram.png"),
            (self.VisualizationWindow.LBPHistogramWindow_label, "lbp_histogram.png"),
            (self.VisualizationWindow.LTPHistogramWindow_label, "ltp_histogram.png")            
        ]
        
        for label, filename in labels:
            filepath = os.path.join(save_directory, filename)
            pixmap = label.grab()
            img = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            width, height = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4)
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            image = cv2.resize(image, (224, 224))
            dicom_dataset = self.convert_image_to_dicom(image)
            dicom_filepath = filepath.replace(".png", ".dcm")
            dicom_dataset.save_as(dicom_filepath, write_like_original=False)
        
        save_directory = None
    
    def convert_image_to_dicom(self, image):
        dicom_dataset = FileDataset(None, {}, preamble=b"\0" * 128)

        dicom_dataset.Rows, dicom_dataset.Columns = image.shape
        dicom_dataset.SamplesPerPixel = 1
        dicom_dataset.BitsAllocated = 8
        dicom_dataset.BitsStored = 8
        dicom_dataset.HighBit = 7
        dicom_dataset.PixelRepresentation = 0
        dicom_dataset.PhotometricInterpretation = "MONOCHROME2"

        dicom_dataset.PixelSpacing = [1.0, 1.0]
        dicom_dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
        dicom_dataset.RescaleIntercept = 0
        dicom_dataset.RescaleSlope = 1

        image = image.astype(np.uint8)
        pixel_data = image.flatten()
        dicom_dataset.PixelData = pixel_data.tobytes()

        dicom_dataset.WindowCenter = (np.max(image).astype(np.int16) + np.min(image).astype(np.int16)) / 2
        dicom_dataset.WindowWidth = (np.max(image) - np.min(image)).astype(np.int16)
        
        dicom_dataset.SOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')  # X-Ray Image Storage
        dicom_dataset.Modality = "DX"

        dicom_dataset.SOPInstanceUID = generate_uid()
        dicom_dataset.file_meta = Dataset()
        dicom_dataset.file_meta.MediaStorageSOPInstanceUID = dicom_dataset.SOPInstanceUID
        dicom_dataset.file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')  # X-Ray Image Storage
        dicom_dataset.file_meta.TransferSyntaxUID = UID('1.2.840.10008.1.2.1')  # Explicit VR Little Endian
        dicom_dataset.file_meta.ImplementationClassUID = UID('1.2.826.0.1.3680043.9.7461.1')

        return dicom_dataset
    #  _____________________________ setter ____________________________________  #
    
    def set_image(self, image):
        self.Conventional_image = image
    
    def set_path(self, path):
        self.Conventional_image_path = path
                
    #  _____________________________ getter ____________________________________  #    
    def get_Medial_ratio(self):
        return self.Medial_ratio
    
    def get_Central_ratio(self):
        return self.Central_ratio
    
    def get_Lateral_ratio(self):
        return self.Lateral_ratio
    
    def get_medial_area_Ratio_TWPA (self):
        return self.medial_area_Ratio_TWPA 
    
    def get_central_area_Ratio_TWPA (self):
        return self.central_area_Ratio_TWPA 
    
    def get_lateral_area_Ratio_TWPA (self):
        return self.lateral_area_Ratio_TWPA 
    
    def get_intensity_mean(self):
        return self.intensity_mean
        
    def get_intensity_stddev(self):
        return self.intensity_stddev
    
    def get_intensity_skewness(self):
        return self.intensity_skewness
    
    def get_intensity_kurtosis(self):
        return self.intensity_kurtosis
    
    def get_cooccurrence_properties(self):
        return self.cooccurrence_properties
    
    def get_lbp_features_Normalized(self):
        return self.lbp_features_Normalized
    
    def get_lbp_variance_Normalized(self):
        return self.lbp_variance_Normalized
    
    def get_lbp_entropy_Normalized(self):
        return self.lbp_entropy_Normalized
    
    def get_ltp_features_Normalized(self):
        return self.ltp_features_Normalized
        
    def get_ltp_variance_Normalized(self):
        return self.ltp_variance_Normalized
    
    def get_ltp_entropy_Normalized(self):
        return self.ltp_entropy_Normalized
    
    def get_HOG_Normalized(self):
        return self.hog_bins_Normalized
   #  ____________________________________________________________________________  #
    
    def calculate_histogram(self):
        if self.user_cropped and self.cropped_images:
            cropped_image = self.cropped_images[-1]
            if cropped_image is not None and not np.isnan(cropped_image).any() and cropped_image.size > 0:
                histogram = cv2.calcHist([cropped_image], [0], None, [256], [0, 256])
                self.VisualizationWindow.set_histogram(histogram)
                self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis = self.calculate_intensity_stats(cropped_image)
                top_left, bottom_right = self.calculate_cropped_rect_coords(cropped_image)
                self.display_intensity_stats_coordinates(self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis, top_left, bottom_right)
        elif not self.user_cropped:
            histogram = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            self.VisualizationWindow.set_histogram(histogram)


    def calculate_cropped_rect_coords(self, cropped_image):
        if cropped_image is not None:
            original_image_size = self.image.shape[:2]
            crop_rect = self.image_label.crop_rect
            top_left = (
                int(crop_rect.left() * original_image_size[1] / self.image_label.width()),
                int(crop_rect.top() * original_image_size[0] / self.image_label.height())
            )
            bottom_right = (
                int(crop_rect.right() * original_image_size[1] / self.image_label.width()),
                int(crop_rect.bottom() * original_image_size[0] / self.image_label.height())
            )
            return top_left, bottom_right
        else:
            return None, None

    def perform_edge_detection(self, img):
        if img is not None:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.perform_intensity_normalization:
                equalized_image = cv2.equalizeHist(gray_image)
                edges = cv2.Canny(equalized_image, 50, 150)
            else:
                edges = cv2.Canny(gray_image, 50, 150)

            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.display_edge_image(edges_rgb)

    def display_edge_image(self, img):
        if img is not None:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.edge_label.setPixmap(QPixmap.fromImage(q_img))
            self.edge_label.setScaledContents(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.image is not None:
                self.image_label.crop_rect.setTopLeft(event.pos())
                self.image_label.crop_rect.setBottomRight(event.pos())
                self.image_label.mouse_pressed = True

    def mouseMoveEvent(self, event):
        if self.image_label and hasattr(self.image_label, 'mouse_pressed') and self.image_label.mouse_pressed:
            self.image_label.crop_rect.setBottomRight(event.pos())
            self.image_label.update()


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.image is not None:
                self.image_label.mouse_pressed = False
                self.image_label.crop_rect.setBottomRight(event.pos())
                
                original_image_size = self.image.shape[:2]
                
                top_left = (int(self.image_label.crop_rect.left() * original_image_size[1] / self.image_label.width()),
                            int(self.image_label.crop_rect.top() * original_image_size[0] / self.image_label.height()))
                bottom_right = (int(self.image_label.crop_rect.right() * original_image_size[1] / self.image_label.width()),
                                int(self.image_label.crop_rect.bottom() * original_image_size[0] / self.image_label.height()))
                
                
                cropped_image = self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                if cropped_image is not None and not np.isnan(cropped_image).any() and cropped_image.size > 0:
                    self.cropped_images.append(cropped_image)
                    self.calculate_histogram()
                    self.user_cropped = True
                    self.calculate_and_save_all_parameters(cropped_image, self.image_path,top_left,bottom_right)
                            
    def mouseDoubleClickEvent(self, event):
        self.main_window.load_main_img()

    def equalize_histogram(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)
            self.display_equalized_image(equalized_image_rgb)

    def display_equalized_image(self, img):
        if img is not None:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.Intensity_label.setPixmap(QPixmap.fromImage(q_img))
            self.Intensity_label.setScaledContents(True)

    def binarization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            ret, binary_otsu = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.display_binarized_image(binary_otsu)
            return binary_otsu
        
    def display_binarized_image(self, img):
        if img is not None:
            if len(img.shape) == 2:
                height, width = img.shape
                bytes_per_line = width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            self.Binarization_label.setPixmap(pixmap)
            self.Binarization_label.setScaledContents(True)
            
    def on_button_click(self):
        self.mouse_double_clicked = 0
        self.Conventional_image = None
        self.Conventional_image_path = None
        self.image = None
        self.image_path = None
        self.cropped_images = []
        self.user_cropped = False
        self.setAcceptDrops(True)
        self.image_label.clear()
        self.JSN_label.clear()
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.JSN_label, image_path)
        self.YOLO_label.clear()
        self.Intensity_label.clear()
        self.Binarization_label.clear()
        self.edge_label.clear()
        self.VisualizationWindow.Histogram_Intensity_label.clear()
        self.VisualizationWindow.LBPImgWindow_label.clear()
        self.VisualizationWindow.LTPImgWindow_label.clear()
        self.VisualizationWindow.LBPHistogramWindow_label.clear()
        self.VisualizationWindow.LTPHistogramWindow_label.clear()
        self.VisualizationWindow.CRegImgWindow_label.clear()
        self.VisualizationWindow.TibImgWindow_label.clear()
        self.VisualizationWindow.Predicted_JSN_label.clear()
        self.VisualizationWindow.HOG_image_label.clear()
        self.HOGHistogram.HOGHistogram_label.clear()
        self.image_label.setText("Manual Feature Extractor")
        image_path = "imgs/Feature_Extraction.png"
        self.set_background_image(self.JSN_label, image_path)
        self.YOLO_label.setText("ROI Detection")
        self.Intensity_label.setText("Equalization")
        self.Binarization_label.setText("Binarization")
        self.edge_label.setText("Edge Detection")
        self.VisualizationWindow.Histogram_Intensity_label.setText("Intensity Histogram Window")
        self.VisualizationWindow.LBPImgWindow_label.setText("LBP")
        self.VisualizationWindow.LTPImgWindow_label.setText("LTP")
        self.VisualizationWindow.LBPHistogramWindow_label.setText("LBP Histogram Window")
        self.VisualizationWindow.LTPHistogramWindow_label.setText("LTP Histogram Window")
        self.VisualizationWindow.CRegImgWindow_label.setText("Centeral Region Width Image")
        self.VisualizationWindow.TibImgWindow_label.setText("Tibial Width Image")
        self.VisualizationWindow.Predicted_JSN_label.setText("Predicted JSN Original Image")
        self.VisualizationWindow.HOG_image_label.setText("HOG Image")
        self.HOGHistogram.HOGHistogram_label.setText("HOG Histogram")
        
        self.switch2 = False
        self.switch2_toggle()
                
                    
            
    
    def set_background_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setStyleSheet(f"background-image: url({image_path}); background-color: transparent; background-repeat: no-repeat; background-position: center; border: none;")
    
    def update_histogram_display(self):
        self.calculate_histogram()

    def calculate_intensity_stats(self, image):
        if image is not None:
            self.intensity_mean = np.mean(image)
            self.intensity_stddev = np.std(image)
            self.intensity_skewness = skew(image, axis=None)
            self.intensity_kurtosis = kurtosis(image, axis=None)
            
            return self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis
        else:
            return None, None, None, None

    def display_intensity_stats_coordinates(self, mean, stddev, intensity_skewness, intensity_kurtosis, top_left, bottom_right):
        if (mean is not None and stddev is not None) and (intensity_skewness is not None and intensity_kurtosis is not None):
            stats_text = f"\n\nIntensity Mean: {mean:.2f}\nIntensity StdDev: {stddev:.2f}\n" + \
             f"Intensity Skewness: {intensity_skewness:.2f}\nIntensity Kurtosis: {intensity_kurtosis:.2f}\n\n" + \
             f"Left-Top: {top_left[0]:.2f}, {top_left[1]:.2f}\n" + \
             f"Right-Bottom: {bottom_right[0]:.2f}, {bottom_right[1]:.2f}"

            self.Intensity_label.setText(stats_text)
        else:
            self.Intensity_label.clear()

    
    def calculate_cooccurrence_parameters_YOLO(self, image):
        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            max_probability = np.max(glcm)

            properties = {
                "contrast": graycoprops(glcm, 'contrast').mean(),
                "energy": graycoprops(glcm, 'energy').mean(),
                "correlation": graycoprops(glcm, 'correlation').mean(),
                "homogeneity": graycoprops(glcm, 'homogeneity').mean(),
                "dissimilarity": graycoprops(glcm, 'dissimilarity').mean(),
                "ASM": graycoprops(glcm, 'ASM').mean(),
                "max_probability": max_probability,
            }

            return properties
    
    def calculate_lbp_features_YOLO(self, image):
        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            radius = 2
            n_points = 8 * radius
            lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
            lbp_histogram, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            lbp_histogram_Normalized = lbp_histogram / (lbp_histogram.sum() + 1e-5)
            # lbp_histogram = lbp_histogram / (lbp_histogram.sum() + 1e-5)
            self.VisualizationWindow.set_lbp_histogram(lbp_histogram)
            # print("Shape of lbp_histogram_Normalized:", lbp_histogram_Normalized.shape)
            
            return lbp_histogram, lbp_image, lbp_histogram_Normalized

    def calculate_ltp_features_YOLO(self, image, num_bins=10):
        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            h, w = image.shape
            ltp_image = np.zeros((h, w), dtype=np.uint8)

            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    center_pixel = image[y, x]
                    binary_pattern = 0
                    binary_pattern |= (image[y - 1, x - 1] >= center_pixel) << 7
                    binary_pattern |= (image[y - 1, x] >= center_pixel) << 6
                    binary_pattern |= (image[y - 1, x + 1] >= center_pixel) << 5
                    binary_pattern |= (image[y, x + 1] >= center_pixel) << 4
                    binary_pattern |= (image[y + 1, x + 1] >= center_pixel) << 3
                    binary_pattern |= (image[y + 1, x] >= center_pixel) << 2
                    binary_pattern |= (image[y + 1, x - 1] >= center_pixel) << 1
                    binary_pattern |= (image[y, x - 1] >= center_pixel)

                    ltp_image[y, x] = binary_pattern

            num_bins += 1
            ltp_histogram, _ = np.histogram(ltp_image, bins=np.arange(0, num_bins), range=(0, num_bins))
            ltp_histogram_Normalized = ltp_histogram / (ltp_histogram.sum() + 1e-5)
            
            self.VisualizationWindow.set_LTPHistogramWindow_histogram(ltp_histogram)
            # print("Shape of ltp_histogram_Normalized:", ltp_histogram_Normalized.shape)
            
            return ltp_histogram, ltp_image, ltp_histogram_Normalized
    
    def compute_hog_features(self, image, cell_size=(8, 8), block_size=(2, 1), bins=9):
        if image is not None:
            image = cv2.resize(image, (224,224))
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            hog_features, hog_image = hog(image, orientations=bins, pixels_per_cell=cell_size,
                                          cells_per_block=block_size, block_norm='L2-Hys', visualize=True)
            epsilon = 1e-6
            hog_features_normalized = hog_features / (np.linalg.norm(hog_features) + epsilon)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            # print("Shape of hog_features:", hog_features.shape)
            # print("Shape of hog_features_normalized:", hog_features_normalized.shape)
            
            # self.HOGHistogram.set_HOG_Histogram(hog_features_normalized)
            self.VisualizationWindow.set_HOG_Image(hog_image_rescaled)
            # self.HistogramOfOrientedGradientsImage.set_HOG_Image(hog_image_rescaled)
            
            return hog_features, hog_features_normalized

        else:
            return []
        
            
    def calculate_and_save_all_parameters(self, cropped_image, image_path,top_left,bottom_right):
            if cropped_image is not None:
                cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                self.cooccurrence_properties = self.calculate_cooccurrence_parameters_YOLO(cropped_image_gray)
                self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis = self.calculate_intensity_stats(cropped_image_gray)
                self.lbp_features, self.lbp_image, self.lbp_features_Normalized = self.calculate_lbp_features_YOLO(cropped_image_gray)
                self.ltp_features, self.ltp_image, self.ltp_features_Normalized = self.calculate_ltp_features_YOLO(cropped_image_gray)               
                
                self.VisualizationWindow.set_lbp_image(self.lbp_image)
                self.VisualizationWindow.set_ltp_image(self.ltp_image)
                
                self.lbp_variance = np.var(self.lbp_features)
                self.ltp_variance = np.var(self.ltp_features)
                self.lbp_entropy = -np.sum(self.lbp_features * np.log2(self.lbp_features + 1e-5))
                self.ltp_entropy = -np.sum(self.ltp_features * np.log2(self.ltp_features + 1e-5))
                
                self.lbp_variance_Normalized = np.var(self.lbp_features_Normalized)
                self.ltp_variance_Normalized = np.var(self.ltp_features_Normalized)
                self.lbp_entropy_Normalized = -np.sum(self.lbp_features_Normalized * np.log2(self.lbp_features_Normalized + 1e-5))
                self.ltp_entropy_Normalized = -np.sum(self.ltp_features_Normalized * np.log2(self.ltp_features_Normalized + 1e-5))
                
                self.hog_bins, self.hog_bins_Normalized = self.compute_hog_features(cropped_image_gray)
                        
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                csv_file_path = "Temp/CSV/Manual.csv"
                write_header = not os.path.exists(csv_file_path)
                try:
                    with open(csv_file_path, mode="a", newline="") as csv_file:
                        fieldnames = ["Image Name", "Class","Crop Top-Left (X, Y)", "Crop Bottom-Right (X, Y)", "Mean", "Sigma", "Skewness", "Kurtosis"] + list(self.cooccurrence_properties.keys()) + \
                                    [f"LBP_{i}" for i in range(len(self.lbp_features))] + ["LBP_Variance", "LBP_Entropy"]+ \
                                    [f"LBP_Normalized_{i}" for i in range(len(self.lbp_features_Normalized))] + ["LBP_Variance_Normalized", "LBP_Entropy_Normalized"] + \
                                    [f"LTP_{i}" for i in range(len(self.ltp_features))] + ["LTP_Variance", "LTP_Entropy"] +\
                                    [f"LTP_Normalized_{i}" for i in range(len(self.ltp_features_Normalized))] + ["LTP_Variance_Normalized", "LTP_Entropy_Normalized"] + \
                                    [f"HOG_{i}" for i in range(len(self.hog_bins))] + \
                                    [f"HOG_Normalized_{i}" for i in range(len(self.hog_bins_Normalized))]
                                    
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                        row_data = {
                            "Image Name": file_name,
                            "Class":self.folder_name,
                            "Crop Top-Left (X, Y)": f"({top_left[0]}, {top_left[1]})",
                            "Crop Bottom-Right (X, Y)": f"({bottom_right[0]}, {bottom_right[1]})",
                            "Mean": self.intensity_mean,
                            "Sigma": self.intensity_stddev,
                            "Skewness": self.intensity_skewness,
                            "Kurtosis": self.intensity_kurtosis,
                            **self.cooccurrence_properties,
                            **{f"LBP_{i}": self.lbp_features[i] for i in range(len(self.lbp_features))},
                            "LBP_Variance": self.lbp_variance,
                            "LBP_Entropy": self.lbp_entropy,
                            **{f"LBP_Normalized_{i}": self.lbp_features_Normalized[i] for i in range(len(self.lbp_features_Normalized))},
                            "LBP_Variance_Normalized": self.lbp_variance_Normalized,
                            "LBP_Entropy_Normalized": self.lbp_entropy_Normalized,
                            **{f"LTP_{i}": self.ltp_features[i] for i in range(len(self.ltp_features))},
                            "LTP_Variance": self.ltp_variance,
                            "LTP_Entropy": self.ltp_entropy,
                            **{f"LTP_Normalized_{i}": self.ltp_features_Normalized[i] for i in range(len(self.ltp_features_Normalized))},
                            "LTP_Variance_Normalized": self.ltp_variance_Normalized,
                            "LTP_Entropy_Normalized": self.ltp_entropy_Normalized,
                            **{f"HOG_{i}": self.hog_bins[i] for i in range(len(self.hog_bins))},
                            **{f"HOG_Normalized_{i}": self.hog_bins_Normalized[i] for i in range(len(self.hog_bins_Normalized))}
                            
                        }
                        writer.writerow(row_data)
                        
                except Exception as e:
                        print("\033[91mError writing to statistics_Manual.csv:\033[0m", str(e))
            else:
                self.Intensity_label.clear()


    def calculate_and_save_features_YOLO_Conventional(self, yolo_results, image_path, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                x1, y1, x2, y2, class_name, self.prob = result
                if self.prob > 0.1:
                    roi_cropped_image = img[y1:y2, x1:x2]
                    if roi_cropped_image is not None:
                        self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis = self.calculate_intensity_stats(roi_cropped_image)
                        self.cooccurrence_properties = self.calculate_cooccurrence_parameters_YOLO(roi_cropped_image)
                        
                        self.lbp_features, self.lbp_image, self.lbp_features_Normalized = self.calculate_lbp_features_YOLO(roi_cropped_image)
                        
                        self.VisualizationWindow.set_lbp_image(self.lbp_image)
                        
                        self.lbp_variance = np.var(self.lbp_features)
                        self.lbp_entropy = -np.sum(self.lbp_features * np.log2(self.lbp_features + 1e-5))
                        self.lbp_variance_Normalized = np.var(self.lbp_features_Normalized)
                        self.lbp_entropy_Normalized = -np.sum(self.lbp_features_Normalized * np.log2(self.lbp_features_Normalized + 1e-5))
                            
                        self.ltp_features, self.ltp_image, self.ltp_features_Normalized = self.calculate_ltp_features_YOLO(roi_cropped_image)
                        
                        self.VisualizationWindow.set_ltp_image(self.ltp_image)
                        
                        if self.ltp_features is not None:
                            self.ltp_variance = np.var(self.ltp_features)
                            self.ltp_entropy = -np.sum(self.ltp_features * np.log2(self.ltp_features + 1e-5))
                        else:
                            self.ltp_variance = None
                            self.ltp_entropy = None
                        
                        if self.ltp_features_Normalized is not None:
                            self.ltp_variance_Normalized = np.var(self.ltp_features_Normalized)
                            self.ltp_entropy_Normalized = -np.sum(self.ltp_features_Normalized * np.log2(self.ltp_features_Normalized + 1e-5))
                        else:
                            self.ltp_variance_Normalized = None
                            self.ltp_entropy_Normalized = None
                            
                        roi_cropped_image_HOG = cv2.resize(roi_cropped_image, (206, 178))
                        self.hog_bins, self.hog_bins_Normalized = self.compute_hog_features(roi_cropped_image_HOG)
                        
                            
    def calculate_and_save_features_YOLO(self, yolo_results, image_path, img):
        if img is not None and yolo_results:
            for result in yolo_results:
                x1, y1, x2, y2, class_name, self.prob = result
                
                width_ROI = x2 - x1
                height_ROI = y2 - y1
                
                if self.prob > 0.1:
                    roi_cropped_image = img[y1:y2, x1:x2]
                    if roi_cropped_image is not None:

                        Histogram_properties_switch_parameter = self.shared_data["Histogram_properties_switch_parameter"]
                        print(f"Histogram_properties_switch_parameter: {Histogram_properties_switch_parameter}")

                        if Histogram_properties_switch_parameter == True:
                            self.intensity_mean, self.intensity_stddev, self.intensity_skewness, self.intensity_kurtosis = self.calculate_intensity_stats(roi_cropped_image)
                        
                        else:
                            self.intensity_mean = None
                            self.intensity_stddev = None
                            self.intensity_skewness = None
                            self.intensity_kurtosis = None   
                            
                        cooccurrence_properties_switch_parameter = self.shared_data["cooccurrence_properties_switch_parameter"]
                        print(f"cooccurrence_properties_switch_parameter: {cooccurrence_properties_switch_parameter}")

                        if cooccurrence_properties_switch_parameter == True:
                            self.cooccurrence_properties = self.calculate_cooccurrence_parameters_YOLO(roi_cropped_image)

                        else:
                            self.cooccurrence_properties = {}
                            
                        
                        lbp_switch_parameter = self.shared_data["lbp_switch_parameter"]
                        print(f"lbp_switch_parameter: {lbp_switch_parameter}")

                        if lbp_switch_parameter == True:
                            self.lbp_features, self.lbp_image, self.lbp_features_Normalized = self.calculate_lbp_features_YOLO(roi_cropped_image)
                            
                            self.VisualizationWindow.set_lbp_image(self.lbp_image)
                            
                            self.lbp_variance = np.var(self.lbp_features)
                            self.lbp_entropy = -np.sum(self.lbp_features * np.log2(self.lbp_features + 1e-5))
                            self.lbp_variance_Normalized = np.var(self.lbp_features_Normalized)
                            self.lbp_entropy_Normalized = -np.sum(self.lbp_features_Normalized * np.log2(self.lbp_features_Normalized + 1e-5))
                        
                        else:
                            self.lbp_image = None
                            self.lbp_features = []
                            self.lbp_features_Normalized = []
                            
                            self.lbp_variance = None
                            self.lbp_entropy = None
                            self.lbp_variance_Normalized = None
                            self.lbp_entropy_Normalized = None
                        
                        ltp_switch_parameter = self.shared_data["ltp_switch_parameter"]
                        print(f"ltp_switch_parameter: {ltp_switch_parameter}")

                        if ltp_switch_parameter == True:
                                
                            self.ltp_features, self.ltp_image, self.ltp_features_Normalized = self.calculate_ltp_features_YOLO(roi_cropped_image)
                            
                            self.VisualizationWindow.set_ltp_image(self.ltp_image)
                            
                            if self.ltp_features is not None:
                                self.ltp_variance = np.var(self.ltp_features)
                                self.ltp_entropy = -np.sum(self.ltp_features * np.log2(self.ltp_features + 1e-5))
                            else:
                                self.ltp_variance = None
                                self.ltp_entropy = None
                            
                            if self.ltp_features_Normalized is not None:
                                self.ltp_variance_Normalized = np.var(self.ltp_features_Normalized)
                                self.ltp_entropy_Normalized = -np.sum(self.ltp_features_Normalized * np.log2(self.ltp_features_Normalized + 1e-5))
                            else:
                                self.ltp_variance_Normalized = None
                                self.ltp_entropy_Normalized = None
                            
                        else:
                                
                            self.ltp_image = None
                            self.ltp_features = []
                            self.ltp_features_Normalized = []
                            
                            self.ltp_variance = None
                            self.ltp_entropy = None
                            self.ltp_variance_Normalized = None
                            self.ltp_entropy_Normalized = None
                            
                            
                        hog_switch_parameter = self.shared_data["hog_switch_parameter"]
                        print(f"hog_switch_parameter: {hog_switch_parameter}")
                        
                        if  hog_switch_parameter == True:
                            roi_cropped_image_HOG = cv2.resize(roi_cropped_image, (206, 178))
                            self.hog_bins, self.hog_bins_Normalized = self.compute_hog_features(roi_cropped_image_HOG)
                        
                        else:
                            self.hog_bins = []
                            self.hog_bins_Normalized = []
                            
                        file_name = os.path.splitext(os.path.basename(image_path))[0]

                        if self.shared_data["csv"] is not None:
                            csv_file_path_yolo = self.shared_data["csv"]
                        else:
                            csv_file_path_yolo = "Temp/CSV/Automatic.csv"
                        
                        
                        write_header = not os.path.exists(csv_file_path_yolo)
                        try:
                            with open(csv_file_path_yolo, mode="a", newline="") as csv_file:
                                fieldnames = ["Image Name", "Class", "percentage", "ROI Width", "ROI Height", "Crop Top-Left (X, Y)", "Crop Bottom-Right (X, Y)", "YOLO_Probability", "Tibial Width (Pixel)", "Tibial Width (mm)", "JSN Avg V.Distance (Pixel)", "JSN Avg V.Distance (mm)", "Medial_distance (Pixel)", "Central_distance (Pixel)", "Lateral_distance (Pixel)","Medial_distance (mm)", "Central_distance (mm)", "Lateral_distance (mm)", "Tibial_Medial_ratio", "Tibial_Central_ratio", "Tibial_Lateral_ratio", "JSN Area (Squared Pixel)", "JSN Area (Squared mm)", "Medial Area (Squared Pixel)", "Central Area (Squared Pixel)", "Lateral Area (Squared Pixel)","Medial Area (Squared mm)", "Central Area (Squared mm)", "Lateral Area (Squared mm)", "Medial Area (JSN Ratio)", "Central Area (JSN Ratio)", "Lateral Area (JSN Ratio)", "Medial Area Ratio TWPA (%)","Central Area Ratio TWPA (%)", "Lateral Area Ratio TWPA (%)","Mean", "Sigma", "Skewness", "Kurtosis"] + list(self.cooccurrence_properties.keys()) + \
                                            [f"LBP_{i}" for i in range(len(self.lbp_features))] + ["LBP_Variance", "LBP_Entropy"] + \
                                            [f"LBP_Normalized_{i}" for i in range(len(self.lbp_features_Normalized))] + ["LBP_Variance_Normalized", "LBP_Entropy_Normalized"] + \
                                            [f"LTP_{i}" for i in range(len(self.ltp_features))] + ["LTP_Variance", "LTP_Entropy"] + \
                                            [f"LTP_Normalized_{i}" for i in range(len(self.ltp_features_Normalized))] + ["LTP_Variance_Normalized", "LTP_Entropy_Normalized"] + \
                                            [f"HOG_{i}" for i in range(len(self.hog_bins))] + \
                                            [f"HOG_Normalized_{i}" for i in range(len(self.hog_bins_Normalized))]
                                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                                if write_header:
                                    writer.writeheader()
                                row_data = {
                                    "Image Name": file_name,
                                    "Class": self.folder_name,
                                    "percentage":self.percentage,
                                    "ROI Width":width_ROI,
                                    "ROI Height":height_ROI,
                                    "Crop Top-Left (X, Y)":f"({x1}, {y1})",
                                    "Crop Bottom-Right (X, Y)":f"({x2}, {y2})",
                                    "YOLO_Probability":f"{self.prob}",
                                    "Tibial Width (Pixel)": self.Tibial_width,
                                    "Tibial Width (mm)": self.Tibial_width/ self.length_ratio,
                                    "JSN Avg V.Distance (Pixel)": self.average_distance,
                                    "JSN Avg V.Distance (mm)": self.average_distance_mm,
                                    "Medial_distance (Pixel)":self.average_medial_distance,
                                    "Central_distance (Pixel)":self.average_central_distance,
                                    "Lateral_distance (Pixel)":self.average_lateral_distance,
                                    "Medial_distance (mm)":self.average_medial_distance_mm,
                                    "Central_distance (mm)":self.average_central_distance_mm,
                                    "Lateral_distance (mm)":self.average_lateral_distance_mm,
                                    "Tibial_Medial_ratio":self.Medial_ratio,
                                    "Tibial_Central_ratio":self.Central_ratio,
                                    "Tibial_Lateral_ratio":self.Lateral_ratio,
                                    "JSN Area (Squared Pixel)": self.JSN_Area_Total,
                                    "JSN Area (Squared mm)": self.JSN_Area_Total_Squared_mm,
                                    "Medial Area (Squared Pixel)":self.medial_area,
                                    "Central Area (Squared Pixel)":self.central_area,
                                    "Lateral Area (Squared Pixel)":self.lateral_area,                                                                                                
                                    "Medial Area (Squared mm)":self.medial_area_Squaredmm,
                                    "Central Area (Squared mm)":self.central_area_Squaredmm,
                                    "Lateral Area (Squared mm)":self.lateral_area_Squaredmm,
                                    "Medial Area (JSN Ratio)":self.medial_area_Ratio,
                                    "Central Area (JSN Ratio)":self.central_area_Ratio,
                                    "Lateral Area (JSN Ratio)":self.lateral_area_Ratio,
                                    "Medial Area Ratio TWPA (%)":self.medial_area_Ratio_TWPA,
                                    "Central Area Ratio TWPA (%)":self.central_area_Ratio_TWPA,
                                    "Lateral Area Ratio TWPA (%)":self.lateral_area_Ratio_TWPA,
                                    "Mean": self.intensity_mean,
                                    "Sigma": self.intensity_stddev,
                                    "Skewness": self.intensity_skewness,
                                    "Kurtosis": self.intensity_kurtosis,
                                    **self.cooccurrence_properties,
                                    **{f"LBP_{i}": self.lbp_features[i] for i in range(len(self.lbp_features))},
                                    "LBP_Variance": self.lbp_variance,
                                    "LBP_Entropy": self.lbp_entropy,
                                    **{f"LBP_Normalized_{i}": self.lbp_features_Normalized[i] for i in range(len(self.lbp_features_Normalized))},
                                    "LBP_Variance_Normalized": self.lbp_variance_Normalized,
                                    "LBP_Entropy_Normalized": self.lbp_entropy_Normalized,
                                    **{f"LTP_{i}": self.ltp_features[i] for i in range(len(self.ltp_features))},
                                    "LTP_Variance": self.ltp_variance,
                                    "LTP_Entropy": self.ltp_entropy,
                                    **{f"LTP_Normalized_{i}": self.ltp_features_Normalized[i] for i in range(len(self.ltp_features_Normalized))},
                                    "LTP_Variance_Normalized": self.ltp_variance_Normalized,
                                    "LTP_Entropy_Normalized": self.ltp_entropy_Normalized,
                                    **{f"HOG_{i}": self.hog_bins[i] for i in range(len(self.hog_bins))},
                                    **{f"HOG_Normalized_{i}": self.hog_bins_Normalized[i] for i in range(len(self.hog_bins_Normalized))}
                                    
                                }
                                writer.writerow(row_data)
                        except Exception as e:
                            print("\033[91mError writing to statistics_YOLO.csv:\033[0m", str(e))
#  ________________________________________________ HomeScreen _______________________________________________________
class HomeScreen(QMainWindow):

    variableChanged = pyqtSignal(int)
    variableChanged2 = pyqtSignal(int)
    variableChanged3 = pyqtSignal(int)
    VariableChanged_Combo_Conventional = pyqtSignal(int)
    VariableChanged_Combo_AI = pyqtSignal(int)
    VariableChanged_Combo_Fused = pyqtSignal(int)
    
    VariableChanged_Combo_Fused_to_Conventional = pyqtSignal(int)
    VariableChanged_Combo_Fused_to_AI = pyqtSignal(int)
    
    variableChanged_Equalize_Feature_Visualization = pyqtSignal(bool)
    variableChanged_Auto_mode = pyqtSignal(bool)
    variableChanged_Visualization = pyqtSignal(bool)
    
    shared_data = {"csv": None, "Histogram_properties_switch_parameter": True,"cooccurrence_properties_switch_parameter": True, "lbp_switch_parameter": True, "ltp_switch_parameter": True, "hog_switch_parameter": True}

    def __init__(self):
        super().__init__()
        screen_resolution = QDesktopWidget().screenGeometry()
        screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
        
        self.setFixedSize(screen_width,screen_height)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.isFullScreen = True
        # self.showFullScreen()

        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.screen_width = screen_width
        self.screen_height = screen_height
# _____________________________________________ Iitialization _______________________________________________
        self.full = False
        self.image = None
        self.file_path = None
        self.perform_intensity_normalization = True
        self.switch = True
        self.switch2 = False
        self.FirstSetVariable = 1
        self.SecondSetVariable = 1
        self.ThirdSetVariable = 1
        self.clasify_indicator = 0
        
        self.folder_name = "Temp"
        folder_path = Path(self.folder_name)
        if not folder_path.is_dir():
            folder_path.mkdir()
            
        self.folder_name2 = "Temp/CSV"
        folder_path2 = Path(self.folder_name2)
        if not folder_path2.is_dir():
            folder_path2.mkdir()
        
        self.folder_name3 = "Temp/Saved Figures"
        folder_path3 = Path(self.folder_name3)
        if not folder_path3.is_dir():
            folder_path3.mkdir()
# _____________________________________________ Classes instances _______________________________________________
        
        self.Conventional_CAD_Screen = Conventional_CAD_Screen(self)
        self.AI_Automated_CAD_Screen = AI_Automated_CAD_Screen(self)
        self.Fused_CVML_and_Ai_CAD_Screen = Fused_CVML_and_Ai_CAD_Screen(self, self.Conventional_CAD_Screen, self.AI_Automated_CAD_Screen)
        self.Feature_Extraction_and_Visualization_Screen = Feature_Extraction_and_Visualization_Screen(self, self.shared_data)
        self.HelpWindow = HelpWindow()
        self.SettingsWindow = SettingsWindow(self.shared_data)
        
        
        self.splash_screen = SplashScreen(self)
        self.splash_screen.variableChanged_Full.connect(self.handle_variable_changed)
        
        self.background_label = QLabel(self)
        self.pixmap = QPixmap("imgs/33blackSplashScreen.png")
        self.background_label.setPixmap(self.pixmap)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.background_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.background_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # Align to top-left
        self.background_label.setVisible(False)
        
        self.central_layout = QGridLayout()
        self.central_layout.setContentsMargins(0,0,0,0)
        self.central_widget = QWidget(self)
        self.central_widget.setLayout(self.central_layout)
        
        self.nav_layout_reduced = QVBoxLayout()
        self.nav_layout_reduced.setContentsMargins(0,0,0,0)
        
        self.nav_layout = QVBoxLayout()
        self.nav_layout.setContentsMargins(0,0,0,0)
        
    
    
    def on_combobox_changed_Conventional(self, index):
        index += 1
        self.VariableChanged_Combo_Conventional.emit(index)
        self.Conventional_CAD_Screen.on_combo_clear()
        
        print(f"index: {index}")
        pass
            
        
    def on_combobox_changed_AI(self, index):
        index += 1
        self.VariableChanged_Combo_AI.emit(index)
        self.AI_Automated_CAD_Screen.on_combo_clear()
        
        print(f"index: {index}")
        pass
            
    def on_combobox_changed_Fused(self, index):
        index += 1
        self.VariableChanged_Combo_Fused.emit(index)
        
        self.VariableChanged_Combo_Fused_to_Conventional.emit(index)
        self.VariableChanged_Combo_Fused_to_AI.emit(index)
        
        self.Fused_CVML_and_Ai_CAD_Screen.on_combo_clear()
        print(f"index: {index}")
        pass
            
    def handle_variable_changed(self, new_value):
        self.full = new_value
        # print("full out of thread", self.full)
        
        if self.full:
            self.show_main_content()
            self.background_label.setVisible(True)
            del self.splash_screen
            
            messages = ["Welcome to the KOA-CAD guide!", "Add a Radiographic knee image.", "Diagnose using Conventional methods,\n Artificial intelligence, and a Fusion of both.", "Here you can extract Features\n for your own dataset!"]
            positions = [QPoint(int(0.02604167 * self.screen_width), int(0.091145833 * self.screen_width)), QPoint(int(0.02604167 * self.screen_width), int(0.091145833 * self.screen_width)), QPoint(int(0.15104167 * self.screen_width), int(0.091145833 * self.screen_width)), QPoint(int(0.02604167 * self.screen_width), int(0.416667 * self.screen_width))]
            self.ex = GuideMessage(messages, positions)
        
    def show_main_content(self):
        # self.loading_timer.stop()
        # self.splash_screen.hide()       
        
        gradient_style = """
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));
        """
        self.setStyleSheet(gradient_style)

        font = QFont()
        font.setPointSize(int(0.00833 * self.screen_width))
        
        self.overlay = QWidget(self)
        self.overlay.setGeometry(self.rect())
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 30);")
        self.overlay_layout = QVBoxLayout(self.overlay)
        self.wait_label = QLabel('Please Wait...', self.overlay)
        self.wait_label.setStyleSheet("color: white; font-size: 75px;")
        self.wait_label.setAlignment(Qt.AlignCenter)
        self.overlay_layout.addWidget(self.wait_label)
        self.overlay.setVisible(False)
               
        self.load_img_button = QPushButton(self)
        self.load_img_button.setIcon(QIcon('imgs/new-document (1).png'))
        self.load_img_button.setFixedSize(int(0.109375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.load_img_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.load_img_button.setToolTip("Add  a  Radiographic  knee  image  to  KOA  CAD")
        self.load_img_button.setText(" Add an image")
        self.load_img_button.setFont(font)
        self.load_img_button.setStyleSheet("""
                                                                       QPushButton { 
                                                                       color: rgba(255, 255, 255, 255);
                                                                       border: none;
                                                                       background: transparent;
                                                                       border-radius: 50px;
                                                                       text-align: center;}
                                                                       
                                                                       QToolTip { 
                                                                            color: black; 
                                                                            background-color: white; 
                                                                            border: 1px solid black; 
                                                                        }
                                                                        
                                                                       """)
        
        
        self.classify_button = QPushButton(self)
        self.classify_button.setIcon(QIcon('imgs/classify.png'))
        self.classify_button.setFixedSize(int(0.109375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.classify_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.classify_button.setToolTip("Diagnose Knee OsteoArthits with  the  Whole  Methods!")
        self.classify_button.setText(" Diagnose")
        self.classify_button.setFont(font)
        self.classify_button.setStyleSheet("""
                                                                       QPushButton { 
                                                                       color: rgba(255, 255, 255, 255);
                                                                       border: none;
                                                                       background: transparent;
                                                                       border-radius: 50px;
                                                                       text-align: center;}
                                                                        QToolTip { 
                                                                            color: black; 
                                                                            background-color: white; 
                                                                            border: 1px solid black; 
                                                                        }
                                                                       """)
        
        
        
        self.load_img_button_layout = QVBoxLayout()
        self.load_img_button_layout.setContentsMargins(0,0,0,0)
        self.load_img_button_layout.addWidget(self.load_img_button, alignment= Qt.AlignLeft)
        self.load_img_button_layout_Widget = QWidget(self)
        self.load_img_button_layout_Widget.setFixedWidth(int(0.109375 * self.screen_width))
        self.load_img_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.load_img_button_layout_Widget.setLayout(self.load_img_button_layout)
        self.load_img_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;                                                                   
                                                                                     """)
        self.load_img_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
              
              
        self.classify_button_layout = QVBoxLayout()
        self.classify_button_layout.setContentsMargins(0,0,0,0)
        self.classify_button_layout.addWidget(self.classify_button, alignment= Qt.AlignLeft)
        self.classify_button_layout_Widget = QWidget(self)
        self.classify_button_layout_Widget.setFixedWidth(int(0.109375 * self.screen_width))
        self.classify_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.classify_button_layout_Widget.setLayout(self.classify_button_layout)
        self.classify_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;                                                                   
                                                                                     """)
        self.classify_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
        
        self.first_button_layout = QHBoxLayout()
        self.first_button_layout.setContentsMargins(0,0,0,0)
        self.first_button_layout.addWidget(self.load_img_button_layout_Widget, alignment= Qt.AlignLeft)
        self.first_button_layout.addWidget(self.classify_button_layout_Widget, alignment= Qt.AlignLeft)
        self.first_button_layout_Widget = QWidget(self)
        self.first_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.first_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.first_button_layout_Widget.setLayout(self.first_button_layout)
        self.first_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;                                                                   
                                                                                     """)
              
        self.load_img_button.clicked.connect(self.load_main_img)
        self.classify_button.clicked.connect(self.classify_all)
        
        
        
        self.Feature_Extraction_and_Visualization_button = AnimatedButton('imgs/Feature_Extraction_button.png')
        self.Feature_Extraction_and_Visualization_button.setMinimumSize(int(0.21875 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Feature_Extraction_and_Visualization_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Feature_Extraction_and_Visualization_button.setToolTip("Extract your own Features")
        self.Feature_Extraction_and_Visualization_button.setText(" Feature Extraction and Visualization")
        self.Feature_Extraction_and_Visualization_button.setFont(font)
        self.Feature_Extraction_and_Visualization_button.setStyleSheet("""
                                                                       QPushButton { 
                                                                       color: rgba(255, 255, 255, 255);
                                                                       border: none;
                                                                       background: transparent;
                                                                       border-radius: 50px;
                                                                       text-align: left;}
                                                                       QToolTip { 
                                                                            color: black; 
                                                                            background-color: white; 
                                                                            border: 1px solid black; 
                                                                        }
                                                                       """)
        
        self.Feature_Extraction_and_Visualization_button_layout = QHBoxLayout()
        self.Feature_Extraction_and_Visualization_button_layout.setContentsMargins(int(0.005208333 * self.screen_width),0,0,0)
        self.Feature_Extraction_and_Visualization_button_layout.addWidget(self.Feature_Extraction_and_Visualization_button, alignment= Qt.AlignLeft)
        self.Feature_Extraction_and_Visualization_button_layout_Widget = QWidget(self)
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setLayout(self.Feature_Extraction_and_Visualization_button_layout)
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Feature_Extraction_and_Visualization_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
              
        self.Feature_Extraction_and_Visualization_button.clicked.connect(self.show_screen1)

        self.Conventional_CAD_button = AnimatedButton('imgs/Conventional_CAD_button.png')
        self.Conventional_CAD_button.setMinimumSize(int(0.21875 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Conventional_CAD_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Conventional_CAD_button.setToolTip("Predict with the Conventional CVML Models")
        self.Conventional_CAD_button.setText(" Conventional CVML")
        self.Conventional_CAD_button.setFont(font)
        self.Conventional_CAD_button.setStyleSheet("""
                                                   QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}
                                                   QToolTip { 
                                                                            color: black; 
                                                                            background-color: white; 
                                                                            border: 1px solid black; 
                                                                        }
                                                   """)
        self.Conventional_CAD_button.clicked.connect(self.show_screen2)
        
        
        self.Conventional_CAD_button_layout = QHBoxLayout()
        self.Conventional_CAD_button_layout.setContentsMargins(int(0.00520833 * self.screen_width),0,0,0)
        self.Conventional_CAD_button_layout.addWidget(self.Conventional_CAD_button, alignment= Qt.AlignLeft)
        self.Conventional_CAD_button_layout_Widget = QWidget(self)
        self.Conventional_CAD_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.Conventional_CAD_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Conventional_CAD_button_layout_Widget.setLayout(self.Conventional_CAD_button_layout)
        self.Conventional_CAD_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Conventional_CAD_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
        font2 = QFont()
        font2.setPointSize(int(0.009895833 * self.screen_width))
        self.Conventional_RadioButtons_layout = QVBoxLayout()
        self.radio_buttons_Conventional = []
        self.radio_buttons_names_Conventional = ["Normal Vs OsteoArthritis", "Normal Vs Mild Vs Severe", "Kellgren-Lawrence 5-Classes","Probability voting 5-Classes", "Apply all Models"]

        for i in range(5):
            radio_button = QRadioButton(self.radio_buttons_names_Conventional[i])
            radio_button.setStyleSheet("""
                QRadioButton {
                    color: darkgray;
                }
                QRadioButton:hover {
                    color: rgba(64, 255, 64, 150);
                }
                QRadioButton::indicator {
                    width: 10px;
                    height: 10px;
                    border-radius: 5px;
                    border: 1px solid rgba(64, 164, 64, 150);
                    background-color: white;
                }
                QRadioButton::indicator:hover {
                    background-color: rgba(64, 255, 64, 150);
                }
                QRadioButton::indicator:checked {
                    background-color: rgba(64, 164, 64, 150);
                    border: 1px solid rgba(64, 255, 64, 255);
                }
                QRadioButton::indicator:checked:hover {
                    background-color: rgba(64, 255, 64, 150);
                }
            """)
        
            radio_button.setFont(font2)
            radio_button.clicked.connect(lambda _, i=i: self.on_radio_button_Conventional_clicked(i))
            self.Conventional_RadioButtons_layout.addWidget(radio_button)
            self.radio_buttons_Conventional.append(radio_button)

            if i == 0:
                radio_button.setChecked(True)


        self.Ai_Automated_CAD_button = AnimatedButton('imgs/AI_Automated_CAD_button.png')
        self.Ai_Automated_CAD_button.setMinimumSize(int(0.21875 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Ai_Automated_CAD_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Ai_Automated_CAD_button.setText(" AI Automated")
        self.Ai_Automated_CAD_button.setToolTip("Predict with the AI Models")
        
        self.Ai_Automated_CAD_button.setFont(font)
        self.Ai_Automated_CAD_button.setStyleSheet("""QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}
                                                   QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                }
                                                                        """)
        self.Ai_Automated_CAD_button.clicked.connect(self.show_screen3)
        
        self.Ai_Automated_CAD_button_layout = QHBoxLayout()
        self.Ai_Automated_CAD_button_layout.setContentsMargins(10,0,0,0)
        self.Ai_Automated_CAD_button_layout.addWidget(self.Ai_Automated_CAD_button, alignment= Qt.AlignLeft)
        self.Ai_Automated_CAD_button_layout_Widget = QWidget(self)
        self.Ai_Automated_CAD_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.Ai_Automated_CAD_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Ai_Automated_CAD_button_layout_Widget.setLayout(self.Ai_Automated_CAD_button_layout)
        self.Ai_Automated_CAD_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Ai_Automated_CAD_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
        
        
        
        
        
        
        
        self.Fused_CVML_and_Ai_button = AnimatedButton('imgs/Fusion.png')
        self.Fused_CVML_and_Ai_button.setMinimumSize(int(0.21875 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Fused_CVML_and_Ai_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Fused_CVML_and_Ai_button.setText(" Fusion CVML-AI")
        self.Fused_CVML_and_Ai_button.setToolTip("Predict with the Fused Conventional CVML and AI Model")
        
        self.Fused_CVML_and_Ai_button.setFont(font)
        self.Fused_CVML_and_Ai_button.setStyleSheet("""
                                                    QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                                    QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}
                                                    """)
        
        self.Fused_CVML_and_Ai_button.clicked.connect(self.show_screen4)
        
        self.Fused_CVML_and_Ai_button_layout = QHBoxLayout()
        self.Fused_CVML_and_Ai_button_layout.setContentsMargins(int(0.00520833 * self.screen_width),0,0,0)
        self.Fused_CVML_and_Ai_button_layout.addWidget(self.Fused_CVML_and_Ai_button, alignment= Qt.AlignLeft)
        self.Fused_CVML_and_Ai_button_layout_Widget = QWidget(self)
        self.Fused_CVML_and_Ai_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.Fused_CVML_and_Ai_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Fused_CVML_and_Ai_button_layout_Widget.setLayout(self.Fused_CVML_and_Ai_button_layout)
        self.Fused_CVML_and_Ai_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Fused_CVML_and_Ai_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
              
              
              
              
              
              
              
              

        self.Reset_button = AnimatedButton('imgs/trash.svg')
        self.Reset_button.setMinimumSize(int(0.21875 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Reset_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Reset_button.setToolTip("Clear All")
        self.Reset_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Reset_button.clicked.connect(self.Clear_All)
        self.Reset_button.setText(" Clear")
        self.Reset_button.setFont(font)
        self.Reset_button.setStyleSheet("""
                                        QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}
                                        QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                        """)

        self.Reset_button_layout = QHBoxLayout()
        self.Reset_button_layout.setContentsMargins(int(0.00520833 * self.screen_width),0,0,0)
        self.Reset_button_layout.addWidget(self.Reset_button, alignment= Qt.AlignLeft)
        self.Reset_button_layout_Widget = QWidget(self)
        self.Reset_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.Reset_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Reset_button_layout_Widget.setLayout(self.Reset_button_layout)
        self.Reset_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Reset_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
        
        
          
        
        self.settings_button = AnimatedButton('imgs/settings.svg')
        self.settings_button.setMinimumSize(int(0.21875 * self.screen_width),int(0.0234375 * self.screen_width))
        self.settings_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.settings_button.setToolTip("Select your desirable exported Features")
        self.settings_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.settings_button.clicked.connect(self.Settings)
        self.settings_button.setText(" Settings")
        self.settings_button.setFont(font)
        self.settings_button.setStyleSheet("""
                                           QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                                    
                                           QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}
                                           """)

        self.settings_button_layout = QHBoxLayout()
        self.settings_button_layout.setContentsMargins(int(0.00520833 * self.screen_width),0,0,0)
        self.settings_button_layout.addWidget(self.settings_button, alignment= Qt.AlignLeft)
        self.settings_button_layout_Widget = QWidget(self)
        self.settings_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.settings_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.settings_button_layout_Widget.setLayout(self.settings_button_layout)
        self.settings_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.settings_button_layout_Widget.setStyleSheet("""
                                                                                     
                                                                                     
                                                                                     
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
        
                   
        self.help_button = AnimatedButton('imgs/help.svg')
        self.help_button.setMinimumSize(int(0.21875 * self.screen_width),int(0.0234375 * self.screen_width))
        self.help_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.help_button.setToolTip("Take a tour about KOA CAD")
        self.help_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.help_button.setText(" Help")
        self.help_button.setFont(font)
        self.help_button.setStyleSheet("""
                                       QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                        QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}
                                        """)
        
        self.help_button.clicked.connect(self.help)
        
        self.help_button_layout = QHBoxLayout()
        self.help_button_layout.setContentsMargins(int(0.00520833 * self.screen_width),0,0,0)
        self.help_button_layout.addWidget(self.help_button, alignment= Qt.AlignLeft)
        self.help_button_layout_Widget = QWidget(self)
        self.help_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.help_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.help_button_layout_Widget.setLayout(self.help_button_layout)
        self.help_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.help_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
        
       
        
        
        self.exit_button = AnimatedButton('imgs/log-out.svg')
        self.exit_button.setMinimumSize(int(0.21875 * self.screen_width),int(0.0234375 * self.screen_width))
        self.exit_button.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.exit_button.setToolTip("You're about to leave KOA CAD!")
        self.exit_button.setStyleSheet("""
                                       QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                                    
                                       QPushButton { border: none; background: transparent; border-radius: 50px; }
                                       """)
        
        self.exit_button.clicked.connect(self.closeEvent)
        self.exit_button.setText(" Exit")
        self.exit_button.setFont(font)
        self.exit_button.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
        self.exit_button_layout = QHBoxLayout()
        self.exit_button_layout.setContentsMargins(int(0.00520833 * self.screen_width),0,0,0)
        self.exit_button_layout.addWidget(self.exit_button, alignment= Qt.AlignLeft)
        self.exit_button_layout_Widget = QWidget(self)
        self.exit_button_layout_Widget.setFixedWidth(int(0.21875 * self.screen_width))
        self.exit_button_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.exit_button_layout_Widget.setLayout(self.exit_button_layout)
        
        self.exit_button_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.exit_button_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(250, 0, 0, 150);}
                                                                                     """)




        self.Ai_RadioButtons_layout = QVBoxLayout()
        self.radio_buttons_Ai = []
        self.radio_buttons_names_Ai = ["Normal Vs OsteoArthritis", "Normal Vs Mild Vs Severe", "Kellgren-Lawrence 5-Classes","Probability voting 5-Classes", "Apply all Models"]
        
        for i in range(5):
            radio_button = QRadioButton(self.radio_buttons_names_Ai[i])
            radio_button.setStyleSheet("""
                QRadioButton {
                    color: darkgray; 
                }
                QRadioButton:hover {
                    color: rgba(64, 255, 64, 150);
                }
                QRadioButton::indicator {
                    width: 10px;
                    height: 10px;
                    border-radius: 5px;
                    border: 1px solid rgba(64, 164, 64, 150);
                    background-color: white;
                }
                QRadioButton::indicator:hover {
                    background-color: rgba(64, 255, 64, 150);
                }
                QRadioButton::indicator:checked {
                    background-color: rgba(64, 164, 64, 150);
                    border: 1px solid rgba(64, 255, 64, 255);
                }
                QRadioButton::indicator:checked:hover {
                    background-color: rgba(64, 255, 64, 150);
                }
            """)
            radio_button.setFont(font2)
            radio_button.clicked.connect(lambda _, i=i: self.on_radio_button_AI_clicked(i))
            self.Ai_RadioButtons_layout.addWidget(radio_button)
            self.radio_buttons_Ai.append(radio_button)

            if i == 0:
                radio_button.setChecked(True)
                
        
        font3 = QFont()
        font3.setPointSize(int(0.009895833 * self.screen_width))
        
        self.Fused_RadioButtons_layout = QVBoxLayout()
        self.radio_buttons_Fused = []
        self.radio_buttons_names_Fused = ["Weighted Average Fusion", "Back Propagation NN Fusion"]
        
        for i in range(2):
            radio_button = QRadioButton(self.radio_buttons_names_Fused[i])
            radio_button.setStyleSheet("""
                QRadioButton {
                    color: darkgray; 
                }
                QRadioButton:hover {
                    color: rgba(64, 255, 64, 150);
                }
                QRadioButton::indicator {
                    width: 10px;
                    height: 10px;
                    border-radius: 5px;
                    border: 1px solid rgba(64, 164, 64, 150);
                    background-color: white;
                }
                QRadioButton::indicator:hover {
                    background-color: rgba(64, 255, 64, 150);
                }
                QRadioButton::indicator:checked {
                    background-color: rgba(64, 164, 64, 150);
                    border: 1px solid rgba(64, 255, 64, 255);
                }
                QRadioButton::indicator:checked:hover {
                    background-color: rgba(64, 255, 64, 150);
                }
            """)
            radio_button.setFont(font3)
            radio_button.clicked.connect(lambda _, i=i: self.on_radio_button_Fused_clicked(i))
            self.Fused_RadioButtons_layout.addWidget(radio_button)
            self.radio_buttons_Fused.append(radio_button)

            if i == 0:
                radio_button.setChecked(True)
                

        spacer_nav_reduced1 = QSpacerItem(0, int(0.0220833 * self.screen_width), QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_nav_reduced3 = QSpacerItem(0, int(0.2604167 * self.screen_width), QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_nav_reduced4 = QSpacerItem(0, int(0.11979167 * self.screen_width), QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_nav_reduced5 = QSpacerItem(0, int(0.11979167 * self.screen_width), QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_nav_reduced6 = QSpacerItem(0, int(0.11979167 * self.screen_width), QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.spacer_nav_reduced3_layout = QVBoxLayout()
        self.spacer_nav_reduced3_layout.setContentsMargins(0,0,0,0)
        self.spacer_nav_reduced3_layout.addSpacerItem(spacer_nav_reduced3)
        self.spacer_nav_reduced3_layout_Widget = QWidget(self)
        self.spacer_nav_reduced3_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.spacer_nav_reduced3_layout_Widget.setFixedHeight(int(0.2604167 * self.screen_width))
        self.spacer_nav_reduced3_layout_Widget.setLayout(self.spacer_nav_reduced3_layout)
        self.spacer_nav_reduced3_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        

        self.spacer_nav_reduced4_layout = QVBoxLayout()
        self.spacer_nav_reduced4_layout.setContentsMargins(0,0,0,0)
        self.spacer_nav_reduced4_layout.addSpacerItem(spacer_nav_reduced4)
        self.spacer_nav_reduced4_layout_Widget = QWidget(self)
        self.spacer_nav_reduced4_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.spacer_nav_reduced4_layout_Widget.setFixedHeight(int(0.2604167 * self.screen_width))
        self.spacer_nav_reduced4_layout_Widget.setLayout(self.spacer_nav_reduced4_layout)
        self.spacer_nav_reduced4_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        
        self.spacer_nav_reduced5_layout = QVBoxLayout()
        self.spacer_nav_reduced5_layout.setContentsMargins(0,0,0,0)
        self.spacer_nav_reduced5_layout.addSpacerItem(spacer_nav_reduced5)
        self.spacer_nav_reduced5_layout_Widget = QWidget(self)
        self.spacer_nav_reduced5_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.spacer_nav_reduced5_layout_Widget.setFixedHeight(int(0.2604167 * self.screen_width))
        self.spacer_nav_reduced5_layout_Widget.setLayout(self.spacer_nav_reduced5_layout)
        self.spacer_nav_reduced5_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        
        self.spacer_nav_reduced6_layout = QVBoxLayout()
        self.spacer_nav_reduced6_layout.setContentsMargins(0,0,0,0)
        self.spacer_nav_reduced6_layout.addSpacerItem(spacer_nav_reduced6)
        self.spacer_nav_reduced6_layout_Widget = QWidget(self)
        self.spacer_nav_reduced6_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.spacer_nav_reduced6_layout_Widget.setFixedHeight(int(0.2604167 * self.screen_width))
        self.spacer_nav_reduced6_layout_Widget.setLayout(self.spacer_nav_reduced6_layout)
        self.spacer_nav_reduced6_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        
        self.load_img_button_reduced = AnimatedButton('imgs/new-document (1).png')
        self.load_img_button_reduced.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.load_img_button_reduced.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.load_img_button_reduced.setToolTip("Add  a  Radiographic  knee  image  to  KOA  CAD")
        self.load_img_button_reduced.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.load_img_button_reduced.clicked.connect(self.load_main_img)

        self.load_img_button_reduced_layout = QHBoxLayout()
        self.load_img_button_reduced_layout.setContentsMargins(0,0,0,0)
        self.load_img_button_reduced_layout.addWidget(self.load_img_button_reduced, alignment= Qt.AlignCenter)
        self.load_img_button_reduced_layout_Widget = QWidget(self)
        self.load_img_button_reduced_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.load_img_button_reduced_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.load_img_button_reduced_layout_Widget.setLayout(self.load_img_button_reduced_layout)
        self.load_img_button_reduced_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.load_img_button_reduced_layout_Widget.setStyleSheet("""
                                                                                     
                                                                                     
                                                                                     
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)




        self.Feature_Extraction_and_Visualization_button_reduced = AnimatedButton('imgs/Feature_Extraction_button.png')
        self.Feature_Extraction_and_Visualization_button_reduced.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Feature_Extraction_and_Visualization_button_reduced.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Feature_Extraction_and_Visualization_button_reduced.setToolTip("Extract your own Features")
        self.Feature_Extraction_and_Visualization_button_reduced.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Feature_Extraction_and_Visualization_button_reduced.clicked.connect(self.show_screen1)

        self.Feature_Extraction_and_Visualization_button_reduced_layout = QHBoxLayout()
        self.Feature_Extraction_and_Visualization_button_reduced_layout.setContentsMargins(0,0,0,0)
        self.Feature_Extraction_and_Visualization_button_reduced_layout.addWidget(self.Feature_Extraction_and_Visualization_button_reduced, alignment= Qt.AlignCenter)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget = QWidget(self)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setLayout(self.Feature_Extraction_and_Visualization_button_reduced_layout)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget.setStyleSheet("""
                                                                                     
                                                                                     
                                                                                     
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)




        self.Conventional_CAD_button_reduced = AnimatedButton('imgs/Conventional_CAD_button.png')
        self.Conventional_CAD_button_reduced.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Conventional_CAD_button_reduced.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Conventional_CAD_button_reduced.setToolTip("Predict with the Conventional CVML Models")
        self.Conventional_CAD_button_reduced.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Conventional_CAD_button_reduced.clicked.connect(self.show_screen2)
        
        self.Conventional_CAD_button_reduced_layout = QHBoxLayout()
        self.Conventional_CAD_button_reduced_layout.setContentsMargins(0,0,0,0)
        self.Conventional_CAD_button_reduced_layout.addWidget(self.Conventional_CAD_button_reduced, alignment= Qt.AlignCenter)
        self.Conventional_CAD_button_reduced_layout_Widget = QWidget(self)
        self.Conventional_CAD_button_reduced_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.Conventional_CAD_button_reduced_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Conventional_CAD_button_reduced_layout_Widget.setLayout(self.Conventional_CAD_button_reduced_layout)
        self.Conventional_CAD_button_reduced_layout_Widget.setStyleSheet("""
                                                                            background-color: transparent;
                                                                            border: none;
                                                                                     """)
        self.Conventional_CAD_button_reduced_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
        self.Ai_Automated_CAD_button_reduced = AnimatedButton('imgs/Ai_Automated_CAD_button.png')
        self.Ai_Automated_CAD_button_reduced.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Ai_Automated_CAD_button_reduced.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Ai_Automated_CAD_button_reduced.setToolTip("Predict with the AI Models")
        self.Ai_Automated_CAD_button_reduced.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Ai_Automated_CAD_button_reduced.clicked.connect(self.show_screen3)

        self.Ai_Automated_CAD_button_reduced_layout = QHBoxLayout()
        self.Ai_Automated_CAD_button_reduced_layout.setContentsMargins(0,0,0,0)
        self.Ai_Automated_CAD_button_reduced_layout.addWidget(self.Ai_Automated_CAD_button_reduced, alignment= Qt.AlignCenter)
        self.Ai_Automated_CAD_button_reduced_layout_Widget = QWidget(self)
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setLayout(self.Ai_Automated_CAD_button_reduced_layout)
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.Ai_Automated_CAD_button_reduced_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
        
        
        
        self.Fused_CVML_and_Ai_button_reduced = AnimatedButton('imgs/Fusion.png')
        self.Fused_CVML_and_Ai_button_reduced.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Fused_CVML_and_Ai_button_reduced.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Fused_CVML_and_Ai_button_reduced.setToolTip("Predict with the Fused Conventional CVML and AI Model")
        self.Fused_CVML_and_Ai_button_reduced.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Fused_CVML_and_Ai_button_reduced.clicked.connect(self.show_screen4)

        self.Fused_CVML_and_Ai_button_reduced_layout = QHBoxLayout()
        self.Fused_CVML_and_Ai_button_reduced_layout.setContentsMargins(0,0,0,0)
        self.Fused_CVML_and_Ai_button_reduced_layout.addWidget(self.Fused_CVML_and_Ai_button_reduced, alignment= Qt.AlignCenter)
        self.Fused_CVML_and_Ai_button_reduced_layout_Widget = QWidget(self)
        self.Fused_CVML_and_Ai_button_reduced_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.Fused_CVML_and_Ai_button_reduced_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Fused_CVML_and_Ai_button_reduced_layout_Widget.setLayout(self.Fused_CVML_and_Ai_button_reduced_layout)
        self.Fused_CVML_and_Ai_button_reduced_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.Fused_CVML_and_Ai_button_reduced_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
        
        
        self.Reset_button1 = AnimatedButton('imgs/trash.svg')
        self.Reset_button1.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Reset_button1.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.Reset_button1.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.Reset_button1.setToolTip("Clear All")
        self.Reset_button1.clicked.connect(self.Clear_All)
        
        self.Reset_button1_layout = QHBoxLayout()
        self.Reset_button1_layout.setContentsMargins(int(0.00520833 * self.screen_width),0,0,0)
        self.Reset_button1_layout.addWidget(self.Reset_button1, alignment= Qt.AlignLeft)
        self.Reset_button1_layout_Widget = QWidget(self)
        self.Reset_button1_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.Reset_button1_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.Reset_button1_layout_Widget.setLayout(self.Reset_button1_layout)
        self.Reset_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.Reset_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
        self.settings_button1 = AnimatedButton('imgs/settings.svg')
        self.settings_button1.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.settings_button1.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.settings_button1.setToolTip("Export your desirable Features")
        self.settings_button1.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.settings_button1.clicked.connect(self.Settings)
        
        self.settings_button1_layout = QHBoxLayout()
        self.settings_button1_layout.setContentsMargins(int(0.00520833 * self.screen_width),0,0,0)
        self.settings_button1_layout.addWidget(self.settings_button1, alignment= Qt.AlignLeft)
        self.settings_button1_layout_Widget = QWidget(self)
        self.settings_button1_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.settings_button1_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.settings_button1_layout_Widget.setLayout(self.settings_button1_layout)
        self.settings_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.settings_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
              
        self.help_button1 = AnimatedButton('imgs/help.svg')
        self.help_button1.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.help_button1.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.help_button1.setToolTip("Take a tour about the KOA CAD")
        self.help_button1.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.help_button1.clicked.connect(self.help)
        
        self.help_button1_layout = QHBoxLayout()
        self.help_button1_layout.setContentsMargins(10,0,0,0)
        self.help_button1_layout.addWidget(self.help_button1, alignment= Qt.AlignLeft)
        self.help_button1_layout_Widget = QWidget(self)
        self.help_button1_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.help_button1_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.help_button1_layout_Widget.setLayout(self.help_button1_layout)
        self.help_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.help_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(64, 240, 64, 150);}
                                                                                     """)
        
        self.exit_button1 = AnimatedButton('imgs/log-out.svg')
        self.exit_button1.setMinimumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.exit_button1.setMaximumSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width))
        self.exit_button1.setToolTip("You're about to leave KOA CAD!")
        self.exit_button1.setIconSize(QSize(int(0.0234375 * self.screen_width),int(0.0234375 * self.screen_width)))
        self.exit_button1.clicked.connect(self.closeEvent)
        
        self.exit_button1_layout = QHBoxLayout()
        self.exit_button1_layout.setContentsMargins(10,0,0,0)
        self.exit_button1_layout.addWidget(self.exit_button1, alignment= Qt.AlignLeft)
        self.exit_button1_layout_Widget = QWidget(self)
        self.exit_button1_layout_Widget.setFixedWidth(int(0.03645833 * self.screen_width))
        self.exit_button1_layout_Widget.setFixedHeight(int(0.03645833 * self.screen_width))
        self.exit_button1_layout_Widget.setLayout(self.exit_button1_layout)
        self.exit_button1_layout_Widget.setStyleSheet("background-color: transparent; border: none;")
        self.exit_button1_layout_Widget.setStyleSheet("""
                                                                                     QWidget:hover {
                                                                                                    background-color: rgba(250, 0, 0, 150);}
                                                                                     """)
        self.nav_layout_reduced.setSpacing(0)

        self.nav_layout_reduced.addSpacerItem(spacer_nav_reduced1)
        self.nav_layout_reduced.addWidget(self.load_img_button_reduced_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout_reduced.addWidget(self.Conventional_CAD_button_reduced_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout_reduced.addWidget(self.spacer_nav_reduced3_layout_Widget)
        self.nav_layout_reduced.addWidget(self.Ai_Automated_CAD_button_reduced_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout_reduced.addWidget(self.spacer_nav_reduced6_layout_Widget)
        self.nav_layout_reduced.addWidget(self.Fused_CVML_and_Ai_button_reduced_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout_reduced.addWidget(self.spacer_nav_reduced5_layout_Widget)
        self.nav_layout_reduced.addWidget(self.Feature_Extraction_and_Visualization_button_reduced_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout_reduced.addWidget(self.spacer_nav_reduced4_layout_Widget)
        self.nav_layout_reduced.addWidget(self.Reset_button1_layout_Widget)
        self.nav_layout_reduced.addWidget(self.settings_button1_layout_Widget)
        self.nav_layout_reduced.addWidget(self.help_button1_layout_Widget)
        self.nav_layout_reduced.addWidget(self.exit_button1_layout_Widget)
        
        self.Conventional_RadioButtons_layout_widget = QWidget(self)
        self.Conventional_RadioButtons_layout_widget.setFixedHeight(int(0.104167 * self.screen_width))
        self.Conventional_RadioButtons_layout_widget.setLayout(self.Conventional_RadioButtons_layout)
        
        self.Ai_RadioButtons_layout_widget = QWidget(self)
        self.Ai_RadioButtons_layout_widget.setFixedHeight(int(0.104167 * self.screen_width))
        self.Ai_RadioButtons_layout_widget.setLayout(self.Ai_RadioButtons_layout)
        
        self.Fused_RadioButtons_layout_widget = QWidget(self)
        self.Fused_RadioButtons_layout_widget.setFixedHeight(int(0.104167 * self.screen_width))
        self.Fused_RadioButtons_layout_widget.setLayout(self.Fused_RadioButtons_layout)
        

        spacer_nav1 = QSpacerItem(0, int(0.020833 * self.screen_width), QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacer_nav3_H = QSpacerItem(int(0.02604167 * self.screen_width), 0, QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer_nav4_H = QSpacerItem(int(0.02604167 * self.screen_width), 0, QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer_nav5_H = QSpacerItem(int(0.015625 * self.screen_width), 0, QSizePolicy.Fixed, QSizePolicy.Fixed)

        # self.start_Extraction
        self.trigger_Extraction_button = QPushButton("Start", self)
        self.trigger_Extraction_button.setToolTip("Hit  start  after  choosing  your  preferable  optons  to  start  Feature  extraction  and  visualization  Mode")
        font = QFont()
        font.setFamily("helvetica")
        font.setPointSize(int(0.0078125 * self.screen_width))
        self.trigger_Extraction_button.setFont(font)
        self.trigger_Extraction_button.setMinimumHeight(int(0.0145833 * self.screen_width))
        self.trigger_Extraction_button.setMaximumHeight(int(0.0145833 * self.screen_width))

        # Calculate the radius to ensure a rounded rectangle
        border_radius = int(0.0145833 * self.screen_width) // 2

        self.trigger_Extraction_button.setStyleSheet(f"""
            QPushButton {{
                color: white; 
                background: rgba(64, 164, 64, 150);
                border-radius: {border_radius}px;
            }}
            QPushButton:hover {{
                background-color: rgba(64, 164, 64, 200); 
                color: white; 
                border-radius: {border_radius}px;
            }}
        """)

        self.trigger_Extraction_button.clicked.connect(self.start_Extraction)
        # background: rgba(34, 100, 34, 200);


        self.Equalize_button = QPushButton(self)
        self.Equalize_button.setIcon(QIcon('imgs/check-square.svg'))
        self.Equalize_button.setMinimumSize(int(0.018229 * self.screen_width),int(0.018229 * self.screen_width))
        self.Equalize_button.setMaximumSize(int(0.018229 * self.screen_width),int(0.018229 * self.screen_width))
        self.Equalize_button.setIconSize(QSize(int(0.018229 * self.screen_width),int(0.018229 * self.screen_width)))
        self.Equalize_button.setToolTip("Toggle intensity equalization effect")
        self.Equalize_button.setStyleSheet("""
                                           QPushButton { border: none; background: transparent; border-radius: 50px; }
                                           QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                           """)
        self.Equalize_button.clicked.connect(self.toggle_intensity_normalization_Equalize)
        
        self.Equalization_label = QLabel("Apply intensity Equalization", self)
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(int(0.0078125 * self.screen_width))
        font.setBold(True)
        self.Equalization_label.setFont(font)
        self.Equalization_label.setMinimumSize(int(0.15625 * self.screen_width), int(0.018229 * self.screen_width))
        self.Equalization_label.setStyleSheet("QLabel { color: darkgray; border: none; background: transparent; border-radius: 50px; }")
        
        self.Auto_F_Extractor_Single = QPushButton(self)
        self.Auto_F_Extractor_Single.setIcon(QIcon('imgs/enable-mode.png'))
        self.Auto_F_Extractor_Single.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.Auto_F_Extractor_Single.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.Auto_F_Extractor_Single.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
        self.Auto_F_Extractor_Single.setToolTip("Extract Features from a Single Radiographic Knee Image Only")
        self.Auto_F_Extractor_Single.setStyleSheet("""
                                                   QPushButton { border: none; background: transparent; border-radius: 50px; }
                                                   QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                                   """)
        self.Auto_F_Extractor_Single.clicked.connect(self.switch_mode_Auto_Mode)
        
        
        self.Auto_F_Extractor_Dataset = QPushButton(self)
        self.Auto_F_Extractor_Dataset.setIcon(QIcon('imgs/disable-mode.svg'))
        self.Auto_F_Extractor_Dataset.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.Auto_F_Extractor_Dataset.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.Auto_F_Extractor_Dataset.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
        self.Auto_F_Extractor_Dataset.setToolTip("Extract Features from a Folder Containing Radiographic Knee Images")
        self.Auto_F_Extractor_Dataset.setStyleSheet("""
                                                    QPushButton { border: none; background: transparent; border-radius: 50px; }
                                                    QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                                    """)
        self.Auto_F_Extractor_Dataset.clicked.connect(self.switch_mode_Auto_Mode)
               
                
        self.Windows_button_button44 = QPushButton(self)
        self.Windows_button_button44.setIcon(QIcon('imgs/eye-off.svg'))
        self.Windows_button_button44.setFixedSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
        self.Windows_button_button44.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
        self.Windows_button_button44.setToolTip("Toggle Visualizations")
        self.Windows_button_button44.setStyleSheet("""
                                                   QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}
                                                   QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                                   """)
        self.Windows_button_button44.clicked.connect(self.Windows_button_Toggling_Visualizations)
        
        self.Save_button44 = QPushButton(self)
        self.Save_button44.setIcon(QIcon('imgs/save.svg'))
        self.Save_button44.setFixedSize(int(0.020033 * self.screen_width),int(0.019033 * self.screen_width))
        self.Save_button44.setIconSize(QSize(int(0.020033 * self.screen_width),int(0.019033 * self.screen_width)))
        self.Save_button44.setToolTip("Save  All  Figures  and  Visualizations  from  Feature  Extraction  and  Visualization  Mode")
        
        self.Save_button44.setStyleSheet("""
                                        QPushButton { border: none; background: transparent; border-radius: 50px; }
                                        QToolTip {
                                                    color: black; 
                                                    background-color: white; 
                                                    border: 1px solid black; 
                                                    }
                                         """)
        self.Save_button44.clicked.connect(self.Feature_Extraction_and_Visualization_Screen.Save_ALL)
              
        self.Single_img_label = QLabel("Single image Mode", self)
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(int(0.0078125 * self.screen_width))
        font.setBold(True)
        self.Single_img_label.setFont(font)
        self.Single_img_label.setMinimumSize(int(0.15625 * self.screen_width),int(0.018229 * self.screen_width))
        self.Single_img_label.setStyleSheet("QLabel { text-align:right; color:darkgray; border: none; background: transparent; border-radius: 50px; }")
        self.Single_img_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)


        self.Auto_mode_label = QLabel("Dataset automatic Mode", self)
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(int(0.0078125 * self.screen_width))
        font.setBold(True)
        self.Auto_mode_label.setFont(font)
        self.Auto_mode_label.setMinimumSize(int(0.15625 * self.screen_width),int(0.018229 * self.screen_width))
        self.Auto_mode_label.setStyleSheet("QLabel { color: darkgray; border: none; background: transparent; border-radius: 50px; }")
        self.Auto_mode_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        self.toggle_visualizations_label = QLabel("Show Visualizations", self)
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(int(0.0078125 * self.screen_width))
        font.setBold(True)
        self.toggle_visualizations_label.setFont(font)
        self.toggle_visualizations_label.setFixedSize(int(0.15625 * self.screen_width),int(0.018229 * self.screen_width))
        self.toggle_visualizations_label.setStyleSheet("QLabel { color: darkgray; border: none; background: transparent; border-radius: 50px; }")
        self.toggle_visualizations_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        

        self.Save_ALL_label = QLabel("Save All Figures", self)
        font = QFont("Segoe UI")
        font.setFamily("Arial")
        font.setPointSize(int(0.0078125 * self.screen_width))
        font.setBold(True)
        self.Save_ALL_label.setFont(font)
        self.Save_ALL_label.setMinimumSize(int(0.15625 * self.screen_width),int(0.018229 * self.screen_width))
        self.Save_ALL_label.setStyleSheet("QLabel { color: darkgray; border: none; background: transparent; border-radius: 50px; }")
        self.Save_ALL_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.Save_ALL_label.setToolTip("Save  All  Figures  and  Visualizations  from  Feature  Extraction  and  Visualization  Mode")
        
        self.nav_1st_Horizontal_Layout = QVBoxLayout()
        self.nav_1st_Horizontal_Layout.setContentsMargins(int(0.018229 * self.screen_width), int(0.018229 * self.screen_width), int(0.018229 * self.screen_width), int(0.018229 * self.screen_width))
        self.nav_1st_Horizontal_Layout.setAlignment(Qt.AlignCenter)
        
        self.nav_1st_Horizontal_Widget = QWidget()
        self.nav_1st_Horizontal_Widget.setLayout(self.nav_1st_Horizontal_Layout)
        self.nav_1st_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")


        self.nav_1st_Horizontal_Layout_4 = QHBoxLayout()
        self.nav_1st_Horizontal_Layout_4.setContentsMargins(0,0,0,0)
        self.nav_1st_Horizontal_Widget_4 = QWidget()
        self.nav_1st_Horizontal_Widget_4.setLayout(self.nav_1st_Horizontal_Layout_4)
        self.nav_1st_Horizontal_Widget_4.setFixedHeight(int(0.020833 * self.screen_width))
        self.nav_1st_Horizontal_Widget_4.setStyleSheet("background-color: transparent; border: none;")
        self.nav_1st_Horizontal_Layout.addWidget(self.nav_1st_Horizontal_Widget_4)
        self.nav_1st_Horizontal_Layout_4.addWidget(self.Equalization_label)
        self.nav_1st_Horizontal_Layout_4.addWidget(self.Equalize_button)


        self.nav_1st_Horizontal_Layout_1 = QHBoxLayout()
        self.nav_1st_Horizontal_Layout_1.setContentsMargins(0,0,0,0)
        
        self.nav_1st_Horizontal_Widget_1 = QWidget()
        self.nav_1st_Horizontal_Widget_1.setLayout(self.nav_1st_Horizontal_Layout_1)
        self.nav_1st_Horizontal_Widget_1.setFixedHeight(int(0.020833 * self.screen_width))
        self.nav_1st_Horizontal_Widget_1.setStyleSheet("background-color: transparent; border: none;")

        self.nav_1st_Horizontal_Layout.addWidget(self.nav_1st_Horizontal_Widget_1)
        self.nav_1st_Horizontal_Layout_1.addWidget(self.Single_img_label)
        self.nav_1st_Horizontal_Layout_1.addWidget(self.Auto_F_Extractor_Single)
        
        self.nav_1st_Horizontal_Layout_2 = QHBoxLayout()
        self.nav_1st_Horizontal_Layout_2.setContentsMargins(0,0,0,0)
        
        self.nav_1st_Horizontal_Widget_2 = QWidget()
        self.nav_1st_Horizontal_Widget_2.setLayout(self.nav_1st_Horizontal_Layout_2)
        self.nav_1st_Horizontal_Widget_2.setFixedHeight(int(0.020833 * self.screen_width))
        self.nav_1st_Horizontal_Widget_2.setStyleSheet("background-color: transparent; border: none;")

        self.nav_1st_Horizontal_Layout.addWidget(self.nav_1st_Horizontal_Widget_2)
        
        self.nav_1st_Horizontal_Layout_2.addWidget(self.Auto_mode_label)        
        self.nav_1st_Horizontal_Layout_2.addWidget(self.Auto_F_Extractor_Dataset)
        
        
        self.nav_1st_Horizontal_Layout_5 = QHBoxLayout()
        self.nav_1st_Horizontal_Layout_5.setContentsMargins(0,0,0,0)
        
        self.nav_1st_Horizontal_Widget_5 = QWidget()
        self.nav_1st_Horizontal_Widget_5.setLayout(self.nav_1st_Horizontal_Layout_5)
        self.nav_1st_Horizontal_Widget_5.setFixedHeight(int(0.020833 * self.screen_width))
        self.nav_1st_Horizontal_Widget_5.setStyleSheet("background-color: transparent; border: none;")

        self.nav_1st_Horizontal_Layout.addWidget(self.nav_1st_Horizontal_Widget_5)
        
        self.nav_1st_Horizontal_Layout_5.addWidget(self.toggle_visualizations_label)        
        self.nav_1st_Horizontal_Layout_5.addWidget(self.Windows_button_button44)
        
        
        self.nav_1st_Horizontal_Layout_3 = QHBoxLayout()
        self.nav_1st_Horizontal_Layout_3.setContentsMargins(0,0,0,0)
        
        self.nav_1st_Horizontal_Widget_3 = QWidget()
        self.nav_1st_Horizontal_Widget_3.setLayout(self.nav_1st_Horizontal_Layout_3)
        self.nav_1st_Horizontal_Widget_3.setFixedHeight(int(0.020833 * self.screen_width))
        self.nav_1st_Horizontal_Widget_3.setStyleSheet("background-color: transparent; border: none;")

        self.nav_1st_Horizontal_Layout.addWidget(self.nav_1st_Horizontal_Widget_3)
        
        self.nav_1st_Horizontal_Layout_3.addWidget(self.Save_ALL_label)        
        self.nav_1st_Horizontal_Layout_3.addWidget(self.Save_button44)
        
        Spacer_FE = QSpacerItem(0, int(0.0104167 * self.screen_width), QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        self.nav_1st_Horizontal_Layout.addSpacerItem(Spacer_FE)
        self.nav_1st_Horizontal_Layout.addWidget(self.trigger_Extraction_button)
        
        
        
        
        
        
        self.combo_box_Conventional = QComboBox(self)
        self.combo_box_Conventional.addItems(["80 - 20 Recommended", "70 - 30", "60 - 40"])
        self.combo_box_Conventional.setCurrentIndex(0)  # Set default choice to the first one
        self.combo_box_Conventional.setFixedSize(int(0.11458333 * self.screen_width), int(0.020833 * self.screen_width))  # Set fixed size

        # Apply a custom style sheet
        self.combo_box_Conventional.setStyleSheet("""
            QComboBox {
                border: 2px solid green;
                border-radius: 10px;
                padding: 5px;
                font-size: 18px;
                background-color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 14px;
                height: 14px;
            }
            QComboBox QAbstractItemView {
                border: 2px solid green;
                selection-background-color: green;
            }
        """)

        self.combo_box_Conventional.currentIndexChanged.connect(self.on_combobox_changed_Conventional)
        
        
        self.label_Combo_Conventional = ImageLabel(self, apply_effects = False)
        self.label_Combo_Conventional.setText("Split-ratio:")
        font1 = QFont()
        font1.setPointSize(int(0.008333 * self.screen_width))
        font1.setFamily("Helvetica")
        
        self.label_Combo_Conventional.setFont(font1)
        self.label_Combo_Conventional.setStyleSheet("color: white;")
        
        spacer_ComboBox_layout_spacer = QSpacerItem(int(0.02604167 * self.screen_width), 0, QSizePolicy.Fixed, QSizePolicy.Fixed)
        

        combo_box_Conventional_Horizontal_Layout = QHBoxLayout()
        combo_box_Conventional_Horizontal_Layout.setContentsMargins(0, 0, 0, 0)
        self.combo_box_Conventional_Horizontal_Widget = QWidget()
        self.combo_box_Conventional_Horizontal_Widget.setFixedHeight(int(0.02604167 * self.screen_width))
        self.combo_box_Conventional_Horizontal_Widget.setLayout(combo_box_Conventional_Horizontal_Layout)
        self.combo_box_Conventional_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        combo_box_Conventional_Horizontal_Layout.addWidget(self.label_Combo_Conventional)
        combo_box_Conventional_Horizontal_Layout.addWidget(self.combo_box_Conventional)
        combo_box_Conventional_Horizontal_Layout.addSpacerItem(spacer_ComboBox_layout_spacer)
        
        
        
        
        
        
        
        
        
        
        
        self.combo_box_AI = QComboBox(self)
        self.combo_box_AI.addItems(["80 - 20 Recommended", "70 - 30", "60 - 40"])
        self.combo_box_AI.setCurrentIndex(0)  # Set default choice to the first one
        self.combo_box_AI.setFixedSize(int(0.11458333 * self.screen_width), int(0.020833 * self.screen_width))  # Set fixed size

        # Apply a custom style sheet
        self.combo_box_AI.setStyleSheet("""
            QComboBox {
                border: 2px solid green;
                border-radius: 10px;
                padding: 5px;
                font-size: 18px;
                background-color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 14px;
                height: 14px;
            }
            QComboBox QAbstractItemView {
                border: 2px solid green;
                selection-background-color: green;
            }
        """)

        self.combo_box_AI.currentIndexChanged.connect(self.on_combobox_changed_AI)
        
        
        self.label_Combo_AI = ImageLabel(self, apply_effects = False)
        self.label_Combo_AI.setText("Split-ratio:")
        font1 = QFont()
        font1.setPointSize(int(0.008333 * self.screen_width))
        font1.setFamily("Helvetica")
        
        self.label_Combo_AI.setFont(font1)
        self.label_Combo_AI.setStyleSheet("color: white;")
        
        spacer_ComboBox_layout_spacer = QSpacerItem(int(0.02604167 * self.screen_width), 0, QSizePolicy.Fixed, QSizePolicy.Fixed)
        

        combo_box_AI_Horizontal_Layout = QHBoxLayout()
        combo_box_AI_Horizontal_Layout.setContentsMargins(0, 0, 0, 0)
        self.combo_box_AI_Horizontal_Widget = QWidget()
        self.combo_box_AI_Horizontal_Widget.setFixedHeight(int(0.02604167 * self.screen_width))
        self.combo_box_AI_Horizontal_Widget.setLayout(combo_box_AI_Horizontal_Layout)
        self.combo_box_AI_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        combo_box_AI_Horizontal_Layout.addWidget(self.label_Combo_AI)
        combo_box_AI_Horizontal_Layout.addWidget(self.combo_box_AI)
        combo_box_AI_Horizontal_Layout.addSpacerItem(spacer_ComboBox_layout_spacer)
        
        
        
        
        
        self.combo_box_Fused = QComboBox(self)
        self.combo_box_Fused.addItems(["80 - 20", "70 - 30 Recommended", "60 - 40"])
        self.combo_box_Fused.setCurrentIndex(1)  # Set default choice to the first one
        self.combo_box_Fused.setFixedSize(int(0.11458333 * self.screen_width), int(0.020833 * self.screen_width))  # Set fixed size

        # Apply a custom style sheet
        self.combo_box_Fused.setStyleSheet("""
            QComboBox {
                border: 2px solid green;
                border-radius: 10px;
                padding: 5px;
                font-size: 18px;
                background-color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 14px;
                height: 14px;
            }
            QComboBox QAbstractItemView {
                border: 2px solid green;
                selection-background-color: green;
            }
        """)

        self.combo_box_Fused.currentIndexChanged.connect(self.on_combobox_changed_Fused)
        
        
        self.label_Combo_Fused = ImageLabel(self, apply_effects = False)
        self.label_Combo_Fused.setText("Split-ratio:")
        font1 = QFont()
        font1.setPointSize(int(0.008333 * self.screen_width))
        font1.setFamily("Helvetica")
        
        self.label_Combo_Fused.setFont(font1)
        self.label_Combo_Fused.setStyleSheet("color: white;")
        
        spacer_ComboBox_layout_spacer = QSpacerItem(int(0.02604167 * self.screen_width), 0, QSizePolicy.Fixed, QSizePolicy.Fixed)
        

        combo_box_Fused_Horizontal_Layout = QHBoxLayout()
        combo_box_Fused_Horizontal_Layout.setContentsMargins(0, 0, 0, 0)
        self.combo_box_Fused_Horizontal_Widget = QWidget()
        self.combo_box_Fused_Horizontal_Widget.setFixedHeight(int(0.02604167 * self.screen_width))
        self.combo_box_Fused_Horizontal_Widget.setLayout(combo_box_Fused_Horizontal_Layout)
        self.combo_box_Fused_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        combo_box_Fused_Horizontal_Layout.addWidget(self.label_Combo_Fused)
        combo_box_Fused_Horizontal_Layout.addWidget(self.combo_box_Fused)
        combo_box_Fused_Horizontal_Layout.addSpacerItem(spacer_ComboBox_layout_spacer)
        
        
        
        
        
        
        
        
        


        nav_2nd_Vertical_Layout = QVBoxLayout()
        nav_2nd_Vertical_Layout.setContentsMargins(0, 0, 0, 0)
        self.nav_2nd_Vertical_Widget = QWidget()
        self.nav_2nd_Vertical_Widget.setLayout(nav_2nd_Vertical_Layout)
        self.nav_2nd_Vertical_Widget.setStyleSheet("background-color: transparent; border: none;")
        nav_2nd_Vertical_Layout.addWidget(self.combo_box_Conventional_Horizontal_Widget)
        nav_2nd_Vertical_Layout.addWidget(self.Conventional_RadioButtons_layout_widget)
        
        
        
        

        nav_2nd_Horizontal_Layout = QHBoxLayout()
        nav_2nd_Horizontal_Layout.setContentsMargins(0, 0, 0, 0)
        self.nav_2nd_Horizontal_Widget = QWidget()
        self.nav_2nd_Horizontal_Widget.setLayout(nav_2nd_Horizontal_Layout)
        self.nav_2nd_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        nav_2nd_Horizontal_Layout.addSpacerItem(spacer_nav3_H)
        nav_2nd_Horizontal_Layout.addWidget(self.nav_2nd_Vertical_Widget)









        nav_3rd_Vertical_Layout = QVBoxLayout()
        nav_3rd_Vertical_Layout.setContentsMargins(0, 0, 0, 0)
        self.nav_3rd_Vertical_Widget = QWidget()
        self.nav_3rd_Vertical_Widget.setLayout(nav_3rd_Vertical_Layout)
        self.nav_3rd_Vertical_Widget.setStyleSheet("background-color: transparent; border: none;")
        nav_3rd_Vertical_Layout.addWidget(self.combo_box_AI_Horizontal_Widget)
        nav_3rd_Vertical_Layout.addWidget(self.Ai_RadioButtons_layout_widget)
        
        
        nav_3rd_Horizontal_Layout = QHBoxLayout()
        nav_3rd_Horizontal_Layout.setContentsMargins(0, 0, 0, 0)
        self.nav_3rd_Horizontal_Widget = QWidget()
        self.nav_3rd_Horizontal_Widget.setLayout(nav_3rd_Horizontal_Layout)
        self.nav_3rd_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        nav_3rd_Horizontal_Layout.addSpacerItem(spacer_nav4_H)
        nav_3rd_Horizontal_Layout.addWidget(self.nav_3rd_Vertical_Widget)
        
        
        
        
        
        
        
        
        
        nav_4th_Vertical_Layout = QVBoxLayout()
        nav_4th_Vertical_Layout.setContentsMargins(0, 0, 0, 0)
        self.nav_4th_Vertical_Widget = QWidget()
        self.nav_4th_Vertical_Widget.setLayout(nav_4th_Vertical_Layout)
        self.nav_4th_Vertical_Widget.setStyleSheet("background-color: transparent; border: none;")
        nav_4th_Vertical_Layout.addWidget(self.combo_box_Fused_Horizontal_Widget)
        nav_4th_Vertical_Layout.addWidget(self.Fused_RadioButtons_layout_widget)
        

        
        nav_4th_Horizontal_Layout = QHBoxLayout()
        nav_4th_Horizontal_Layout.setContentsMargins(0, 0, 0, 0)
        nav_4th_Horizontal_Layout.setAlignment(Qt.AlignRight)
        
        self.nav_4th_Horizontal_Widget = QWidget()
        self.nav_4th_Horizontal_Widget.setLayout(nav_4th_Horizontal_Layout)
        self.nav_4th_Horizontal_Widget.setStyleSheet("background-color: transparent; border: none;")
        nav_4th_Horizontal_Layout.addSpacerItem(spacer_nav5_H)
        nav_4th_Horizontal_Layout.addWidget(self.nav_4th_Vertical_Widget)
        
        self.nav_layout.setSpacing(0)

        self.nav_layout.addSpacerItem(spacer_nav1)
        self.nav_layout.addWidget(self.first_button_layout_Widget, alignment= Qt.AlignLeft)
        
        self.nav_layout.addWidget(self.Conventional_CAD_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.nav_2nd_Horizontal_Widget, stretch=1)
        self.nav_layout.addWidget(self.Ai_Automated_CAD_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.nav_3rd_Horizontal_Widget,  stretch=1)
        self.nav_layout.addWidget(self.Fused_CVML_and_Ai_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.nav_4th_Horizontal_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.Feature_Extraction_and_Visualization_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.nav_1st_Horizontal_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.Reset_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.settings_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.help_button_layout_Widget, alignment= Qt.AlignLeft)
        self.nav_layout.addWidget(self.exit_button_layout_Widget, alignment= Qt.AlignLeft)
        
        self.nav_Widget_reduced = QWidget(self)
        self.nav_Widget = QWidget(self)
        self.nav_Widget_reduced.hide()
        self.nav_Widget.show()
        self.nav_Widget_reduced.setFixedWidth(int(0.033854167 * self.screen_width))

        self.menu_button = AnimatedButton("imgs/menu.svg", self)
        self.menu_button.setToolTip("Menu")
        self.menu_button.setMaximumSize(int(0.015625 * self.screen_width),int(0.015625 * self.screen_width))
        self.menu_button.clicked.connect(self.Toggle_nav_Widget)


        self.label_Title_name = QLabel("Knee OsteoArthritis CAD")
        self.label_Title_name.setStyleSheet("color: rgba(255, 255, 255, 150);")
        font = QFont()
        font.setBold(True)
        font.setPointSize(int(0.009275 * self.screen_width))
        self.label_Title_name.setFont(font)

        self.Minimize_button = AnimatedButton('imgs/minimize.svg', self)
        self.Minimize_button.setToolTip("Minimize")
        self.Minimize_button.setMinimumSize(int(0.013020833 * self.screen_width), int(0.013020833 * self.screen_width))
        self.Minimize_button.setMaximumSize(int(0.013020833 * self.screen_width), int(0.013020833 * self.screen_width))
        self.Minimize_button.setIconSize(QSize(int(0.013020833 * self.screen_width), int(0.013020833 * self.screen_width)))
        self.Minimize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Minimize_button.clicked.connect(self.toggleMinimized)  # Connect to the new toggleMinimized method
        
        self.Expand_button = AnimatedButton('imgs/reduce.svg', self)
        self.Expand_button.setToolTip("Maximize")
        self.Expand_button.setMinimumSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width))
        self.Expand_button.setMaximumSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width))
        self.Expand_button.setIconSize(QSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width)))
        self.Expand_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Expand_button.clicked.connect(self.Expand_Function)
        
        
        self.Restart_button = AnimatedButton('imgs/restart.svg', self)
        self.Restart_button.setToolTip("Need to Restart KOA CAD?")
        self.Restart_button.setMinimumSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width))
        self.Restart_button.setMaximumSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width))
        self.Restart_button.setIconSize(QSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width)))
        self.Restart_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.Restart_button.clicked.connect(self.restart_code)
        
        self.EXIT_button = AnimatedButton('imgs/x.svg', self)
        self.EXIT_button.setToolTip("close")
        self.EXIT_button.setMinimumSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width))
        self.EXIT_button.setMaximumSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width))
        self.EXIT_button.setIconSize(QSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width)))
        self.EXIT_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        self.EXIT_button.clicked.connect(self.closeEvent)
        
        self.titleBar_layout = QHBoxLayout()
        spacer_titleBar = QSpacerItem(int(0.3977 * self.screen_width), 0, QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer_titleBar2 = QSpacerItem(int(0.02604167 * self.screen_width), 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        
        self.titleBar_layout.setSpacing(int(0.00520833 * self.screen_width))
        self.titleBar_layout.addWidget(self.menu_button)
        self.titleBar_layout.addItem(spacer_titleBar)
        self.titleBar_layout.addWidget(self.label_Title_name)
        self.titleBar_layout.addItem(spacer_titleBar2)
        self.titleBar_layout.addWidget(self.Restart_button)
        self.titleBar_layout.addWidget(self.Minimize_button)
        self.titleBar_layout.addWidget(self.Expand_button)
        self.titleBar_layout.addWidget(self.EXIT_button)
        
        self.title_bar = QWidget(self)
        self.title_bar.setLayout(self.titleBar_layout)
        self.title_bar.setFixedHeight(int(0.044270833 * self.screen_width))
        self.central_layout.setSpacing(0)
        
        self.central_layout.addWidget(self.title_bar, 0, 0)

        gradient_style2 = """
            background: qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 rgba(0, 0, 0, 75), stop:1 rgba(0, 0, 0, 0));
        """
     
        self.nav_Widget_reduced.setLayout(self.nav_layout_reduced)
        self.nav_Widget.setLayout(self.nav_layout)
        self.nav_Widget_reduced.setStyleSheet(gradient_style2)
        self.nav_Widget.setStyleSheet(gradient_style2)

        self.Main_layout = QGridLayout()
        self.screen_layout = QGridLayout()
        
        self.Main_layout.setSpacing(0)
        self.Main_layout.addWidget(self.nav_Widget_reduced, 0, 0)
        self.Main_layout.addWidget(self.nav_Widget, 0, 0)
        self.Main_layout.addLayout(self.screen_layout, 0, 1)
        self.central_layout.addLayout(self.Main_layout, 1, 0)

        self.drag_position = None
        self.setMouseTracking(True)

        self.current_screen = None
        self.setCentralWidget(self.central_widget)
        self.show_screen4()
        
        
    def load_main_img(self):
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Images (*.DCM *.DICOM *.IMG *.DICONDE *.ITK *.VTK *.IMA *.png *.jpg *.jpeg)")
            
            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                
                self.Clear_All()
                
                self.file_path = file_paths[0]
                
                if self.file_path.lower().endswith(('.dcm', '.ima','.DICOM','.IMG','.DICONDE','.VTK','.ITK')):
                    dicom_data = pydicom.dcmread(self.file_path)
                    pixel_array = dicom_data.pixel_array
                    image_data_np = pixel_array.astype(np.uint8)
                    self.file_path = self.save_image(image_data_np)

                self.image = cv2.imread(self.file_path)

                if self.image is not None:
                    self.image = self.Feature_Extraction_and_Visualization_Screen.Apply_Padding(self.image)
                    height, width = self.image.shape[:2]
                    format = QImage.Format_Grayscale8 if len(self.image.shape) == 2 else QImage.Format_RGB888
                    q_img = QImage(self.image.data, width, height, self.image.strides[0], format)

                    pixmap = QPixmap.fromImage(q_img).scaled(int(0.3125 * self.screen_width), int(0.3125 * self.screen_width), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    # rounded_pixmap = self.create_round_image(pixmap)

                    if pixmap:
                        self.display_image(pixmap, self.Conventional_CAD_Screen.image_label)
                        self.display_image(pixmap, self.AI_Automated_CAD_Screen.image_label)
                        self.display_image(pixmap, self.Fused_CVML_and_Ai_CAD_Screen.image_label)
                        self.display_image(pixmap, self.Feature_Extraction_and_Visualization_Screen.JSN_label)
                        
        
                        self.Feature_Extraction_and_Visualization_Screen.JSN_label.disableEffects()        
                        
                        self.Feature_Extraction_and_Visualization_Screen.mouse_double_clicked = 1
                        self.AI_Automated_CAD_Screen.mouse_double_clicked = 1
                        
                        
                        
                        
                        self.Feature_Extraction_and_Visualization_Screen.image_path = self.file_path
                        self.Feature_Extraction_and_Visualization_Screen.image = self.image
        
                        self.Conventional_CAD_Screen.file_path = self.file_path
                        self.Conventional_CAD_Screen.image = self.image
                        
                        self.AI_Automated_CAD_Screen.file_path = self.file_path
                        self.AI_Automated_CAD_Screen.image = self.image
                        
                        
                        print(f"self.Feature_Extraction_and_Visualization_Screen.mouse_double_clicked: {self.Feature_Extraction_and_Visualization_Screen.mouse_double_clicked}") 
                        print(f"self.AI_Automated_CAD_Screen.mouse_double_clicked: {self.AI_Automated_CAD_Screen.mouse_double_clicked}") 
                else:
                    print("Error: Unable to load the image.")
                    image_path = "imgs/Feature_Extraction.png"
                    self.Feature_Extraction_and_Visualization_Screen.set_background_image(self.Conventional_CAD_Screen.image_label, image_path)
                    self.Feature_Extraction_and_Visualization_Screen.set_background_image(self.AI_Automated_CAD_Screen.image_label, image_path)
                    self.Feature_Extraction_and_Visualization_Screen.set_background_image(self.Fused_CVML_and_Ai_CAD_Screen.image_label, image_path)
                    
                    
                    
                        


    @pyqtSlot()
    def classify_all(self):
        if self.file_path is not None:
            self.overlay.setVisible(True)
            QApplication.processEvents()  # Process UI events to keep it responsive
            
            self.Conventional_CAD_Screen.carry_out()
            self.AI_Automated_CAD_Screen.carry_out_Ai_CAD()
            self.Fused_CVML_and_Ai_CAD_Screen.carry_out()
            self.clasify_indicator = 1
            
            self.overlay.setVisible(False)
            
        else:
            pass


    @pyqtSlot()
    def on_computation_finished(self):
        self.overlay.setVisible(False)
    
    @pyqtSlot()
    def start_Extraction(self):
        if self.file_path is not None:
            self.overlay.setVisible(True)
            QApplication.processEvents()  # Process UI events to keep it responsive
            
            self.Feature_Extraction_and_Visualization_Screen.load_image(self.file_path)
        
            self.overlay.setVisible(False)
        
        else:
            pass
        
        
    # def create_round_image(self, pixmap):
    #     if pixmap.isNull():
    #         return None

    #     size = pixmap.size()
    #     mask = QPixmap(size)
    #     mask.fill(Qt.transparent)
    #     painter = QPainter(mask)
    #     painter.setRenderHint(QPainter.Antialiasing)
    #     painter.setBrush(Qt.black)
    #     painter.setPen(Qt.transparent)
    #     path = QPainterPath()
    #     path.addRoundedRect(0, 0, size.width(), size.height(), int(0.0520833 * self.screen_width), int(0.0520833 * self.screen_width))
    #     painter.drawPath(path)
    #     painter.end()
    #     result = QPixmap(size)
    #     result.fill(Qt.transparent)
    #     painter.begin(result)
    #     painter.setClipPath(path)
    #     painter.drawPixmap(0, 0, pixmap)
    #     painter.end()
    #     return result

    def display_image(self, img, target_label=None):
        if img is not None:
            if isinstance(img, QPixmap):
                if target_label is not None:
                    target_label.setPixmap(img)
                    target_label.setScaledContents(True)
                else:
                    self.Conventional_CAD_Screen.image_label.setPixmap(img)
                    self.Conventional_CAD_Screen.image_label.setScaledContents(True)
                    
                    self.AI_Automated_CAD_Screen.image_label.setPixmap(img)
                    self.AI_Automated_CAD_Screen.image_label.setScaledContents(True)
                    
                    self.Fused_CVML_and_Ai_CAD_Screen.image_label.setPixmap(img)
                    self.Fused_CVML_and_Ai_CAD_Screen.image_label.setScaledContents(True)
            else:
                print("\033[91mUnsupported image format.\033[0m")
                if target_label is not None:
                    target_label.setText("\033[91mUnsupported image format.\033[0m")
                else:
                    self.image_label.setText("\033[91mUnsupported image format.\033[0m")
                    
    def save_image(self, image):
        _, temp_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(temp_path, image)
        return temp_path
            
    
    def Windows_button_Toggling_Visualizations(self):
        if (self.switch2 == False):
            self.Windows_button_button44.setIcon(QIcon('imgs/eye.svg'))
            self.Windows_button_button44.setFixedSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
            self.Windows_button_button44.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
            self.toggle_visualizations_label.setText("Hide Visualizations")
            self.Windows_button_button44.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
            self.switch2 = True
            self.variableChanged_Visualization.emit(self.switch2)
            self.Feature_Extraction_and_Visualization_Screen.switch2_toggle()
        else:
            self.Windows_button_button44.setIcon(QIcon('imgs/eye-off.svg'))
            self.Windows_button_button44.setFixedSize(int(0.020833 * self.screen_width), int(0.020833 * self.screen_width))
            self.Windows_button_button44.setIconSize(QSize(int(0.020833 * self.screen_width), int(0.020833 * self.screen_width)))
            self.toggle_visualizations_label.setText("Show Visualizations")
            self.Windows_button_button44.setStyleSheet("QPushButton { color: rgba(255, 255, 255, 255); border: none; background: transparent; border-radius: 50px; text-align: left;}")
            self.switch2 = False
            self.variableChanged_Visualization.emit(self.switch2)
            self.Feature_Extraction_and_Visualization_Screen.switch2_toggle()
            
    def switch_mode_Auto_Mode(self):
        if (self.switch == True):
            self.Auto_F_Extractor_Single.setIcon(QIcon('imgs/disable-mode.svg'))
            self.Auto_F_Extractor_Single.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.Auto_F_Extractor_Single.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.Auto_F_Extractor_Single.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.Auto_F_Extractor_Single.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
            self.Auto_F_Extractor_Dataset.setIcon(QIcon('imgs/enable-mode.png'))
            self.Auto_F_Extractor_Dataset.setMinimumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.Auto_F_Extractor_Dataset.setMaximumSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width))
            self.Auto_F_Extractor_Dataset.setIconSize(QSize(int(0.0208333 * self.screen_width),int(0.0208333 * self.screen_width)))
            self.Auto_F_Extractor_Dataset.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
            text = "It will be minimized, then restored again after the automation completes.\n"
            msgBox = CustomMessageBox("Automatic Mode", "imgs/alert-triangle.png", "OK", 0)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Automatic Mode")
            msgBox.setWindowIcon(QIcon("imgs/alert-triangle.png"))
            msgBox.setText(text)
            font = QFont()
            font.setPointSize(int(0.005729166 * self.screen_width))
            # font.setPointSize(11)
            msgBox.setFont(font)
            # msgBox.addButton(QMessageBox.Ok)
            msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
            msgBox.exec_()
            self.switch= False
            self.variableChanged_Auto_mode.emit(self.switch)
            
        else:
            self.Auto_F_Extractor_Single.setIcon(QIcon('imgs/enable-mode.png'))
            self.Auto_F_Extractor_Single.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
            self.Auto_F_Extractor_Single.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
            self.Auto_F_Extractor_Single.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
            self.Auto_F_Extractor_Single.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            self.Auto_F_Extractor_Single.setToolTip("switch to Data set Automatic Feature Extractor")
            
            self.Auto_F_Extractor_Dataset.setIcon(QIcon('imgs/disable-mode.svg'))
            self.Auto_F_Extractor_Dataset.setMinimumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
            self.Auto_F_Extractor_Dataset.setMaximumSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width))
            self.Auto_F_Extractor_Dataset.setIconSize(QSize(int(0.020833 * self.screen_width),int(0.020833 * self.screen_width)))
            self.Auto_F_Extractor_Dataset.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
            self.switch = True
            self.variableChanged_Auto_mode.emit(self.switch)
            
    def toggle_intensity_normalization_Equalize(self):
        if self.perform_intensity_normalization == False:
            self.perform_intensity_normalization = True
            self.variableChanged_Equalize_Feature_Visualization.emit(self.perform_intensity_normalization)
            
            self.Equalize_button.setIcon(QIcon('imgs/check-square.svg'))
            self.Equalize_button.setMinimumSize(int(0.0182292 * self.screen_width),int(0.0182292 * self.screen_width))
            self.Equalize_button.setMaximumSize(int(0.0182292 * self.screen_width),int(0.0182292 * self.screen_width))
            self.Equalize_button.setIconSize(QSize(int(0.0182292 * self.screen_width),int(0.0182292 * self.screen_width)))
            self.Equalize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
            
        else:

            self.perform_intensity_normalization = False
            self.variableChanged_Equalize_Feature_Visualization.emit(self.perform_intensity_normalization)

            self.Equalize_button.setIcon(QIcon('imgs/square.svg'))
            self.Equalize_button.setMinimumSize(int(0.0182292 * self.screen_width),int(0.0182292 * self.screen_width))
            self.Equalize_button.setMaximumSize(int(0.0182292 * self.screen_width),int(0.0182292 * self.screen_width))
            self.Equalize_button.setIconSize(QSize(int(0.0182292 * self.screen_width),int(0.0182292 * self.screen_width)))
            self.Equalize_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")

            self.Feature_Extraction_and_Visualization_Screen.Intensity_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.Binarization_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.LBPImgWindow_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.LTPImgWindow_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.LBPHistogramWindow_label.clear()
            
            # self.Feature_Extraction_and_Visualization_Screen.LTPHistogramWindow.LTPHistogramWindow_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.LTPHistogramWindow_label.clear()
            
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.HOG_image_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.HOGHistogram.HOGHistogram_label.clear()
            self.Feature_Extraction_and_Visualization_Screen.Binarization_label.setText("Binarization")
            self.Feature_Extraction_and_Visualization_Screen.Intensity_label.setText("Equalization")
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.LBPImgWindow_label.setText("LBP")
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.LTPImgWindow_label.setText("LTP")
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.LBPHistogramWindow_label.setText("LBP Histogram Window")
            
            # self.Feature_Extraction_and_Visualization_Screen.LTPHistogramWindow.LTPHistogramWindow_label.setText("LTP Histogram Window")
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.LTPHistogramWindow_label.setText("LTP Histogram Window")
            
            self.Feature_Extraction_and_Visualization_Screen.VisualizationWindow.HOG_image_label.setText("HOG Image")
            self.Feature_Extraction_and_Visualization_Screen.HOGHistogram.HOGHistogram_label.setText("HOG Histogram")
       
        # if self.Feature_Extraction_and_Visualization_Screen.image_path is not None:
        #     self.Feature_Extraction_and_Visualization_Screen.load_image(self.Feature_Extraction_and_Visualization_Screen.image_path)
        # else:
        #     pass
        

    def show_screen1(self):
        self.clear_screen_layout()
        
        self.nav_1st_Horizontal_Widget.setVisible(True)
        self.nav_2nd_Horizontal_Widget.setVisible(False)
        self.nav_3rd_Horizontal_Widget.setVisible(False)
        self.nav_4th_Horizontal_Widget.setVisible(False)
                
        self.spacer_nav_reduced4_layout_Widget.setVisible(True)
        self.spacer_nav_reduced3_layout_Widget.setVisible(False)
        self.spacer_nav_reduced6_layout_Widget.setVisible(False)
        self.spacer_nav_reduced5_layout_Widget.setVisible(False)
        
        self.current_screen = self.Feature_Extraction_and_Visualization_Screen

        self.screen_layout.addWidget(self.current_screen)
        self.current_screen.show()

    def show_screen2(self):
        self.clear_screen_layout()

        self.nav_1st_Horizontal_Widget.setVisible(False)
        self.nav_2nd_Horizontal_Widget.setVisible(True)
        self.nav_3rd_Horizontal_Widget.setVisible(False)
        self.nav_4th_Horizontal_Widget.setVisible(False)
        
        
        self.spacer_nav_reduced4_layout_Widget.setVisible(True)
        self.spacer_nav_reduced3_layout_Widget.setVisible(False)
        self.spacer_nav_reduced6_layout_Widget.setVisible(False)
        self.spacer_nav_reduced5_layout_Widget.setVisible(False)

        self.current_screen = self.Conventional_CAD_Screen
        
        self.screen_layout.addWidget(self.current_screen)
        self.current_screen.show()

    def show_screen3(self):
        self.clear_screen_layout()
        
        self.nav_1st_Horizontal_Widget.setVisible(False)
        self.nav_2nd_Horizontal_Widget.setVisible(False)
        self.nav_3rd_Horizontal_Widget.setVisible(True)
        self.nav_4th_Horizontal_Widget.setVisible(False)
        
        self.spacer_nav_reduced4_layout_Widget.setVisible(True)
        self.spacer_nav_reduced3_layout_Widget.setVisible(False)
        self.spacer_nav_reduced6_layout_Widget.setVisible(False)
        self.spacer_nav_reduced5_layout_Widget.setVisible(False)
        
        self.current_screen = self.AI_Automated_CAD_Screen
        
        self.screen_layout.addWidget(self.current_screen)
        self.current_screen.show()

    def show_screen4(self):
        self.clear_screen_layout()
        
        self.current_screen = self.Fused_CVML_and_Ai_CAD_Screen
        
        self.nav_1st_Horizontal_Widget.setVisible(False)
        self.nav_2nd_Horizontal_Widget.setVisible(False)
        self.nav_3rd_Horizontal_Widget.setVisible(False)
        self.nav_4th_Horizontal_Widget.setVisible(True)
        
        
        self.spacer_nav_reduced4_layout_Widget.setVisible(True)
        self.spacer_nav_reduced3_layout_Widget.setVisible(False)
        self.spacer_nav_reduced6_layout_Widget.setVisible(False)
        self.spacer_nav_reduced5_layout_Widget.setVisible(False)
        
        
        self.screen_layout.addWidget(self.current_screen)
        self.current_screen.show()


    def clear_screen_layout(self):
        while self.screen_layout.count() > 0:
            item = self.screen_layout.takeAt(0)
            widget = item.widget()
            if widget:
                self.screen_layout.removeWidget(widget)
                widget.setParent(None)
    
    def Toggle_nav_Widget(self):
        current_widget = self.nav_Widget if self.nav_Widget.isVisible() else self.nav_Widget_reduced
        next_widget = self.nav_Widget_reduced if current_widget == self.nav_Widget else self.nav_Widget
        self.animation = QPropertyAnimation(current_widget, b"geometry")
        if current_widget == self.nav_Widget:
            self.animation.setDuration(125)  # Duration for the collapsing transition
        else:
            self.animation.setDuration(100)  # Duration for the expanding transition

        self.animation.setStartValue(current_widget.geometry())  # The animation starts from the current widget's geometry
        self.animation.setEndValue(QRect(next_widget.geometry().x(), current_widget.geometry().y(), next_widget.geometry().width(), current_widget.geometry().height()))
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()
        self.animation.finished.connect(lambda: self.nav_Widget.setVisible(not self.nav_Widget.isVisible()))
        self.animation.finished.connect(lambda: self.nav_Widget_reduced.setVisible(not self.nav_Widget_reduced.isVisible()))

    def on_radio_button_Conventional_clicked(self, option):
        self.FirstSetVariable = option + 1
        self.variableChanged.emit(self.FirstSetVariable)
        
        if self.clasify_indicator == 1:
            self.Conventional_CAD_Screen.show_Predictions()
        else:
            pass
            print(f"clasify_indicator {self.clasify_indicator}")
        
    def on_radio_button_AI_clicked(self, option):
        self.SecondSetVariable = option + 1
        self.variableChanged2.emit(self.SecondSetVariable)
        
        if self.clasify_indicator == 1:
            self.AI_Automated_CAD_Screen.show_predictions()
        else:
            pass
            print(f"clasify_indicator {self.clasify_indicator}")
            
    
    def on_radio_button_Fused_clicked(self, option):
        self.ThirdSetVariable = option + 1
        self.variableChanged3.emit(self.ThirdSetVariable)
        
        if self.clasify_indicator == 1:
            self.Fused_CVML_and_Ai_CAD_Screen.show_predictions()
        else:
            pass
            print(f"clasify_indicator {self.clasify_indicator}")
            

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.rect().contains(event.pos()):
            self.Expand_Function()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.isFullScreen:
            self.drag_position = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = None
            event.accept()

    def toggleMinimized(self):
        if self.isMinimized():
            self.showNormal()
        else:
            self.showMinimized()
# __________________________________________ resize to Full Screen _______________________________________________
    def resizeEvent(self, event):
        pixmap1 = self.pixmap.scaled(self.size(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatioByExpanding)
        self.background_label.setPixmap(pixmap1)
        if hasattr(self, 'background_label'):
            self.background_label.setGeometry(0, 0, self.width(), self.height())

    def restart_code(self):
        msgBox = CustomMessageBox("Do you want to restart?", "imgs/restart.svg", "Restart", 1)
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        reply = msgBox.exec_()

        if reply == QMessageBox.AcceptRole:
            QApplication.quit()
            subprocess.Popen([sys.executable] + sys.argv)
        else:
            pass
        
    def Settings(self):
        self.SettingsWindow.setVisible(not self.SettingsWindow.isVisible())

    def Clear_All(self):
        self.file_path = None
        self.image = None
        
        self.Feature_Extraction_and_Visualization_Screen.first_itr = 1
        self.Feature_Extraction_and_Visualization_Screen.interval = 0
        self.Feature_Extraction_and_Visualization_Screen.duration = 0
        
        self.Feature_Extraction_and_Visualization_Screen.image_label.enableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.JSN_label.disableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.YOLO_label.enableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.Intensity_label.enableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.Binarization_label.enableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.edge_label.enableEffects() 
        
        self.Feature_Extraction_and_Visualization_Screen.image_label.leaveEvent(True)        
        self.Feature_Extraction_and_Visualization_Screen.JSN_label.disableEffects()        
        self.Feature_Extraction_and_Visualization_Screen.YOLO_label.leaveEvent(True)        
        self.Feature_Extraction_and_Visualization_Screen.Intensity_label.leaveEvent(True)        
        self.Feature_Extraction_and_Visualization_Screen.Binarization_label.leaveEvent(True)        
        self.Feature_Extraction_and_Visualization_Screen.edge_label.leaveEvent(True) 
        
        self.Feature_Extraction_and_Visualization_Screen.on_button_click()
        self.AI_Automated_CAD_Screen.on_button_click()
        self.Conventional_CAD_Screen.on_button_click()
        self.Fused_CVML_and_Ai_CAD_Screen.on_button_click()
        self.HelpWindow.reset_to_initial_Pos()
        
        # self.combo_box_Conventional.setCurrentIndex(0)
        # self.combo_box_AI.setCurrentIndex(0)
        # self.combo_box_Fused.setCurrentIndex(0)
        
        self.switch2 = True
        self.Windows_button_Toggling_Visualizations()
            
        
    def help(self):
        self.HelpWindow.setVisible(not self.HelpWindow.isVisible())
        
    def closeEvent(self, event):
        msgBox = CustomMessageBox("Do you want to exit?", "imgs/x.svg", "Exit", 1)
        msgBox.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        reply = msgBox.exec_()

        if reply == QMessageBox.AcceptRole:
            exit(0)
        else:
            if isinstance(event, QEvent):
                event.ignore()
            else:
                pass
        
    def Expand_Function(self):
        if self.isFullScreen:
            self.setWindowFlag(Qt.FramelessWindowHint)
            self.showNormal()
            self.isFullScreen = False
            self.Expand_button.setIcon(QIcon('imgs/expand.svg'))
            self.Expand_button.setMinimumSize(int(0.0104167 * self.screen_width),int(0.0104167 * self.screen_width))
            self.Expand_button.setMaximumSize(int(0.0104167 * self.screen_width),int(0.0104167 * self.screen_width))
            self.Expand_button.setIconSize(QSize(int(0.0104167 * self.screen_width),int(0.0104167 * self.screen_width)))
            self.Expand_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")
        
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.showFullScreen()
            self.isFullScreen = True
            self.Expand_button.setIcon(QIcon('imgs/reduce.svg'))
            self.Expand_button.setMinimumSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width))
            self.Expand_button.setMaximumSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width))
            self.Expand_button.setIconSize(QSize(int(0.013020833 * self.screen_width),int(0.013020833 * self.screen_width)))
            self.Expand_button.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 50px; }")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen:
            self.Expand_Function()
        else:
            super().keyPressEvent(event)
#  __________________________________________________ main() func. ____________________________________________________

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = HomeScreen()
    main_window.show()
    sys.exit(app.exec_())