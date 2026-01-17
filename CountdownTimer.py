# CountdownTimer.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

class CountdownTimer(QWidget):
    def __init__(self, total_time_sec):
        super().__init__()
        # self.total_time_sec = total_time_sec
        self.remaining_time_sec = total_time_sec
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.timer_label = QLabel(self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        font = QFont('Arial', 48, QFont.Bold)
        self.timer_label.setFont(font)
        self.timer_label.setStyleSheet("color: white;")
        self.update_timer_display()
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.timer_label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # Update timer every second
        
        screen_geometry = QApplication.primaryScreen().geometry()
        widget_width = 150
        widget_height = 80
        x_pos = screen_geometry.width() - widget_width - 50
        y_pos = screen_geometry.height() - widget_height - 50
        
        self.setGeometry(x_pos, y_pos, widget_width, widget_height)

        self.hide()

    def update_timer_display(self):
        minutes = int(self.remaining_time_sec) // 60
        seconds = int(self.remaining_time_sec) % 60
        self.timer_label.setText(f"{minutes:02}:{seconds:02}")
        
    def update_timer(self):
        if self.remaining_time_sec > 0:
            self.remaining_time_sec -= 1
            self.update_timer_display()
        else:
            self.timer.stop()
            self.close()  # Close the application window when countdown ends


def main():
    app = QApplication(sys.argv)
    timer_window = CountdownTimer(10)  # Example: 5 minutes countdown
    timer_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
