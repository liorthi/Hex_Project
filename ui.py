import math
from PySide6.QtWidgets import QWidget, QMessageBox
from PySide6.QtGui import QPainter, QPolygonF, QColor, QPen
from PySide6.QtCore import Qt, QPointF, QTimer

from board import RED, BLUE, EMPTY


class HexWidget(QWidget):
    """
    View component - handles only display and user input
    All game logic is delegated to the Controller
    """

    def __init__(self, controller):
        super().__init__()

        self.controller = controller
        self.size = controller.board_size

        # Visual settings
        self.hex_radius = 30
        self.margin = 100

        self.setWindowTitle("Hex")
        self.setMinimumSize(700, 700)

        # Start the game after UI is ready
        QTimer.singleShot(100, self.controller.start_game)

    # ---------- Geometry ----------
    def hex_center(self, r, c):
        """Calculate hex center for diamond layout with pointy-top hexagons"""
        R = self.hex_radius
        # For pointy-top hexagons in diamond layout
        x = self.margin + c * math.sqrt(3) * R + r * math.sqrt(3) * R / 2
        y = self.margin + r * 1.5 * R
        return QPointF(x, y)

    def hex_polygon(self, center):
        """Create a pointy-top hexagon"""
        R = self.hex_radius
        points = []
        for i in range(6):
            # Start at 30 degrees for pointy-top
            angle = math.pi / 180 * (60 * i + 30)
            points.append(QPointF(
                center.x() + R * math.cos(angle),
                center.y() + R * math.sin(angle)
            ))
        return QPolygonF(points)

    # ---------- Painting ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get board state from controller
        board_grid = self.controller.get_board_state()

        # Draw hexagons
        for r in range(self.size):
            for c in range(self.size):
                center = self.hex_center(r, c)
                poly = self.hex_polygon(center)

                cell = board_grid[r, c]
                if cell == RED:
                    painter.setBrush(QColor("red"))
                elif cell == BLUE:
                    painter.setBrush(QColor("blue"))
                else:
                    painter.setBrush(QColor("white"))

                painter.setPen(Qt.black)
                painter.drawPolygon(poly)

        # Draw colored borders
        self.draw_borders(painter)

    def draw_borders(self, painter):
        """Draw red borders on top/bottom and blue borders on left/right"""
        border_width = 8

        # RED: Top edge (upper-right diagonal)
        pen = QPen(QColor("red"), border_width)
        painter.setPen(pen)
        for c in range(self.size):
            center = self.hex_center(0, c)
            poly = self.hex_polygon(center)
            painter.drawLine(poly[5], poly[4])
            painter.drawLine(poly[3], poly[4])

        # RED: Bottom edge (lower-left diagonal)
        for c in range(self.size):
            center = self.hex_center(self.size - 1, c)
            poly = self.hex_polygon(center)
            painter.drawLine(poly[1], poly[2])
            painter.drawLine(poly[0], poly[1])

        # BLUE: Left edge (upper-left diagonal)
        pen = QPen(QColor("blue"), border_width)
        painter.setPen(pen)
        for r in range(self.size):
            center = self.hex_center(r, 0)
            poly = self.hex_polygon(center)
            painter.drawLine(poly[2], poly[3])
            painter.drawLine(poly[1], poly[2])

        # BLUE: Right edge (lower-right diagonal)
        for r in range(self.size):
            center = self.hex_center(r, self.size - 1)
            poly = self.hex_polygon(center)
            painter.drawLine(poly[5], poly[4])
            painter.drawLine(poly[5], poly[0])

    # ---------- Mouse handling ----------
    def mousePressEvent(self, event):
        """Handle mouse clicks - only process if human player's turn"""
        if not self.controller.is_human_turn():
            return

        if not self.controller.is_game_active():
            return

        pos = event.position()
        for r in range(self.size):
            for c in range(self.size):
                center = self.hex_center(r, c)
                poly = self.hex_polygon(center)

                if poly.containsPoint(pos, Qt.OddEvenFill):
                    # Delegate move to controller
                    self.controller.place_tile(r, c)
                    return

    # ---------- Slots (called by Controller) ----------
    def update_display(self):
        """Update the display - called by controller when board changes"""
        self.update()

    def handle_game_over(self, winner):
        """Handle game over - called by controller"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Game Over")

        if winner == "TIE":
            msg.setText("It's a tie!")
        else:
            msg.setText(f"{winner} wins!")

        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Reset)
        msg.button(QMessageBox.Reset).setText("New Game")

        result = msg.exec()

        if result == QMessageBox.Reset:
            self.controller.reset_game()