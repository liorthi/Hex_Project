import math
from PySide6.QtWidgets import QWidget, QApplication, QMessageBox
from PySide6.QtGui import QPainter, QPolygonF, QColor, QPen
from PySide6.QtCore import Qt, QPointF, QTimer

from game import Game
from board import RED, BLUE, EMPTY
from player import RandomAI, HumanPlayer

class HexWidget(QWidget):
    HUMAN_PLAYER_TYPES = [HumanPlayer]

    def __init__(self, size=7, red_player=None, blue_player=None):
        super().__init__()

        red_player = red_player if red_player else HumanPlayer()
        blue_player = blue_player if blue_player else RandomAI()

        self.game = Game(size, red_player, blue_player)
        self.size = size

        self.hex_radius = 30
        self.margin = 100

        self.setWindowTitle("Hex")
        self.setMinimumSize(700, 700)

        QTimer.singleShot(100, self.check_and_make_ai_move)


    # ---------- Player utilities ----------
    def get_current_player(self):
        return self.game.players[self.game.current]

    def is_human_player(self, player):
        return type(player) in self.HUMAN_PLAYER_TYPES

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

        # Draw hexagons
        for r in range(self.size):
            for c in range(self.size):
                center = self.hex_center(r, c)
                poly = self.hex_polygon(center)

                cell = self.game.board.grid[r, c]
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
        current_player = self.get_current_player()
        if not self.is_human_player(current_player):
            return

        pos = event.position()
        for r in range(self.size):
            for c in range(self.size):
                center = self.hex_center(r, c)
                poly = self.hex_polygon(center)

                if poly.containsPoint(pos, Qt.OddEvenFill):
                    if self.game.board.grid[r, c] == EMPTY:
                        self.make_move(r, c)
                    return

    # ---------- Game logic ----------
    def make_move(self, r, c):
        game = self.game

        game.board.place(r, c, game.current)
        self.update()

        # Check win
        if game.current == RED and game.red_wins():
            self.show_win_dialog("RED")
            return

        if game.current == BLUE and game.blue_wins():
            self.show_win_dialog("BLUE")
            return

        # Switch player
        game.switch_player()

        QApplication.processEvents()
        self.check_and_make_ai_move()

    def check_and_make_ai_move(self):
        player = self.get_current_player()

        if not self.is_human_player(player):
            r, c = player.get_move(self.game.board)
            self.make_move(r, c)

    def show_win_dialog(self, winner):
        msg = QMessageBox(self)
        msg.setWindowTitle("Game Over")
        msg.setText(f"{winner} wins!")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Reset)
        msg.button(QMessageBox.Reset).setText("New Game")

        result = msg.exec()

        if result == QMessageBox.Reset:
            self.game = Game(
                self.size,
                self.game.players[RED],
                self.game.players[BLUE]
            )
            self.update()
            QTimer.singleShot(100, self.check_and_make_ai_move)
