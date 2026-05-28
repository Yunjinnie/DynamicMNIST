import os
import sys
import csv
import time
import datetime
import random
import argparse
import string
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QPushButton, QLabel, QApplication
from PyQt6.QtCore import Qt

class Canvas(QtWidgets.QLabel):
    def __init__(self, width, height, trj_guidance, print_activity=False, guide_font='Helvetica'):
        '''
        width: horizontal, x axis.
        height: vertical, y axis.
        '''
        super().__init__()
        self.width = width
        self.height = height
        self.trj_guidance = trj_guidance
        self.print_activity = print_activity
        self.guide_font = guide_font
        self.canvas_color = 'white'
        self.pen_color = 'black'
        self.pen_width = 12
        self.h_guidelines = 3
        self.v_guidelines = 3
        self.target_character = None

        # Set a pixel map used as a canvas.
        self.pixmap = QtGui.QPixmap(width, height)
        self.reset()

        # Set a cross-shaped cursor.
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_character(self, target_character):
        self.target_character = target_character

    def save_canvas(self, path):
        self.pixmap.save(path, 'png')

    def reset(self):
        self.i_stroke = 0
        self.x_prev, self.y_prev = None, None
        self.time = None
        self.time_start = None
        self.dots = list()
        self.pixmap.fill(QtGui.QColor(self.canvas_color))
        # START: Set guidelines.
        painter = QtGui.QPainter(self.pixmap)
        pen = painter.pen()
        pen.setWidth(1) # 4 was used for guide images
        pen.setColor(QtGui.QColor('#D3D3D3')) # light gray
        painter.setPen(pen)
        for i in range(1,self.h_guidelines+1):
            painter.drawLine(int(i*self.width/self.h_guidelines - 0.5*self.width/self.h_guidelines), 0, int(i*self.width/self.h_guidelines) - int(0.5*self.width/self.h_guidelines), self.height)
        for i in range(1,self.v_guidelines+1):
            painter.drawLine(0, int(i*self.height/self.v_guidelines - 0.5*self.height/self.v_guidelines), self.width, int(i*self.height/self.v_guidelines - 0.5*self.height/self.v_guidelines))
        painter.end()
        # END: Set guidelines.
        if self.trj_guidance and self.target_character is not None:
            self.draw_guide_trajectory(self.target_character)
        self.setPixmap(self.pixmap)

    def configure_pen(self, pen):
        pen.setWidth(self.pen_width)
        pen.setColor(QtGui.QColor(self.pen_color))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

    def mouseMoveEvent(self, e):
        # Use this event handler if you want to write with a mouse.
        pass

    def mouseReleaseEvent(self, e):
        if self.print_activity:
            print('detached')
        self.x_prev = None
        self.y_prev = None
        x = e.position().x()
        y = e.position().y()
        if (x < 0) or ((self.width - 1) <= x) \
                or (y < 0) or ((self.height - 1) <= y):
            # Detached out of the canvas.
            self.reset()
        else:
            # Detached in the canvas.
            self.i_stroke += 1

    def tabletEvent(self, e):
        # Get the current data from the tablet.
        x = e.position().x()
        y = e.position().y()
        xT = int(e.xTilt()) # Tile value is always an integer.
        yT = int(e.yTilt())
        p = e.pressure() # Between 0 and 1.
        # e.rotation() # Always zero.

        # Reset the canvas if the position is out of the canvas.
        if (x < 0) or ((self.width - 1) <= x) \
                or (y < 0) or ((self.height - 1) <= y):
            self.reset()
            return

        # Save the current dot.
        ## Case 1: Frist event.
        if self.x_prev is None: # First event.
            self.x_prev = x
            self.y_prev = y
            self.time = 0
            self.e_time = 0
            self.time_start = time.time() * 1000 # ms
            self.time_start_e = e.timestamp() # ms
            self.dots.append((self.i_stroke, self.time, self.e_time, x, y, xT, yT, p))
            if self.print_activity:
                print('len {} | i_stroke {} | (t,x,y,xT,yT,pres) = ({:.2f}, {:.2f}, {:.2f}, {:d}, {:d}, {:.2f})'.format(
                    len(self.dots), self.i_stroke, self.time, x, y, xT, yT, p
                ))
            return # Ignore the first time.

        ## Case 2: Events after the first event.
        self.time = time.time() * 1000 - self.time_start # ms
        self.e_time = e.timestamp() - self.time_start_e # ms
        self.dots.append((self.i_stroke, self.time, self.e_time, x, y, xT, yT, p))
        if self.print_activity:
            print('len {} | i_stroke {} | (t,x,y,xT,yT,pres) = ({:.2f}, {:.2f}, {:.2f}, {:d}, {:d}, {:.2f})'.format(
                len(self.dots), self.i_stroke, self.time, x, y, xT, yT, p
            ))

        # START: Painter session.
        self.painter = QtGui.QPainter(self.pixmap)

        # 1. Set the pen.
        pen = self.painter.pen()
        self.configure_pen(pen)
        self.painter.setPen(pen)

        # 2. Draw a line from the previous position to the current one.
        self.painter.drawLine(int(self.x_prev), int(self.y_prev), int(x), int(y))

        # END: Painter session.
        self.painter.end()

        #self.update() # Revive this if any problem occurs.

        # Display painter information on the pixel map.
        self.setPixmap(self.pixmap)

        # Update the origin for next time.
        self.x_prev = x
        self.y_prev = y

    def print_saved_dots(self):
        print('=====[Saved dots]=====')
        for i, dot in enumerate(self.dots):
            print('index {} | i_stroke {} | (t,x,y,xT,yT,pres) = ({:.2f}, {:.2f}, {:.2f}, {:d}, {:d}, {:.2f})'.format(
                i, dot[0], dot[1], dot[2], dot[3], dot[4], dot[5], dot[6]
            ))

    def saveDraw(self, target_character, save_dir, user_id, save_canvas_img=False):
        if len(self.dots) == 0:
            print('No dots to be saved! Draw it!')
            return
        os.makedirs(save_dir, exist_ok=True)
        time_in_second = time.time()
        time_string = datetime.datetime.fromtimestamp(time_in_second).strftime('%Y-%m-%d-%H%M%S')
        file_name = '{}_{}x{}_{}_{}.csv'.format(
            time_string, self.width, self.height, target_character, user_id)
        data_path = os.path.join(save_dir, file_name)
        with open(data_path,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['i_stroke', 'time_ms', 'e_time_ms', 'x', 'y', 'xTilt', 'yTilt', 'pressure'])
            writer.writerows(self.dots)
        print('Data are saved in {} !'.format(data_path))
        if save_canvas_img:
            if target_character in string.digits:
                path = '{}.png'.format(target_character)
            elif target_character in string.ascii_lowercase:
                path = '{}-lower.png'.format(target_character)
            elif target_character in string.ascii_uppercase:
                path = '{}-upper.png'.format(target_character)
            path = os.path.join(save_dir, path)
            self.save_canvas(path)
            print('The canvas image is saved in {} !'.format(path))

    def draw_guide_trajectory(self, character):
        if character == '@':
            self.draw_guide_spiral()
            return

        painter = QtGui.QPainter(self.pixmap)
        pen = painter.pen()
        pen.setColor(QtGui.QColor('#D3D3D3')) # light gray
        painter.setPen(pen)
        font = QtGui.QFont(self.guide_font, int(self.height/self.v_guidelines*(self.v_guidelines-1)))
        #font.setStyleSheet(QtGui.QColor('#D3D3D3')) # light gray
        painter.setFont(font)
        painter.drawText(
            0,          int(                  self.height/self.v_guidelines - 0.5*self.height/self.v_guidelines),
            self.width, int(self.v_guidelines*self.height/self.v_guidelines - 0.5*self.height/self.v_guidelines),
            Qt.AlignmentFlag.AlignCenter,
            character)
        painter.end()
        self.setPixmap(self.pixmap)

    def draw_guide_spiral(self):
        painter = QtGui.QPainter(self.pixmap)

        pen = painter.pen()
        pen.setWidth(self.pen_width)
        pen.setColor(QtGui.QColor('#D3D3D3')) # light gray
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        n_periods = 4
        max_radius = min(
            (self.height - self.height / self.v_guidelines / 2 ) / 2,
            (self.width - self.width / self.h_guidelines / 2) / 2)
        radians = np.arange(0, n_periods*2*np.pi, np.pi / 180)
        radiuses = np.linspace(0, max_radius, len(radians))
        x_center = self.width / 2
        y_center = self.height / 2

        for radian1, radius1, radian2, radius2 in zip(radians[:-1], radiuses[:-1], radians[1:], radiuses[1:]):
            x1 = radius1 * np.cos(radian1) + x_center
            y1 = radius1 * np.sin(radian1) + y_center
            x2 = radius2 * np.cos(radian2) + x_center
            y2 = radius2 * np.sin(radian2) + y_center

            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        painter.end()
        self.setPixmap(self.pixmap)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, samples, canvas_width, canvas_height, save_dir, user_id, image_guidance, trj_guidance, print_dots=False):
        super().__init__()
        self.samples = samples
        self.n_total_samples = len(samples)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.save_dir = save_dir
        self.user_id = user_id
        self.image_guidance = image_guidance
        self.trj_guidance = trj_guidance
        self.print_dots = print_dots

        self.size_btn = QtCore.QSize(int(canvas_width / 2), int(canvas_height))

        # Create the saving button.
        self.btn_save = QPushButton('Save')
        self.btn_save.setFixedSize(self.size_btn)
        self.btn_save.pressed.connect(self.save)

        # Create the message label.
        self.messeage = QLabel()
        self.messeage.setFont(QtGui.QFont('Arial', 12))
        self.messeage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_message()

        # Create the label to show a character to draw.
        self.char_write = QLabel()
        self.char_write.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create the clearing button.
        self.btn_clear = QPushButton('Clear')
        self.btn_clear.setFixedSize(self.size_btn)
        self.btn_clear.pressed.connect(self.clear)

        # Create the drawing canvas.
        self.canvas = Canvas(canvas_width, canvas_height, trj_guidance)

        self.set_new_character()

        # Construct the whole QtWidgets and add component widgets.
        w = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.messeage, 0, 0, 1, 3)
        layout.addWidget(self.char_write, 1, 1, 1, 1)
        layout.addWidget(self.btn_save, 2, 0, 1, 1)
        layout.addWidget(self.canvas, 2, 1, 1, 1)
        layout.addWidget(self.btn_clear,2, 2, 1, 1)
        w.setLayout(layout)

        # Set the widget as the central widiget.
        self.setCentralWidget(w)

    def save(self):
        if len(self.canvas.dots) == 0:
            print('There are no data to save.')
            return
        self.canvas.saveDraw(self.target_character, self.save_dir, self.user_id)
        if self.print_dots:
            self.canvas.print_saved_dots()
        self.canvas.reset()
        if len(self.samples) == 0:
            print('All data have been collected. This program is being closed.')
            exit()
        self.update_message()
        self.set_new_character()

    def clear(self):
        self.canvas.reset()

    def set_new_character(self):
        self.target_character = self.samples.pop()
        self.canvas.set_character(self.target_character)
        if self.target_character in string.ascii_lowercase:
            path = 'guide_images/{}-lower.png'.format(self.target_character)
        elif self.target_character in string.ascii_uppercase:
            path = 'guide_images/{}-upper.png'.format(self.target_character)
        else:
            path = 'guide_images/{}.png'.format(self.target_character)
        if self.image_guidance and os.path.exists(path):
            self.char_write.setPixmap(QtGui.QPixmap(path).scaledToWidth(int(self.canvas_width/3)))
        else:
            # Use text if the guide image does not exist.
            self.char_write.setFont(QtGui.QFont('Helvetica', 44))
            self.char_write.setText(self.target_character)

        if self.trj_guidance:
            self.canvas.draw_guide_trajectory(self.target_character)

    def update_message(self):
        self.messeage.setText('Drawn samples : {}/{} ({}%)'.format(
            self.n_total_samples - len(self.samples), self.n_total_samples, int((self.n_total_samples - len(self.samples)) / self.n_total_samples * 100)
        ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a data collector for Dynamic MNIST.')
    parser.add_argument('--user', type=str, required=True,
        help='Enter who will write down characters.')
    parser.add_argument('--hand', type=str, choices={'R', 'L'}, required=True,
        help='Which hand the user operates: R or L.')
    parser.add_argument('--weak_hand', action='store_true',
        help='Is the hand weak or dominant?')
    parser.add_argument('--save_dir', type=str, default='data_vector',
        help='Path of a directory where stroke files are saved.')
    parser.add_argument('--width', type=int, default=256,
        help='Width of the drawing canvas.')
    parser.add_argument('--height', type=int, default=256,
        help='Height of the drawing canvas.')
    parser.add_argument('--character_type', type=str,
        default='0123456789',
        #default='abcdefghijklmnopqrstuvwxyz',
        #default='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        #default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
        help='What types of characters to collect given in a string. @ is assumed to be a spiral.')
    parser.add_argument('--samples_per_character', type=int, default=1,
        help='How many samples to collect per character.')
    parser.add_argument('--image_guidance', action='store_true',
        help='Show images instead of text as a guidance.')
    parser.add_argument('--trj_guidance', action='store_true',
        help='Show a trajectory as a guidance on the canvas. A spiral will appear if @ is given.')
    parser.add_argument('--print_dots', action='store_true',
        help='Print all recorded dots after these are saved.')
    parser.add_argument('--shuffle', action='store_true',
        help='Print all recorded dots after these are saved.')
    args = parser.parse_args()

    # Create a user ID that identifies the subject of motor control.
    weak_hand_tag = 'W' if args.weak_hand else 'D'
    user_id = '{}_{}{}'.format(args.user.lower(), args.hand.upper(), weak_hand_tag)

    # Make a sequence of shuffled character samples to be collected.
    sample_types = sorted(list(args.character_type))
    samples = sample_types * args.samples_per_character
    if args.shuffle:
        random.shuffle(samples)
    else:
        samples = sorted(samples, reverse=True)

    # Define the window and its components
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('icon.png'))
    window = MainWindow(samples, args.width, args.height, args.save_dir, user_id, args.image_guidance, args.trj_guidance, args.print_dots)
    window.setWindowTitle('Dynamic MNIST Collector')
    window.show()
    window.setFixedSize(window.width(), window.height())

    # Position at the center of the screen.
    screen_size = app.primaryScreen().size()
    window.setGeometry(
        screen_size.width()//2-window.width()//2,
        screen_size.height()//2-window.height()//2,
        window.width(),
        window.height())

    # Run the QtWidgets application.
    app.exec()
