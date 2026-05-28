import tkinter
from tkinter import *
from tkinter.ttk import *
from tkinter.colorchooser import askcolor
import datetime
import time
import random
import os
import csv
import argparse

# Global variables ####################
pen = {'color':'black', 'size':5} # Pen configuration
samples_types = list(range(10)) # 10 characters (from 0 to 9).
i_stroke = -1 # ID for the current stroke. Started from 0.
lastx, lasty = None, None # location (x,y) of the last dot.
dots = list() # Where dots are stored.

def usepen():
    window.config(cursor='dot')
    cv.bind('<1>', activate_paint) # When left mouse button's clicked, call activate_paint.
    cv.place(x=0,y=h_b,width=w_cv,height=h_cv)

def activate_paint(e):
    global lastx, lasty, t_start, t_current, dots, i_stroke
    if len(dots) == 0:
        t_start = time.time() * 1000 # unit: ms.
        t_current = 0
    else:
        t_current = time.time() * 1000 - t_start # unit: ms.
    lastx, lasty = e.x, e.y
    cv.bind('<B1-Motion>', paint) # When left mouse button's clicked & cursor moves, call paint.
    i_stroke += 1
    dots.append((i_stroke, t_current, e.x, e.y))

def paint(e):
    global lastx, lasty, h_cv, w_cv
    t_current = time.time() * 1000 - t_start # unit: ms.
    if (e.x >= h_cv) or (e.x < 0) or (e.y >= w_cv) or (e.y < 0):
        reset()
        return
    dots.append((i_stroke, t_current, e.x, e.y))
    tag = str((i_stroke, t_current, e.x, e.y))
    cv.create_line((lastx, lasty, e.x, e.y),
        width=pen['size'], fill=pen['color'], smooth=True, splinesteps=30, tags=tag)
    lastx, lasty = e.x, e.y

def reset():
    global dots, i_stroke
    cv.delete('all')
    dots.clear()
    i_stroke = -1

def resetDraw():
    global label_character, label_samples, target_character
    print('{} out of {} samples ({}%) are left.'.format(
        len(samples), n_total_samples, int(len(samples)/n_total_samples*100)))
    text_samples = '{}/{}'.format(
        len(samples), n_total_samples, int(len(samples)/n_total_samples*100))
    label_samples.configure(text=text_samples)
    target_character = samples.pop()
    label_character.configure(text=target_character)

def saveDraw():
    global h_cv, w_cv
    if len(dots) == 0:
        print('No dots to be saved! Draw it!')
        return
    if dots[0][0] == -1: # if i_stroke of the first dot is -1
        print('Data have been corrupted! Draw it again!')
        reset()
        return
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')
    file_name = '{}_{}x{}_{}_{}_{}{}.csv'.format(
        st, w_cv, h_cv, target_character, args.user,
        args.hand.upper(), weak_hand_tag)
    data_path = os.path.join(args.save_dir, file_name)
    with open(data_path,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['i_stroke', 'time_ms', 'x', 'y'])
        writer.writerows(dots)
    print('Data are saved in {} !'.format(data_path))
    if args.print_dots:
        print_dots(dots)
    if len(samples) == 0:
        print('All data have been collected. This program is being closed.')
        exit()
    resetDraw()
    reset()

def print_dots(dots):
    print('=====[Saved dots]=====')
    for dot in dots:
        i_stroke, t_current, x, y = dot
        print('(i_stroke,t,x,y) | len(dots): ({:02d}, {:4f} ms, {:3d}, {:3d}) | {}'.format(i_stroke, t_current, x, y, len(dots)))

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
    parser.add_argument('--samples_per_character', type=int, default=10,
        help='How many samples to collect per character.')
    parser.add_argument('--print_dots', action='store_true',
        help='Print all recorded dots after these are saved.')
    args = parser.parse_args()

    global weak_hand_tag
    weak_hand_tag = 'W' if args.weak_hand else 'D'

    # Define the window and its components
    window = tkinter.Tk()
    window.title('Dynamic MNIST Collector')
    photo = PhotoImage(file='icon.png')
    window.iconphoto(False, photo)

    h_scr = window.winfo_screenheight()
    w_scr = window.winfo_screenwidth()
    h_cv, w_cv = args.height, args.width # canvas width and height
    h_b, w_b = 128, (w_cv // 3)
    h_wd, w_wd = h_cv + h_b, w_cv
    x_wd, y_wd = (w_scr - w_wd) // 2, (h_scr - h_wd) // 2

    window.geometry('%dx%d+%d+%d' % (w_wd, h_wd, x_wd, y_wd))

    # Make a sequence of shuffled character samples to be collected.
    samples = samples_types * args.samples_per_character
    n_total_samples = len(samples)
    random.shuffle(samples)
    print('{} out of {} samples ({}%) are left.'.format(
        len(samples), n_total_samples, int(len(samples)/n_total_samples*100)))

    # Button to clear the canvas and recorded data.
    button_clear = tkinter.Button(window, text='Clear', foreground='black', width=w_b, height=h_b, command=reset)
    button_clear.place(x=0,y=0,width=w_b,height=h_b)

    # Label to disply how many samples to collect are left.
    text_samples = '{}/{}'.format(
        len(samples), n_total_samples)
    label_samples = Label(window, text=text_samples, foreground='black', anchor='n')
    label_samples.place(x=w_b,y=0,width=w_b,height=h_b//5)

    # Label for instruction.
    label_instruction = Label(window, text='Write down', foreground='black', anchor='n')
    label_instruction.place(x=w_b,y=h_b//5,width=w_b,height=h_b//5)

    # Take out a sample from the sample sequence.
    target_character = samples.pop()

    # Label to show what character to write down on the canvas.
    label_character = Label(window, text=target_character, foreground='black')
    label_character.config(font=('Courier', 44), anchor=CENTER)
    label_character.place(x=w_b,y=h_b//5*2,width=w_b,height=h_b//5*3)

    # Button to save trajectoies on the canvas.
    button_save = tkinter.Button(window, text='Save', foreground='black', width=w_b, height=h_b, command=saveDraw)
    button_save.place(x=2*w_b,y=0,width=w_b,height=h_b)

    # Initialize and put a canvas on the window.
    cv = Canvas(window, bg='white', width=w_cv, height=h_cv)
    cv.place(x=0,y=h_b,width=w_cv,height=h_cv)

    # Activate the pen mode.
    usepen()

    # Activate the window interface.
    window.config()
    window.mainloop()
