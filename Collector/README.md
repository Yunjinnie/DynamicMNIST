# Dynamic MNIST Collector

Through this software, you can collect hand-written character of digits (0-9), lowercase alphabets (a-z), and uppercase alphabets (A-Z), which are the classes of the MNIST and EMNIST dataset.
Each sample contain trajectories of a hand-written character in the format of a sequence of `(i_stroke, time_ms, x, y, xTilt, yTilt, pressure)`.

- `i_stroke`:
  The ordered identity of a stroke, which starts from 0.
  This tells us what stroke a dot is included in.
- `time_ms`:
  The time when a dot is written.
  Its unit is millisecond.
- `x`:
  The horizontal location of a dot.
  The left-most value is 0.
- `y`:
  The vertical location of a dot.
  The top-most value is 0.
- `xTilt`:
  The angle between the z axis and the pen direction projected onto the yz plane.
  This value ranges between 0 and 60, the resolution of which is 1 degree (device: WACOM Intuos Pro S; PTH-460).
- `yTilt`:
  The angle between the z axis and the pen direction projected onto the xz plane.
  This value ranges between 0 and 60, the resolution of which is 1 degree.
- `pressure`:
  The pressure of the pen is used.
  This value ranges between 0 and 1, the resolution of which is 1/8192 .


You can collect this kind of data using the dynamic MNIST collector by running `main_qt.py`.
You can also export those collected data into raster images in the formats of `NPY` and `GIF`, by running `rasterize.py`.

## Install
```
pip install PyQt6
```

## Tutorial 1: Dynamic MNIST collector

[YouTube demo](https://youtu.be/02N3cdvcTAw)

To start this collector, run the following command.
You have to give the name of a user who will draw characters.
You can optionally set the width and height of the canvas.
The default value of the two is set 256.
You can set how many samples to collect per character using the `samples_per_character` option (default: 10).

Use the PyQt6 version to acquire data with pen tilt and pressure as well as less latency.

```bash
# Prototype
python main_qt.py --user <user_name> --hand <R or L> [--character_type <character_string>] [--weak_hand] [--save_dir <dir_to_save_results>] [--width <canvas_width>] [--height <canvas_height>] [--samples_per_character <samples_per_character>] [--print_dots]
# Example
python main_qt.py --user sungjae --hand R --character_type 0123456789 --save_dir data_vector_example --width 256 --height 256 --samples_per_character 10
```

The following is implemented by `Tkinter`.

```bash
# Prototype
python main.py --user <user_name> --hand <R or L> [--weak_hand] [--save_dir <dir_to_save_results>] [--width <canvas_width>] [--height <canvas_height>] [--samples_per_character <samples_per_character>] [--print_dots]
# Example
python main.py --user sungjae --hand R --save_dir data_vector_example --width 256 --height 256 --samples_per_character 10
```


A window below pops up.
You can draw a character on the white canvas below.
If yours is not properly drawn, you can clear it out through the clear button.
If the cursor get out of the canvas during drawing, your strokes will be cleared out.
If you think your character is properly drawn, then save your drawing with the save button.
Then, the character to write will be updated, which is randomly chosen out of 10 digits.

<center><img src="screenshot_qt.png" width="300" align="center"></center>

The drawn data are saved in the `data_vector_qt_example` directory.
The name is given in the format of `{%Y-%m-%d-%H%M%S}_{width_canvas}x{height_canvas}_{target_character}_{user}_{hand_RorL}{weak_or_dominant_hand_WorD}.csv`.
The headers of the CSV file are `i_stroke`, `time_ms`, `x`, `y`, `xTilt`, `yTilt`, and `pressure`.
You can check that example from `data_vector_qt_example/2022-12-31-163525_256x256_9_sungjae_RD.csv`.

## DynamicMNIST-Data

In [DynamicMNIST-Data](https://github.com/lab-taco/DynamicMNIST-Data), you can
- find data collected by this DynamicMNIST-Data,
- rasterize vector images, and
- convert vector images to velocity sequences.

## Reference
- In writing `main_qt.py`, the following was referenced: [https://www.pythonguis.com/tutorials/pyqt6-bitmap-graphics/](https://www.pythonguis.com/tutorials/pyqt6-bitmap-graphics/).
- In writing `main.py`, the following was referenced: [github.com/swaroopmaddu/Paint-Tkinter](https://github.com/swaroopmaddu/Paint-Tkinter).
- `icon.png` is from [flaticon.com/free-icon/hand-writing_1617609](https://www.flaticon.com/free-icon/hand-writing_1617609).
