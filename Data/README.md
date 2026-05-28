# Dynamic MNIST Data

This repository store data collected by the Dynamic MNIST Collector.
You can collect this kind of data using the dynamic MNIST collector by running `main.py`.
You can also export those collected data into raster images in the formats of `NPY` and `GIF`, by running `rasterize.py`.

## Tutorial 1: Rasterizing vector images into NPY and GIF

```bash
# Prototype
python rasterize.py --data_path <dir_vector_image> --save_dir <dir_to_save_results> [--sampling_length <sampling_length_in_millisecond>] [--stroke_width <stroke_width_in_pixel>] [--remove_hover] [--raw_movement] [--mnist]
# Example
python rasterize.py --data_path data_vector_example --save_dir data_raster_example --sampling_length 20 --stroke_width 10
```

Rasterized data are then saved in `data_raster_example`.
The name is given in the format of `{%Y-%m-%d-%H%M%S}_{width_canvas}x{height_canvas}_{target_character}_{user}_{hand_RorL}{weak_or_dominant_hand_WorD}_{stroke_width}px_{sampling_length}ms` with the extensions of `.npy` or `.gif`.
You can check those examples from `data_raster_example/2021-07-19-132706_256x256_4_sungjae_RD_10px_20ms.npy` and `2021-07-19-132706_256x256_4_sungjae_RD_10px_20ms.gif`.
You can also get the final GIF image in the format of PNG.

<center><img src="data_raster_example/2021-07-19-132706_256x256_4_sungjae_RD_smv_10px_20ms.gif" width="300" align="center"></center>

## Tutorial 2: Creating a dataset of the speed and direction vectors from trajectory samples

```bash
# Prototype
python get_velocity_data.py --dir_vector_data <dir_vector_data> --dir_velocity_data <dir_velocity_data> --dt <sampling_length_in_millisecond>
# Example
python get_velocity_data.py --dir_vector_data "dir_vector_data" --dir_velocity_data "dir_velocity_data" --dt 20
```
- `dir_vector_data`:
    The path where the positional vector images are stored.
- `dir_velocity_data`:
    The path where velocity data to be saved. The format of data is CSV.
    Its columns are `time_ms`, `dx`, `dy`, `direction`, `speed` and `hover`.
    The `time_ms` column contains the moments when velocities are sampled.
    You can exploit two representations of velocity: the Cartesian coordinate `(dx,dy)` and the polar one `(direction,speed)`.
    The `hover` column contains whether sampled movements are hovering (`1`) or not (`0`).
- `sampling_length_in_millisecond`: The time length in millisecond to sample the dots in trajectories. 20ms is the default value.
