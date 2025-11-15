

"""Build dataset of complex objects from images.

For each image in `datasets/base` and `datasets/sparse` this script:
 - loads the image as grayscale (OpenCV),
 - center-crops (or pads) to 512x512,
 - generates a Zernike phase map (using the project's utility),
 - combines amplitude and phase into a complex field,
 - saves results as compressed .npz files under a dated output directory
   grouped into `case_1` .. `case_10`.

Run from the repository root (so `datasets/` is visible) or pass `--datasets-root`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import random as _random

import contextlib
import jax.numpy as jnp
import cv2 as cv
import numpy as np

import data_generation_utils as dgu


DEFAULT_IMAGE_SIZE = 512
NUM_TEST_CASES = 15


def save_near_field(out_dir: Path, near_field: jnp.ndarray, spot_center: np.ndarray, img_path: Path) -> None:
    """Save near-field magnitude to compressed .npz file."""

    # Ensure output dir exists.
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{img_path.stem}.npz"
    out_path = out_dir / out_name
    
    np.savez_compressed(
        str(out_path),
        near_field=near_field.astype(np.complex128),
        filename=str(img_path.name),
        spot_center=spot_center.astype(np.int32),
    )


@contextlib.contextmanager
def load_near_field_from_image(img_path: Path, output_dir: Path, image_size: int = DEFAULT_IMAGE_SIZE):
    """Context manager that loads an image, center-crops to image_size and yields the near-field magnitude.

    Usage:
        with load_near_field_from_image(path, image_size, output_dir) as magnitude:
            # use magnitude (jax array)

    If output_dir is provided it will be created (parents=True).
    """
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Warning: couldn't read image {img_path}, skipping.")

    # Crop center.
    h, w = img.shape[:2]
    start_h = (h - image_size) // 2
    start_w = (w - image_size) // 2
    cropped_image = img[start_h : start_h + image_size, start_w : start_w + image_size]
    
    # Normalize and convert units.
    intensity = cropped_image / np.max(cropped_image)
    near_field = np.sqrt(intensity).astype(np.complex128)

    # Generate a location for the bright spot.
    spot_center = np.asarray([np.random.randint(0, image_size), np.random.randint(0, image_size)])

    try:
        yield near_field, spot_center
    finally:
        save_near_field(output_dir, near_field, spot_center, img_path)


def build_dataset(datasets_root: Path, output_root: Path, image_size: int) -> None:
    # input directories
    regular_dir = datasets_root / "regular"
    sparse_dir = datasets_root / "sparse"

    # create dated output base: include size
    date = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
    sizedir = f"{image_size}x{image_size}"
    output_base = output_root / f"dataset_generated_{date}_{sizedir}"

    print(f"Output base directory: {output_base}")

    if not regular_dir.exists() or not sparse_dir.exists():
        raise FileNotFoundError(f"Input directories {regular_dir} and/or {sparse_dir} do not exist.")

    # list image files
    regular_imgs = sorted([p for p in regular_dir.iterdir() if p.is_file()])
    sparse_imgs = sorted([p for p in sparse_dir.iterdir() if p.is_file()])
    if not regular_imgs or not sparse_imgs:
        raise FileNotFoundError(f"No files found in {regular_dir} or {sparse_dir}")

    for case in range(1, NUM_TEST_CASES + 1):
        case_dir = output_base / f"case_{case}"

        if case == 1:
            ## Case 1: Sanity Test ##

            # Generate real and imaginary parts from N(0, 1)
            real_part = np.random.randn(image_size, image_size)
            imag_part = np.random.randn(image_size, image_size)

            # Generate a location for the bright spot.
            spot_center = np.asarray([np.random.randint(0, image_size), np.random.randint(0, image_size)])

            # Add bright spot.
            magnitude = dgu.add_delta_spot(real_part, x0=spot_center[0], y0=spot_center[1], radius=1, magnitude_multiplier=10)

            # Remove phase at spot.
            imag_part[spot_center[0], spot_center[1]] = 0.0

            # Combine into a complex-valued array
            complex_map = real_part + 1j * imag_part

            save_near_field(case_dir, complex_map, spot_center, img_path=Path("1"))

        if case == 2:
            ## Case 2: Generic Images ##

            for img_path in regular_imgs:
                with load_near_field_from_image(img_path, case_dir, image_size=image_size) as (near_field, spot_center):

                    # Generate phase map.
                    phase_map = dgu.generate_zernike_phase_map((image_size, image_size), aperature="cropped")

                    # Add bright spot.
                    magnitude = dgu.add_delta_spot(np.real(near_field), x0=spot_center[0], y0=spot_center[1], radius=1, magnitude_multiplier=1)

                    # Remove phase at spot.
                    phase_map[spot_center[0], spot_center[1]] = 0.0

                    # Collect near field and complex object.
                    near_field[:] = magnitude * np.exp(1j * phase_map)

        elif case == 3:
            ## Case 3: Constant Images ##

            for i in range(1, 6):
                magnitude = np.full((image_size, image_size), np.random.randn())

                # Generate phase map
                phase_map = dgu.generate_zernike_phase_map((image_size, image_size), aperature="cropped")

                # Generate a location for the bright spot.
                spot_center = np.asarray([np.random.randint(0, image_size), np.random.randint(0, image_size)])
            
                # Add bright spot.
                magnitude = dgu.add_delta_spot(magnitude, x0=spot_center[0], y0=spot_center[1], radius=1, magnitude_multiplier=1)

                # Remove phase at spot.
                phase_map[spot_center[0], spot_center[1]] = 0.0
                
                # Collect near field and complex object.
                near_field = magnitude * np.exp(1j * phase_map)
                save_near_field(case_dir, near_field, spot_center=spot_center, img_path=Path(str(i)))

        elif case == 4:
            ## Case 4: Sparse Images ##

            for img_path in sparse_imgs:
                with load_near_field_from_image(img_path, case_dir, image_size=image_size) as (near_field, spot_center):

                    # Generate phase map
                    phase_map = dgu.generate_zernike_phase_map((image_size, image_size), aperature="cropped")

                    # Add bright spot.
                    magnitude = dgu.add_delta_spot(np.real(near_field), x0=spot_center[0], y0=spot_center[1], radius=1, magnitude_multiplier=1)

                    # Remove phase at spot.
                    phase_map[spot_center[0], spot_center[1]] = 0.0

                    # Collect near field and complex object.
                    near_field[:] = magnitude * np.exp(1j * phase_map)

        elif case == 5:
            ## Case 5: Sparse Images with Phase Pattern ##

            pass

        elif case == 6:
            ## Case 6: Constant Intensity and Phase Object ##

            with load_near_field_from_image(regular_imgs[0], case_dir, image_size=image_size) as (near_field, spot_center):

                magnitude = np.full((image_size, image_size), np.random.randn())
            
                # Generate phase map and duplicate the magnitude into the phase.
                phase_map = dgu.generate_zernike_phase_map((image_size, image_size), aperature="cropped") + np.real(near_field)

                # Add bright spot.
                magnitude = dgu.add_delta_spot(magnitude, x0=spot_center[0], y0=spot_center[1], radius=1, magnitude_multiplier=1)

                # Remove phase at spot.
                phase_map[spot_center[0], spot_center[1]] = 0.0

                # Collect near field and complex object.
                near_field[:] = magnitude * np.exp(1j * phase_map)

        elif case == 7:
            ## Case 7: Generic Image and Phase Object ##

            with load_near_field_from_image(regular_imgs[0], case_dir, image_size=image_size) as (near_field, spot_center):

                # Generate phase map and duplicate the magnitude into the phase.
                phase_map = dgu.generate_zernike_phase_map((image_size, image_size), aperature="cropped") + np.real(near_field)

                # Add bright spot.
                magnitude = dgu.add_delta_spot(np.real(near_field), x0=spot_center[0], y0=spot_center[1], radius=1, magnitude_multiplier=1)

                # Remove phase at spot.
                phase_map[spot_center[0], spot_center[1]] = 0.0

                # Collect near field and complex object.
                near_field[:] = magnitude * np.exp(1j * phase_map)

        elif case == 8:
            ## Case 8: Generic Image with Weak Bright Spot ##

            with load_near_field_from_image(regular_imgs[0], case_dir, image_size=image_size) as (near_field, spot_center):

                # Generate phase map.
                phase_map = dgu.generate_zernike_phase_map((image_size, image_size), aperature="cropped")

                # Add bright spot.
                magnitude = dgu.add_delta_spot(np.real(near_field), x0=spot_center[0], y0=spot_center[1], radius=1, magnitude_multiplier=0.1)

                # Remove phase at spot.
                phase_map[spot_center[0], spot_center[1]] = 0.0

                # Collect near field and complex object.
                near_field[:] = magnitude * np.exp(1j * phase_map)

        elif case == 9:
            ## Case 9: Generic Image with Gaussian Bright Spot ##
            
            for img_path in sorted(regular_imgs)[:5]:
                with load_near_field_from_image(img_path, case_dir, image_size=image_size) as (near_field, spot_center):

                    # Generate phase map.
                    phase_map = dgu.generate_zernike_phase_map((image_size, image_size), aperature="cropped")

                    # Add bright spot.
                    magnitude = dgu.add_gaussian_spot(np.real(near_field), x0=spot_center[0], y0=spot_center[1], magnitude_multiplier=10)

                    # Remove phase at spot region.
                    phase_map[spot_center[0] - 5: spot_center[0] + 5, spot_center[1] - 5: spot_center[1] + 5] = 0.0

                    # Collect near field and complex object.
                    near_field[:] = magnitude * np.exp(1j * phase_map)

    print("Done building dataset.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build complex-field dataset from images")
    parser.add_argument("--datasets-root", type=str, help="Path to datasets root (contains base/ and sparse/)")
    parser.add_argument("--output-root", type=str, help="Root directory to write generated datasets under")
    parser.add_argument("--size", type=int, default=DEFAULT_IMAGE_SIZE, help="Output crop size (default 512 512)")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_root = Path(args.datasets_root)
    output_root = Path(args.output_root)
    image_size = int(args.size)

    build_dataset(datasets_root, output_root, image_size=image_size)


if __name__ == "__main__":
    main()
