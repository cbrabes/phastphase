import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import data, filters, feature
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage import morphology

import phastphase
from phastphase.retrieval_jax import retrieve
import phasephase_helper_funcs as help_funcs


def plot_farfield(reconstruction,far_field):
  fft = help_funcs.Fourier(reconstruction)
  input = far_field**2
  input = input / np.sum(np.abs(input))
  reconstructed = np.abs(fft)**2
  reconstructed = reconstructed/ np.sum(np.abs(reconstructed))

  fig,ax = plt.subplots(2,3, figsize = (12,6))
  ax[0,0].imshow(input[0:300,0:300])
  ax[0,0].set_title('Input')
  ax[0,1].imshow(reconstructed[0:300,0:300])
  ax[0,1].set_title('Reconstruction')
  cax1 = ax[0,2].imshow(reconstructed[0:300,0:300] - input[0:300,0:300])
  ax[0,2].set_title('Diff')
  ax[1,0].imshow(far_field[300:500,300:500]**2)
  ax[1,0].set_title('Input')
  ax[1,1].imshow(np.abs(fft[300:500,300:500])**2)
  ax[1,1].set_title('Reconstruction')
  cax2 = ax[1,2].imshow(reconstructed[300:500,300:500]- input[300:500,300:500])
  ax[1,2].set_title('Diff')

  divider1 = make_axes_locatable(ax[0,2])
  divider2 = make_axes_locatable(ax[1,2])
  cbar_ax1 = divider1.append_axes("right", size="5%", pad=0.05)
  cbar_ax2 = divider2.append_axes("right", size="5%", pad=0.05)

  fig.colorbar(cax1, cax=cbar_ax1, orientation='vertical')
  fig.colorbar(cax2, cax=cbar_ax2, orientation='vertical')
  for b in range(2):
    for a in ax[b,:]:
      a.set_xticks([])
      a.set_yticks([])
  return 1


def align_global_phase(x_true, x_rec):
    # Calculate the inner product (complex)
    inner_product = jnp.sum(jnp.conj(x_true) * x_rec)
    # Get the phase of that inner product
    phase_shift = jnp.sign(inner_product)
    # Rotate the reconstruction to match the truth
    return x_rec / phase_shift


def run_phastphase(
    ground_truth,
    far_field_oversampled,
    support_mask,
    max_iters=1000,
    descent_method=0,
    grad_tolerance=1e-8,
    wind_method=0,
    winding_guess=None,
):             
    x_out, val = retrieve(
        far_field_oversampled,
        support_mask,
        max_iters=max_iters,
        descent_method=descent_method,
        grad_tolerance=grad_tolerance,
        wind_method=wind_method,
        winding_guess=winding_guess,
    )

    # Normalization (Phase Retrieval ambiguity handling)
    # Note: In real phase retrieval, you might need to align the global phase 
    # (e.g., x_out * jnp.exp(-1j * angle)) before comparing. 
    # Here we just normalize norms for simplicity.
    x_out = x_out / jnp.linalg.norm(x_out)
    x_gt = ground_truth / jnp.linalg.norm(ground_truth)

    # FFT is invariant to global phase,
    # so we need to cancel relative angle between input and output before comparison. 
    x_out = align_global_phase(x_true=x_gt, x_rec=x_out)

    # Error Calculation (Relative Error)
    err = jnp.linalg.norm(x_out - x_gt) / jnp.linalg.norm(x_gt)

    return x_out, err


def expand_circles(mask, main_scale_factor=1.2, reference_scale_factor=2):
    """
    Finds circles in a binary mask and returns a new mask 
    with each circle scaled up by the given factor.
    """
    # 1. Label the objects (Find the 5 distinct islands)
    # labeled_array has 1s for object 1, 2s for object 2, etc.
    labeled_array, num_features = ndimage.label(mask)
    
    print(f"Found {num_features} circles.")
    
    # Create an empty canvas for the result
    new_mask = np.zeros_like(mask)
    
    # Create coordinate grids for drawing circles later
    h, w = mask.shape
    Y, X = np.ogrid[:h, :w]
    
    # Find the main circle (largest area)
    areas = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
    main_circle_label = np.argmax(areas) + 1  # +1

    # 2. Iterate through each object
    for i in range(1, num_features + 1):
        # Create a boolean mask for just this single object
        obj_mask = (labeled_array == i)
        
        # A. Find Center of Mass
        cy, cx = ndimage.center_of_mass(obj_mask)
        
        # B. Find Current Radius
        # We estimate radius from Area: Area = pi * r^2  ->  r = sqrt(Area / pi)
        area = np.sum(obj_mask)
        current_radius = np.sqrt(area / np.pi)
        
        # C. Calculate New Radius
        new_radius = current_radius * (main_scale_factor if i == main_circle_label else reference_scale_factor)
        
        # D. Draw the new bigger circle on the canvas
        # Formula: (x-cx)^2 + (y-cy)^2 <= r^2
        dist_from_center = (Y - cy)**2 + (X - cx)**2
        circle_mask = dist_from_center <= new_radius**2
        
        # Add this circle to our final mask
        new_mask[circle_mask] = 1.0
        
    return new_mask


def recorrelate(recovered_mask):
    # 3. Propagate to Far Field (FFT)
    # We use fftshift to handle the center-origin correctly
    far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(recovered_mask)))

    # 4. Measure Intensity (Simulate Camera)
    measured_intensity = np.abs(far_field)**2

    # 5. Compute Autocorrelation (Validation)
    # This is what you compare to your experimental result
    autocorr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(measured_intensity)))
    abs_autocorr = np.abs(autocorr)

    log_recorrelated_mask = np.abs(np.log(abs_autocorr))  # Adding a small constant to avoid log(0)
    recorrelated_threshold_value = filters.threshold_li(log_recorrelated_mask)
    recorrelated_binary_image = log_recorrelated_mask < recorrelated_threshold_value  # Threshold the image
    recorrelated_binary_image = recorrelated_binary_image.astype(float)
    return recorrelated_binary_image


def find_max_overlap_offset(mask1, mask2):
    """
    Finds the (y, x) shift required to align mask2 with mask1 
    such that their overlap is maximized.
    """
    # 1. Calculate Cross-Correlation
    # 'full' mode ensures we calculate overlap even at the edges
    # 'fft' method is much faster for large arrays
    correlation = signal.correlate(mask1, mask2, mode='full', method='fft')

    # 2. Find the position of the maximum value in the correlation matrix
    # This value represents the highest number of overlapping pixels
    max_overlap_value = np.max(correlation)
    y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)

    # 3. Convert the peak index to a relative shift
    # The center of the correlation matrix corresponds to "zero shift"
    # Formula: Shift = Peak_Index - (Shape_of_Reference_Image - 1)
    y_shift = y_peak - (mask1.shape[0] - 1)
    x_shift = x_peak - (mask1.shape[1] - 1)

    return y_shift, x_shift, max_overlap_value


def crop_far_field(far_field_R, far_field_L):
   # Apply global thresholding using Otsu's method
    threshold_value = 0.01 #filters.threshold_otsu(far_field_R)
    blurred_image = gaussian(far_field_L, sigma=2, mode='reflect')
    binary_image = blurred_image > threshold_value  # Threshold the image

    # Apply Harris corner detection
    contour = feature.corner_harris(binary_image, method='eps', sigma=1)

    # Find coordinates of the detected corners
    coords = feature.corner_peaks(contour, min_distance=5)
    coords = help_funcs.sort_contour_points(coords)
    corners = help_funcs.find_corners(coords)

    miny = np.amin(corners[:, 0]) - 10#33
    minx = np.amin(corners[:, 1]) - 10#33
    maxx = np.amax(corners[:, 1]) + 10#33
    maxy = np.amax(corners[:, 0]) + 10#33

    far_field_R_cropped = far_field_R[miny:maxy,minx:maxx]
    far_field_L_cropped = far_field_L[miny:maxy,minx:maxx]
    
    return far_field_R_cropped, far_field_L_cropped


def recover_autocorrelated_mask(far_field):
    fft = help_funcs.Fourier(far_field**2)
    logfft = np.log(np.abs(fft))
    threshold_value = filters.threshold_li(logfft)
    binary_image = logfft > threshold_value  # Threshold the image
    return binary_image


def extract_mask_from_autocorrelation(autocorrelated_mask):
    spots = [0,1]

    props = regionprops(label(autocorrelated_mask))

    tot = np.zeros(np.shape(autocorrelated_mask))
    tot[autocorrelated_mask] = 1

    for spot in spots:
        tot = tot + help_funcs.set_center(autocorrelated_mask, props[spot].centroid)
    support = tot>2

    # Define a structuring element
    structuring_element = morphology.disk(0)  # Circular structuring element with radius 5

    # Perform dilation to enlarge the shapes
    support_large = morphology.dilation(support.astype(float), structuring_element)

    props1 = regionprops(label(support_large))
    y0,x0 = props1[2].centroid

    tight_support_box = help_funcs.bound_image(support_large)

    props_tight_support = regionprops(label(tight_support_box))
    y,x = props_tight_support[2].centroid

    reference_point = (int(y),int(x))
    support_rolled = np.roll(support_large, shift=int(y)-int(y0), axis=0)
    support_rolled = np.roll(support_rolled, shift=int(x)-int(x0), axis=1)

    return support_rolled, reference_point


def pad_for_far_field(near_field, far_field):
    target_shape = far_field.shape
    pad_shape = near_field.shape

    # Compute padding for each dimension and Apply padding
    pad_width = [(0, target_shape[i] - pad_shape[i]) for i in range(len(target_shape))]
    padded_matrix = np.pad(near_field, pad_width, mode='constant', constant_values=0)
    near_field_padded = np.zeros(np.shape(padded_matrix))
    near_field_padded[padded_matrix>0] = 1

    return near_field_padded
