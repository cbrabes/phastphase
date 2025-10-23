import jax.numpy as jnp
import cv2 as cv
import jax
import random
import numpy as np
import matplotlib.pyplot as plt
import zernike


def import_normalized_figure_data(figure_name):
    img_bgr = cv.imread(figure_name)
    img_hls = cv.cvtColor(img_bgr, cv.COLOR_BGR2HLS_FULL)
    hls_array = jnp.array(img_hls, dtype=jnp.float64)/255.0

    return hls_array


def convert_hls_to_complex(hsl_array):
    h_channel = hsl_array[:,:,0]
    l_channel = hsl_array[:,:,1]
    s_channel = hsl_array[:,:,2]

    magnitude = jnp.sqrt(l_channel)
    phase = h_channel * 2 * jnp.pi

    real_part = magnitude * jnp.cos(phase)
    imag_part = magnitude * jnp.sin(phase)

    complex_image = real_part + 1j * imag_part

    return complex_image, s_channel


def convert_complex_to_hls(complex_image, saturation_channel):
    real_part = jnp.real(complex_image)
    imag_part = jnp.imag(complex_image)

    magnitude = jnp.sqrt(real_part**2 + imag_part**2)
    phase = jnp.angle(complex_image)

    l_channel = jnp.clip(magnitude**2, 0.0, 1.0)
    h_channel = jnp.mod(phase / (2 * jnp.pi), 1.0)
    s_channel = jnp.clip(saturation_channel, 0.0, 1.0)

    hls_image = jnp.stack([h_channel, l_channel, s_channel], axis=-1)

    return hls_image


def import_complex_figure_data(figure_name):
    hls_array = import_normalized_figure_data(figure_name)
    complex_image, saturation_channel = convert_hls_to_complex(hls_array)

    return complex_image, saturation_channel


def plot_all(reconstruction, s = 0):
    image = np.array(reconstruction.copy())
    if s>0:
        [i,j] = np.unravel_index(np.argmax(np.abs(image)),np.shape(image))
        if i<s:
            i=s
        if i>np.shape(image)[0]-s:
            i=np.shape(image)[0]-s
        if j<s:
            j=s
        if j>np.shape(image)[1]-s:
            j=np.shape(image)[1]-s
        image[i-s:i+s,j-s:j+s] = 0
    fig , ax = plt.subplots(1,2, figsize = (8,4))
    im0 = ax[0].imshow(np.abs(image), cmap="gray")
    #plt.colorbar(im0, ax=ax[1], fraction=0.046, pad=0.04)
    ax[0].set_title('Amplitude')
    im1 = ax[1].imshow(np.angle(image),cmap="jet", interpolation="none")
    ax[1].set_yticks([])
    #plt.colorbar(im1, ax=ax[2], fraction=0.046, pad=0.04)
    ax[1].set_title('Phase')
    fig.colorbar(im1, ax=ax[1],  shrink=0.7, location='bottom')
    fig.colorbar(im0, ax=ax[0],  shrink=0.7,location='bottom')
    plt.show()


def Fourier(x):
    N = np.shape(x)[0]*np.shape(x)[1]
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x))) / np.sqrt(N)


def calc_cost(far_field,output,type_cost = 0):
   #type_cost = 0 for L2_MAG_LOSS
   #Far field is INTENSITY, near field is COMPLEX AMPLITUDE

    if np.shape(far_field) != np.shape(output):
        target_shape = np.shape(far_field)
        pad_shape = np.shape(output)
        # Compute padding for each dimension
        pad_width = [(0, target_shape[i] - pad_shape[i]) for i in range(len(target_shape))]

        # Apply padding
        output = np.pad(output, pad_width, mode='constant', constant_values=0)
        print('Padded for calculation')

    FX = Fourier(output)
    far_field = far_field + 1e-10
    return np.square(np.linalg.vector_norm((np.abs(FX)**2)/np.sqrt(far_field) - np.sqrt(far_field))) /8 # the 8 is a istake in the origin


def random_zernike_phase(size, max_j=20, strength=np.pi):
    # Generate grid on the unit disk
    y, x = np.linspace(-1, 1, size), np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    mask = rho <= 1

    # Initialize the Zernike object
    cart = zernike.RZern(6)
    assert max_j <= zernike.Zern.nm2noll(6, 6)
    cart.make_cart_grid(X, Y)

    # Generate random coefficients for the phase map
    coeffs = np.zeros(cart.nk)
    for j in range(2, max_j + 1):  # skip piston term j=1
        coeffs[j - 1] = np.random.randn() * np.exp(-0.1 * j)  # decay higher orders

    # Generate phase map
    phase_map = cart.eval_grid(coeffs, matrix=True)

    # Normalize and scale
    mask = ~np.isnan(phase_map)  # True for valid points
    max_val = np.max(np.abs(phase_map[mask]))
    phase_map[mask] = phase_map[mask] / max_val * strength
    phase_map[~mask] = 0

    return phase_map


def image_with_random_zernike_phase(filename, size):
    # Load image as near field intensity
    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image,(size,size))
    intensity = np.asarray(image)
    intensity = intensity / np.max(intensity)

    # Generate phase map
    phase_map = random_zernike_phase(size)

    # Combine into complex field
    combined_map = np.sqrt(intensity) * np.exp(1j * phase_map)
    
    return phase_map, combined_map


def add_gaussian_spot(image, x0=None, y0=None, sigma=None, amplitude=None, normalize=True):
    """
    Add a Gaussian bright spot to a 2D image.
    - image: 2D numpy array (float), intensity-like (non-negative).
    - x0, y0: spot center in pixel coordinates (cols, rows).
    - sigma: standard deviation in pixels.
    - amplitude: peak additional intensity (same units as image).
    - normalize: if True, clip image to non-negative after adding.
    Returns new image (copy).
    """
    H, W = image.shape

    x0 = x0 if x0 is not None else random.randrange(0, H)
    y0 = y0 if y0 is not None else random.randrange(0, W)
    sigma = sigma if sigma is not None else np.random.uniform(2.0, 10.0)
    amplitude = amplitude if amplitude is not None else np.random.uniform(0.5, 3.0)

    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    g = np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
    g *= amplitude  # peak = amplitude
    out = image.astype(float).copy()
    out += g
    if normalize:
        out = np.clip(out, 0, None)
    return out
