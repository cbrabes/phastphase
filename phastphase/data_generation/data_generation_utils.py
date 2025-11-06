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


def generate_zernike_phase_map(shape, min_j=2, max_j=20, decay=0.1, aperature="circular"):
    unit_circle = True
    sample_scale = 1
    
    if aperature == "circular":
        unit_circle = True
        assert shape[0] == shape[1]
    elif aperature == "rectangular":
        unit_circle = False
    elif aperature == "cropped":
        unit_circle = True
        sample_scale = jnp.sqrt(shape[0]**2 + shape[1]**2) / min(shape[0], shape[1])
    else:
        raise NotImplementedError(f"Unsupported aperature: {aperature}")
    
    # Generate grid on the unit disk
    y, x = jnp.linspace(-1, 1, int(shape[0] * sample_scale)), jnp.linspace(-1, 1, int(shape[1] * sample_scale))
    X, Y = jnp.meshgrid(x, y)
    rho = jnp.sqrt(X**2 + Y**2)
    mask = rho <= 1

    # Initialize the Zernike object
    cart = zernike.RZern(6)
    assert max_j <= zernike.Zern.nm2noll(6, 6)
    cart.make_cart_grid(X, Y, unit_circle=unit_circle)

    # Generate random coefficients for the phase map
    coeffs = jnp.zeros(cart.nk)
    for j in range(min_j, max_j + 1):  # skip piston term j=1
        coeffs = coeffs.at[j - 1].set(random.gauss(mu=0, sigma=1) * jnp.exp(-decay * j))  # decay higher orders

    # Generate phase map
    phase_map = jnp.array(cart.eval_grid(coeffs, matrix=True))

    # Wrap to [-pi, pi]
    phase_map = jnp.angle(jnp.exp(1j * phase_map))

    # Set NaN -> 0
    if unit_circle:
        mask = ~jnp.isnan(phase_map)  # True for valid points
        phase_map = phase_map.at[~mask].set(0)

    # Return after crop (if scaling was performed).
    if sample_scale != 1:
        middle_point = (int(shape[0] * sample_scale / 2), int(shape[1] * sample_scale / 2))
        phase_map = phase_map[
            int(middle_point[0] - shape[0]/2): int(middle_point[0] + shape[0]/2),
            int(middle_point[1] - shape[1]/2): int(middle_point[1] + shape[1]/2)
        ]
    
    return phase_map


def image_with_random_phase_map(filename, shape, decay=0.1, strength=jnp.pi, aperature="circular"):
    # Load image as near field intensity
    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, shape)
    intensity = jnp.asarray(image)
    intensity = intensity / jnp.max(intensity)

    # Generate phase map
    phase_map = generate_zernike_phase_map(shape, decay=decay, strength=strength, aperature=aperature)

    # Combine into complex field
    combined_map = jnp.sqrt(intensity) * jnp.exp(1j * phase_map)
    
    return phase_map, combined_map


def add_heaviside_spot(image, x0=None, y0=None, radius=None, amplitude_multiplier=None):
    H, W = image.shape

    x0 = x0 if x0 is not None else random.randrange(0, H)
    y0 = y0 if y0 is not None else random.randrange(0, W)
    radius = radius if radius is not None else random.randrange(0, 10)
    amplitude_multiplier = amplitude_multiplier if amplitude_multiplier is not None else jnp.random.uniform(0.5, 2.0)

    total_image_intensity = jnp.sum(image)
    amplitude = total_image_intensity * amplitude_multiplier

    yy = jnp.arange(H)[:, None]
    xx = jnp.arange(W)[None, :]

    spot_map = jnp.zeros((H, W), dtype=float)
    dist = jnp.sqrt((xx - x0)**2 + (yy - y0)**2)
    spot_map = spot_map.at[dist <= radius].add(amplitude)
    
    return image + spot_map


def add_gaussian_spot(image, x0=None, y0=None, sigma=None, amplitude_multiplier=None):
    """
    Add a Gaussian bright spot to a 2D image.
    - image: 2D numpy array (float), intensity-like (non-negative).
    - x0, y0: spot center in pixel coordinates (cols, rows).
    - sigma: standard deviation in pixels.
    - amplitude: peak additional intensity (same units as image).
    Returns new image (copy).
    """
    H, W = image.shape

    x0 = x0 if x0 is not None else random.randrange(0, H)
    y0 = y0 if y0 is not None else random.randrange(0, W)
    sigma = sigma if sigma is not None else np.random.uniform(2.0, 10.0)
    amplitude_multiplier = amplitude_multiplier if amplitude_multiplier is not None else np.random.uniform(0.5, 2.0)

    total_image_intensity = np.sum(image)
    amplitude = total_image_intensity * amplitude_multiplier

    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    g = np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
    g *= amplitude  # peak = amplitude
    image += g
