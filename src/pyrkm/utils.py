from __future__ import annotations

import glob
import gzip
import io
import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as fft
from scipy.linalg import sqrtm
from scipy.optimize import fsolve
from torchvision.models import inception_v3

# Global cache for inception model to avoid reloading
_inception_model_cache = None


def get_inception_model(device):
    """Get cached inception model or create new one if needed."""
    global _inception_model_cache
    if _inception_model_cache is None or _inception_model_cache[1] != device:
        model = inception_v3(pretrained=True, transform_input=False).to(device)
        model.eval()
        # Compile model for faster inference if using recent PyTorch
        try:
            model = torch.jit.script(model)
        except Exception:
            print(
                'Warning: Scripting Inception model failed, using unscripted version.'
            )
            pass  # Fall back to regular model if scripting fails
        _inception_model_cache = (model, device)
    return _inception_model_cache[0]


def load_model(name, delete_previous=False, model_state_path='model_states/'):
    """Load a model from the specified path.

    Parameters
    ----------
    name : str
        The name of the model to load.
    delete_previous : bool, optional
        If True, delete previous model checkpoints except the latest one (default is False).
    model_state_path : str, optional
        The path to the model state directory (default is 'model_states/').

    Returns
    -------
    bool
        True if the model is loaded successfully, False otherwise.
    object
        The loaded model if successful, otherwise an empty list.
    """
    # Check if you have model load points
    filename_list = glob.glob(model_state_path + '{}_t*.pkl'.format(name))
    if len(filename_list) > 0:
        all_loadpoints = sorted(
            [int(x.split('_t')[-1].split('.pkl')[0]) for x in filename_list])
        last_epoch = all_loadpoints[-1]
        print('** Model {} trained up to epoch {}, so I load it'.format(
            name, last_epoch),
              flush=True)
        with open(model_state_path + '{}_t{}.pkl'.format(name, last_epoch),
                  'rb') as file:
            model = pickle.load(file)
        if delete_previous:
            # Remove all the previous loadpoints
            for x in all_loadpoints[:-1]:
                os.remove(model_state_path + '{}_t{}.pkl'.format(name, x))
        return True, model
    else:
        print('** No load points for {}'.format(name), flush=True)
        return False, []


def show_and_save(file_name,
                  img,
                  cmap='gray',
                  vmin=None,
                  vmax=None,
                  save=False,
                  savename=''):
    """Display and optionally save an image.

    Parameters
    ----------
    file_name : str
        The title of the image.
    img : array-like
        The image data.
    cmap : str, optional
        The colormap to use (default is 'gray').
    vmin : float, optional
        The minimum data value that corresponds to colormap (default is None).
    vmax : float, optional
        The maximum data value that corresponds to colormap (default is None).
    save : bool, optional
        If True, save the image to a file (default is False).
    savename : str, optional
        The name of the file to save the image (default is '').
    """
    plt.title(file_name)
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    if save:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()


def make_grid(array, nrow=8, padding=2):
    """Create a grid of images.

    Parameters
    ----------
    array : array-like
        The array of images.
    nrow : int, optional
        The number of images in each row (default is 8).
    padding : int, optional
        The amount of padding between images (default is 2).

    Returns
    -------
    array-like
        The grid of images.
    """
    N = array.shape[0]
    H = array.shape[1]
    W = array.shape[2]
    grid_h = int(np.ceil(N / float(nrow)))
    grid_w = nrow
    grid = np.zeros(
        [grid_h * (H + padding) + padding, grid_w * (W + padding) + padding])
    k = 0
    for y in range(grid_h):
        for x in range(grid_w):
            if k < N:
                grid[y * (H + padding):y * (H + padding) + H,
                     x * (W + padding):x * (W + padding) + W] = array[k]
                k = k + 1
    return grid


def getbasebias(data):
    """Returns the maximum likelihood estimate of the visible bias,
    given the data. If no data is given the RBMs bias value is return,
    but is highly recommended to pass the data.

    Parameters
    ----------
    data : array-like
        The input data.

    Returns
    -------
    torch.Tensor
        The base bias.
    """
    save_mean = torch.clip(data.mean(0), 0.00001, 0.99999)
    return (torch.log(save_mean) - torch.log(1.0 - save_mean))


def Covariance_error(centered_data_original, centered_data_model, Nv):
    """Compute the covariance error between original and model data.

    Parameters
    ----------
    centered_data_original : torch.Tensor
        The centered original data.
    centered_data_model : torch.Tensor
        The centered model data.
    Nv : int
        The number of visible units.

    Returns
    -------
    torch.Tensor
        The covariance error.
    """
    covariance_matrix_original = torch.matmul(centered_data_original.T,
                                              centered_data_original).mean(0)
    covariance_matrix_model = torch.matmul(centered_data_model.T,
                                           centered_data_model).mean(0)
    return torch.pow(covariance_matrix_original - covariance_matrix_model,
                     2).triu().sum() * 2 / (Nv * (Nv - 1))


def Third_moment_error(centered_data_original, centered_data_model, Nv):
    """Compute the third moment error between original and model data.

    Parameters
    ----------
    centered_data_original : torch.Tensor
        The centered original data.
    centered_data_model : torch.Tensor
        The centered model data.
    Nv : int
        The number of visible units.

    Returns
    -------
    torch.Tensor
        The third moment error.
    """
    C_ijk_original = torch.einsum(
        'ni,nj,nk->ijk', centered_data_original, centered_data_original,
        centered_data_original) / centered_data_model.shape[0]
    C_ijk_model = torch.einsum(
        'ni,nj,nk->ijk', centered_data_model, centered_data_model,
        centered_data_model) / centered_data_model.shape[0]
    C_ijk = torch.pow(C_ijk_model - C_ijk_original, 2)
    sum_ijk = 0.0
    upper_triangular = torch.triu(C_ijk, diagonal=1)
    sum_ijk = upper_triangular.sum(dim=(0, 1, 2))
    return sum_ijk * 6 / (Nv * (Nv - 1) * (Nv - 2))


def PowerSpectrum_MSE(v, v_model):
    """Compute the mean squared error of the power spectrum between original and model data.

    Parameters
    ----------
    v : torch.Tensor
        The original data.
    v_model : torch.Tensor
        The model data.

    Returns
    -------
    torch.Tensor
        The mean squared error of the power spectrum.
    """
    # Apply 2D FFT to the signal
    signal_fft_original = fft.fft2(v)
    signal_fft_model = fft.fft2(v_model)
    # Compute the power spectrum
    power_spectrum_original = torch.mean(torch.abs(signal_fft_original)**2, 0)
    power_spectrum_model = torch.mean(torch.abs(signal_fft_model)**2, 0)
    # MSE of the power spectrum
    return torch.mean((torch.log(power_spectrum_original) -
                       torch.log(power_spectrum_model))**2)


def ComputeAATS(v, v_model):
    """Compute the Average Absolute Truth Score (AATS) between original and model data.

    Parameters
    ----------
    v : torch.Tensor
        The original data.
    v_model : torch.Tensor
        The model data.

    Returns
    -------
    float
        The AATS for true samples.
    float
        The AATS for synthetic samples.
    """
    CONCAT = torch.cat((v, v_model), 1)
    dAB = torch.cdist(CONCAT.t(), CONCAT.t())
    torch.diagonal(dAB).fill_(float('inf'))
    dAB = dAB.cpu().numpy()

    # the next line is use to tranform the matrix into
    #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
    #  d_FT d_FF        d_FT  d_FF
    dAB[:int(dAB.shape[0] / 2), :] = dAB[:int(dAB.shape[0] / 2), ::-1]
    closest = dAB.argmin(axis=1)
    n = int(closest.shape[0] / 2)

    ninv = 1 / n
    AAtruth = (closest[:n] >= n).sum() * ninv
    AAsyn = (closest[n:] >= n).sum() * ninv

    return AAtruth, AAsyn


# Global cache for real image statistics to enable automatic optimization
_real_image_stats_cache: Dict = {}


def Compute_FID(synthetic_images, real_images, dtype=torch.float64):
    """Compute the Frechet Inception Distance (FID) between synthetic and real images.

    Automatically uses precomputed statistics
    for efficiency when the same real images are used repeatedly.

    Parameters
    ----------
    synthetic_images : torch.Tensor
        The synthetic images.
    real_images : torch.Tensor
        The real images.
    dtype : torch.dtype, optional
        Data type for processing (default is torch.float64 for compatibility).

    Returns
    -------
    float
        The FID score.
    """
    device = synthetic_images.device
    inception_model = get_inception_model(device)
    batch_size = 128  # Increased batch size for better performance

    def preprocess_images_batch(images, target_dtype):
        """Optimized batch preprocessing with memory efficiency."""
        images = images.reshape(-1, 28, 28).unsqueeze(1)
        images = images.expand(-1, 3, -1, -1).to(target_dtype)
        images = torch.nn.functional.interpolate(images,
                                                 size=299,
                                                 mode='bilinear',
                                                 align_corners=False)
        return images

    def get_activations_batch(images, target_dtype):
        """Process images in batches to avoid memory issues."""
        activations = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            batch = preprocess_images_batch(batch, target_dtype)

            with torch.no_grad():
                batch_output = inception_model(batch)
                # Handle InceptionOutputs - extract logits if it's a named tuple
                if hasattr(batch_output, 'logits'):
                    batch_activations = batch_output.logits.detach()
                else:
                    batch_activations = batch_output.detach()
                activations.append(batch_activations)

            if torch.cuda.is_available() and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()

        return torch.cat(activations, dim=0).cpu().numpy()

    def compute_statistics(activations):
        """Compute mean and covariance with numerical stability."""
        mu = np.mean(activations, axis=0)
        centered = activations - mu
        sigma = np.cov(centered, rowvar=False)
        epsilon = 1e-6
        sigma += np.eye(sigma.shape[0]) * epsilon
        return mu, sigma

    # Create a hash for real images to check if we've seen them before
    if isinstance(real_images, torch.Tensor):
        real_images_np = real_images.detach().cpu().numpy()
    else:
        real_images_np = np.array(real_images)

    # Simple hash based on shape and a sample of values for cache key
    real_hash = hash(
        (real_images_np.shape,
         tuple(real_images_np.flat[::max(1, real_images_np.size // 100)])))

    # Check if we have cached statistics for these real images
    if real_hash in _real_image_stats_cache:
        mu_real, sigma_real = _real_image_stats_cache[real_hash]
    else:
        # Compute real image statistics and cache them
        real_images_tensor = torch.tensor(real_images_np,
                                          device=device,
                                          dtype=dtype)
        real_activations = get_activations_batch(real_images_tensor, dtype)
        mu_real, sigma_real = compute_statistics(real_activations)
        _real_image_stats_cache[real_hash] = (mu_real, sigma_real)

    # Always compute synthetic statistics (they change each time)
    synthetic_images = synthetic_images.to(device).to(dtype)
    synthetic_activations = get_activations_batch(synthetic_images, dtype)
    mu_synthetic, sigma_synthetic = compute_statistics(synthetic_activations)

    # Compute FID
    diff = mu_synthetic - mu_real
    ssdiff = np.sum(diff**2.0)

    # Use numerically stable matrix square root computation
    try:
        covmean = sqrtm(sigma_synthetic.dot(sigma_real))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception:
        u, s, vh = np.linalg.svd(sigma_synthetic.dot(sigma_real))
        covmean = u.dot(np.diag(np.sqrt(np.maximum(s, 0)))).dot(vh)

    fid = ssdiff + np.trace(sigma_synthetic + sigma_real - 2.0 * covmean)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fid


def precompute_real_image_statistics(real_images,
                                     dtype=torch.float32,
                                     batch_size=64,
                                     device=None):
    """Precompute statistics for real images to speed up repeated FID calculations.

    Use this when you'll compute FID multiple times with the same real dataset.

    Parameters
    ----------
    real_images : torch.Tensor or numpy.ndarray
        The real images dataset.
    dtype : torch.dtype, optional
        Data type for processing (default is torch.float32).
    batch_size : int, optional
        Batch size for processing (default is 64).
    device : torch.device, optional
        Device to use. If None, uses CUDA if available.

    Returns
    -------
    tuple
        (mu_real, sigma_real) - precomputed statistics for real images.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inception_model = get_inception_model(device)

    # Convert to tensor if needed
    if not isinstance(real_images, torch.Tensor):
        real_images = torch.tensor(real_images, device=device)
    else:
        real_images = real_images.to(device)

    def preprocess_images_batch(images, target_dtype):
        images = images.reshape(-1, 28, 28).unsqueeze(1)
        images = images.expand(-1, 3, -1, -1).to(target_dtype)
        images = torch.nn.functional.interpolate(images,
                                                 size=299,
                                                 mode='bilinear',
                                                 align_corners=False)
        return images

    # Get activations in batches
    activations = []
    for i in range(0, real_images.shape[0], batch_size):
        batch = real_images[i:i + batch_size]
        batch = preprocess_images_batch(batch, dtype)

        with torch.no_grad():
            batch_output = inception_model(batch)
            # Handle InceptionOutputs - extract logits if it's a named tuple
            if hasattr(batch_output, 'logits'):
                batch_activations = batch_output.logits.detach()
            else:
                batch_activations = batch_output.detach()
            activations.append(batch_activations)

        if torch.cuda.is_available() and i % (batch_size * 4) == 0:
            torch.cuda.empty_cache()

    real_activations = torch.cat(activations, dim=0).cpu().numpy()

    # Compute statistics
    mu_real = np.mean(real_activations, axis=0)
    real_centered = real_activations - mu_real
    sigma_real = np.cov(real_centered, rowvar=False)

    # Add regularization
    epsilon = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * epsilon

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mu_real, sigma_real


def Compute_S(v, v_gen):
    """Compute the relative entropy between
    original and generated data.

    Parameters
    ----------
    v : torch.Tensor
        The original data.
    v_gen : torch.Tensor
        The generated data.

    Returns
    -------
    float
        The relative entropy.
    """
    v = v.detach().cpu().numpy()
    try:
        v_gen = v_gen.detach().cpu().numpy()
    except Exception:
        v_gen = v_gen

    # define a mixed set for crossentropy:
    # this set will contain the first half of the original set and the second half of the generated set
    v_cross = v.copy()
    v_cross[:int(0.5 * v_cross.shape[0])] = v_gen[:int(0.5 * v_cross.shape[0])]

    # Convert the array to bytes
    bytes_io = io.BytesIO()
    np.save(bytes_io, v)
    bytes_src = bytes_io.getvalue()
    np.save(bytes_io, v_cross)
    bytes_cross = bytes_io.getvalue()

    # Compress the bytes using gzip
    compressed_src = gzip.compress(bytes_src)
    compressed_cross = gzip.compress(bytes_cross)

    # Calculate the entropy
    byte_count_src = len(compressed_src)
    byte_count_cross = len(compressed_cross)
    value_counts_src = np.bincount(
        np.frombuffer(compressed_src, dtype=np.uint8))
    value_counts_cross = np.bincount(
        np.frombuffer(compressed_cross, dtype=np.uint8))
    probabilities_src = value_counts_src / byte_count_src
    probabilities_cross = value_counts_cross / byte_count_cross
    probabilities_src = probabilities_src[probabilities_src > 0]
    probabilities_cross = probabilities_cross[probabilities_cross > 0]
    entropy_src = -np.sum(probabilities_src * np.log2(probabilities_src))
    entropy_cross = -np.sum(probabilities_cross * np.log2(probabilities_cross))

    # the final measure is this relative entropy that is centered around 0.
    return entropy_cross / entropy_src - 1


def generate_S_matrix(shape, target):
    """Generate a random matrix with values between 0 and 1, adjusted to achieve the desired average.

    Parameters
    ----------
    shape : tuple
        The shape of the matrix.
    target : float
        The target average value.

    Returns
    -------
    numpy.ndarray
        The generated matrix.
    """
    random_matrix = np.random.rand(*shape)
    adjusted_matrix = random_matrix + (target - np.mean(random_matrix))
    adjusted_matrix = np.clip(adjusted_matrix, 0, 1)
    return adjusted_matrix


def generate_synthetic_data(target_entropy, data_size, structured=True):
    """Generate synthetic data with the specified target entropy.

    Parameters
    ----------
    target_entropy : float
        The target entropy value.
    data_size : tuple
        The size of the data to generate.
    structured : bool, optional
        If True, generate structured data (default is True).

    Returns
    -------
    numpy.ndarray
        The generated synthetic data.
    """

    def S_lambda(x):
        return -x * np.nan_to_num(np.log2(x)) - (1 - x) * np.nan_to_num(
            np.log2(1 - x))

    pixel_target = generate_S_matrix((data_size[1], data_size[2]),
                                     target_entropy)
    if structured:
        flat_indices = np.argsort(pixel_target.flatten())
        pixel_target = pixel_target.flatten()[flat_indices].reshape(
            pixel_target.shape)

    initial_guess = np.zeros((data_size[1], data_size[2]))
    P = fsolve(lambda x: S_lambda(x) - pixel_target.flatten(),
               initial_guess.flatten())

    generated_data = ((np.random.rand(data_size[0],
                                      data_size[1] * data_size[2])
                       < P).astype(int)).reshape(data_size)
    S_image, S_pixel = my_entropy(generated_data)
    print(f'\nTarget = {target_entropy}')
    print(f'generated entropy (image) = {S_image.mean()}')
    print(f'generated entropy (pixels) = {S_pixel.mean()}')

    return generated_data


def my_entropy(data):
    """Compute the entropy per image and per pixel.

    Parameters
    ----------
    data : numpy.ndarray
        The input data.

    Returns
    -------
    numpy.ndarray
        The entropy per image.
    numpy.ndarray
        The entropy per pixel.
    """
    X = data.mean(1).mean(1)
    S_image = -X * np.nan_to_num(np.log2(X)) - (1 - X) * np.nan_to_num(
        np.log2(1 - X))
    Y = data.mean(0)
    S_pixel = -Y * np.nan_to_num(np.log2(Y)) - (1 - Y) * np.nan_to_num(
        np.log2(1 - Y))
    return S_image, S_pixel


def binarize_image(image, threshold=128):
    """Binarize the image using a threshold.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    threshold : int, optional
        The threshold value (default is 128).

    Returns
    -------
    numpy.ndarray
        The binarized image.
    """
    binary_image = (image > threshold).astype(int)
    return binary_image


def ensure_dir(dirname):
    """Create a directory if it does not exist.

    Parameters
    ----------
    dirname : str
        The name of the directory to create.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def unpickle(file):
    """Unpickle a file.

    Parameters
    ----------
    file : str
        The file to unpickle.
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
