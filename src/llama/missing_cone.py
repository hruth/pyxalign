import numpy as np
import cupy as cp
import scipy
import cupyx.scipy.fft as cufft
from tqdm import tqdm
from llama.gpu_utils import memory_releasing_error_handler
import matplotlib.pyplot as plt
from llama.regularization import chambolleLocalTV3D
from IPython.display import clear_output

timerOn = True


@memory_releasing_error_handler
def fillMissingCone(
    rec,
    p,
    laminoAngle,
    maskRelax=0.05,
    maxScale=16,
    Niter=10,
    deltaBackground=1.2e-5,
    deltaMaximal=4.26e-5,
    TV_lambda=1e-7,
):
    #### CHANGE TO USER-SET VARIABLES LATER ####
    # maskRelax = np.float32(0.05)  # 0.15
    # Process the reconstruction using multiscale approach, start at 0.5^max_scale
    # maxScale = 16
    # Number of iterations in each step for filling the missing cone
    # Niter = 10
    deltaBackground = np.float32(deltaBackground)  # 0
    deltaMaximal = np.float32(deltaMaximal)  # 2.5e-5
    TV_lambda = np.float32(TV_lambda)
    #### #### #### #### #### #### #### ####

    # Initialization
    factor = np.float32(p["lambda"] / (2 * np.pi * p["dx_spec"][0]))
    rec = (rec * factor).astype(np.float32)
    rec = rec - np.median(rec)

    Npix = np.array(rec.shape, dtype=int)

    # weakly suppress nonzero values in empty regions (along vertical axis)
    maskVert = np.mean(np.abs(rec), axis=(1, 2))
    maskVert = 1 - maskRelax + maskRelax * maskVert / np.max(maskVert)

    # NpixFull = rec.shape
    # maxBlockSize = [512, 512]

    borderSize = np.array([32, 32], dtype=int)
    blockSize = np.array([512, 512], dtype=int) - 2 * borderSize

    # borderSize = np.array([32, 32], dtype=int)
    # blockSize = np.array([1024, 1024], dtype=int) - 2 * borderSize

    # blockSize = [1024, 1024]
    # blockSize = [128, 128]

    scales = (2 ** np.arange(np.log2(maxScale), -1, -1)).astype(int)
    for scale in scales:
        # for scale in [16]:
        print("Running scale", scale, "...")

        if scale > 1:
            recSmall = interpFT_3D(rec, np.ceil(Npix / scale))
            maskVertSmall = maskVert[::scale, None, None]
            # maskVertSmall = interpFT_3D(
            #     maskVert[:, np.newaxis, np.newaxis], [np.ceil(Npix[0] / scale), 1, 1]
            # ) # this isn't working properly, probably bc of issues with interpFT_3D
        else:
            recSmall = rec
            maskVertSmall = maskVert[:, np.newaxis, np.newaxis]
        lowFreqProtection = scale < maxScale

        recRegularized = applyLaminoConstraints(
            recSmall,
            maskVertSmall,
            laminoAngle,
            lowFreqProtection,
            deltaMaximal,
            deltaBackground,
            Niter,
            TV_lambda,
            borderSize=borderSize,
            blockSize=blockSize,
        )

        rec = rec + interpFT_3D(recRegularized - recSmall, Npix)
        # mid = int(rec.shape[0]/2)
        # clear_output(wait=True)
        # plt.title(scale)
        # plt.imshow(rec[mid], cmap="bone")
        # plt.show()

        del recRegularized, recSmall

    return rec


def blockProc(func):
    """Decorator for splitting the input into blocks, so that the GPU
    memory is not exceeded. Similar to matlab's blockproc."""

    def wrappedFunc(*args, **kwargs):
        args = list(args)

        blockSize = kwargs["blockSize"]

        h, w = args[0].shape[1:]
        m, n = blockSize

        img = args[0] * 1
        for x in range(0, h, m):
            for y in range(0, w, n):
                print("block shape:", args[0][:, x : x + m, y : y + n].shape)
                block = img[:, x : x + m, y : y + n]
                block[:, :, :] = func(block, *args[1:], **kwargs)
        return img

    return wrappedFunc


@memory_releasing_error_handler
def padInputs(func):
    """Decorator for padding the input volume before a function call and
    removing the padding after the function call"""

    def wrappedFunc(*args, **kwargs):
        args = list(args)

        borderSize = kwargs["borderSize"]

        # Pad on each side
        args[0] = np.pad(args[0], ([0, 0], borderSize, borderSize), "symmetric")
        # Call the function
        results = func(*args, **kwargs)
        # Remove padding
        results = results[
            :,
            borderSize[0] : results.shape[1] - borderSize[0],
            borderSize[1] : results.shape[2] - borderSize[1],
        ]
        return results

    return wrappedFunc


@memory_releasing_error_handler
@blockProc
@padInputs
def applyLaminoConstraints(
    volume,
    mask,
    laminoAngle,
    lowFreqProtection,
    valueMax,
    valueMin,
    Niter,
    TV_lambda,
    borderSize,
    blockSize,
):
    Npix = volume.shape
    fftMask = getLaminoFourierMask(Npix, laminoAngle, True)
    fftMask = cp.array(fftMask)
    mask = cp.array(mask)

    if lowFreqProtection:
        # Avoid modifying low spatial frequencies that were already
        # refined
        fftMask = scipy.fft.fftshift(fftMask)
        pts = []
        for i in range(3):
            pts = pts + [
                [
                    int(np.ceil(Npix[i] / 2) - np.ceil(Npix[i] / 8)),
                    int(np.ceil(Npix[i] / 2) + np.floor(Npix[i] / 8)),
                ]
            ]
        pts = np.array(pts, dtype=int)
        fftMask = fftMask.astype(int)
        fftMask[pts[0][0] - 1 : pts[0][1], pts[1][0] - 1 : pts[1][1], pts[2][0] - 1 : pts[2][1]] = 0
        fftMask = scipy.fft.fftshift(fftMask)

    volume = cp.array(volume)
    volumeNew = volume * 1

    for i in tqdm(range(Niter)):
        # time.sleep(2)
        volumeNew = chambolleLocalTV3D(volumeNew, TV_lambda, 10)

        # Positivity constraint
        volumeNew[volumeNew < valueMin] = valueMin
        volumeNew[volumeNew > valueMax] = valueMax
        volumeNew = volumeNew * mask

        # Go to the Fourier space
        with scipy.fft.set_backend(cufft):
            fftVolume = scipy.fft.fftn(volume)
            fftVolumeNew = scipy.fft.fftn(volumeNew)

        # Merge updated and original dataset in the Fourier space
        # Use overrelaxation of the constraint to get faster convergence
        relax = np.float32(1.5)
        regularize = 0
        fftVolume = fftVolume * (1 - relax * fftMask) + fftVolumeNew * relax * fftMask
        fftVolume = fftVolume * (1 - regularize * fftMask).astype(np.float32)
        del fftVolumeNew

        # Go back to real space
        with scipy.fft.set_backend(cufft):
            volumeNew = np.real(scipy.fft.ifftn(fftVolume))
        del fftVolume

        volume = volumeNew

        clear_output(wait=True)
        plt.title(i)
        plt.imshow(volumeNew[int(volumeNew.shape[0] / 2)].get(), cmap="bone")
        plt.show()

    volumeNew = volumeNew.get()

    return volumeNew  # , update


@memory_releasing_error_handler
def getLaminoFourierMask(Npix, laminoAngle, keepOnGPU=False):
    grid = []
    for i in range(3):
        grid = grid + [scipy.fft.fftshift(np.linspace([[-1]], [[1]], Npix[i], axis=i))]
    fftMask = getMask(grid[1], grid[2], grid[0], laminoAngle)

    return fftMask


@memory_releasing_error_handler
def getMask(xGrid, yGrid, zGrid, laminoAngle):
    fftMask = (
        np.ceil(180 / np.pi * np.arctan(np.abs(zGrid) / np.sqrt(xGrid**2 + yGrid**2))) > laminoAngle
    )

    return fftMask


## Utils


@memory_releasing_error_handler
def interpFT_3D(img, Nout):
    # Functionality is questionable -- needs to be tested

    Nin = np.array(img.shape, dtype=int)
    Nout = np.array(Nout, dtype=int)

    imFT = scipy.fft.fftshift(scipy.fft.fftn(img))
    imOut = cropPad3D(imFT, Nout)
    imOut = scipy.fft.ifftn(scipy.fft.ifftshift(imOut)) * Nout.prod() / Nin.prod()

    isReal = img.dtype != np.complexfloating
    if isReal:
        imOut = imOut.astype(img.dtype)

    return imOut


@memory_releasing_error_handler
def cropPad3D(img, Nout):
    # Functionality is questionable -- needs to be tested

    Nin = img.shape
    center = np.floor(np.array(img.shape) / 2)
    imOut = np.zeros(Nout, dtype=img.dtype)
    centerOut = np.floor(Nout / 2)
    A = centerOut - center

    idxOut = {}
    idxIn = {}
    for i in range(3):
        idxOut[i] = np.arange(
            np.append(A[i], 0).max(), np.append(A[i] + Nin[i], Nout[i]).min(), dtype=int
        )
        idxIn[i] = np.arange(
            np.append(-A[i], 0).max(), np.append(-A[i] + Nout[i], Nin[i]).min(), dtype=int
        )

    imOut[
        idxOut[0][0] : idxOut[0][-1], idxOut[1][0] : idxOut[1][-1], idxOut[2][0] : idxOut[2][-1]
    ] = img[idxIn[0][0] : idxIn[0][-1], idxIn[1][0] : idxIn[1][-1], idxIn[2][0] : idxIn[2][-1]]

    isComplex = img.dtype == np.complexfloating
    if isComplex:
        imOut = imOut.astype(img.dtype)

    return imOut
