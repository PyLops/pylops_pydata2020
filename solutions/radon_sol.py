# Solutions for Radon tutorial - PyData Global 2020 Tutorial

def radon_noise():
    """Create noise to add to projections
    """
    sigman = 5e-1 # play with this...
    n = np.random.normal(0., sigman, projection1.shape)
    projection += n[projection.shape[0] // 2 - (nx - inner) // 2:
    projection.shape[0] // 2 + (nx - inner) // 2]
    projection1 += n

    # copy-paste here the inversion code(s)...


def radon_morereg():
    """Add Wavelet Transform regularization to SplitBregman reconstruction
    """
    # Wavelet transform operator
    Wop = pylops.signalprocessing.DWT2D(image.shape, wavelet='haar', level=5)
    DWop = Dop + [Wop, ]

    # Solve inverse problem
    mu = 0.2
    lamda = [1., 1., .5]
    niter = 5
    niterinner = 1

    image_tvw, niter = pylops.optimization.sparsity.SplitBregman(Radop, DWop,
                                                                 projection1.ravel(),
                                                                 niter,
                                                                 niterinner,
                                                                 mu=mu,
                                                                 epsRL1s=lamda,
                                                                 tol=1e-4,
                                                                 tau=1.,
                                                                 show=True,
                                                                 **dict(
                                                                     iter_lim=10,
                                                                     damp=1e-2))
    image_tvw = np.real(image_tvw.reshape(nx, ny))
    mse_tvw = np.linalg.norm(
        image_tvw[pad // 2:-pad // 2, pad // 2:-pad // 2] -
        image[pad // 2:-pad // 2, pad // 2:-pad // 2])
    print(f"TV+W MSE reconstruction error: {mse_tvw:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image_tvw[pad // 2:-pad // 2, pad // 2:-pad // 2], cmap='gray',
               vmin=0, vmax=1)
    ax1.set_title("Reconstruction\nTV+W Regularized inversion")
    ax1.axis('tight')
    ax2.imshow(image_tvw[pad // 2:-pad // 2, pad // 2:-pad // 2] -
               image[pad // 2:-pad // 2, pad // 2:-pad // 2],
               cmap='gray', vmin=-0.2, vmax=0.2)
    ax2.set_title("Reconstruction error\nTV+W Regularized inversion")
    ax2.axis('tight')


def radon_kk():
    """Perform reconstruction in KK domain
    """
    # 2D FFT operator
    Fop = pylops.signalprocessing.FFT2D(dims=(nx, ny), dtype=np.complex)

    # Restriction operator
    thetas = np.arange(-43, 47, 5)
    kx = np.arange(nx) - nx // 2
    ikx = np.arange(nx)

    mask = np.zeros((nx, ny))
    for theta in thetas:
        ky = kx * np.tan(np.deg2rad(theta))
        iky = np.round(ky).astype(np.int) + nx // 2
        sel = (iky >= 0) & (iky < nx)
        mask[ikx[sel], iky[sel]] = 1
    mask = np.logical_or(mask, np.fliplr(mask.T))
    mask = np.fft.ifftshift(mask)

    Rop = pylops.Restriction(nx * ny, np.where(mask.ravel() == 1)[0],
                             dtype=np.complex)

    # kk spectrum
    kk = Fop * image.ravel()
    kk = kk.reshape(image.shape)

    # restricted kk spectrum
    kkrestr = Rop.mask(kk.ravel())
    kkrestr = kkrestr.reshape(ny, nx)
    kkrestr.data[:] = np.fft.fftshift(kkrestr.data)
    kkrestr.mask[:] = np.fft.fftshift(kkrestr.mask)

    # data
    KOp = Rop * Fop
    y = KOp * image.ravel()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(17, 4))
    ax1.imshow(image, cmap='gray')
    ax1.set_title(r"Input $\mathbf{i}$")
    ax1.axis('tight')
    ax2.imshow(np.fft.fftshift(np.abs(kk)), vmin=0, vmax=1, cmap='rainbow')
    ax2.set_title("KK Spectrum")
    ax2.axis('tight')
    ax3.imshow(kkrestr.mask, vmin=0, vmax=1, cmap='gray')
    ax3.set_title(r"Sampling matrix")
    ax3.axis('tight')
    ax4.imshow(np.abs(kkrestr), vmin=0, vmax=1, cmap='rainbow')
    ax4.set_title(r"Sampled KK Spectrum $\mathbf{k}$")
    ax4.axis('tight')

    # Solve L2 inverse problem
    D2op = pylops.Laplacian(dims=(nx, ny), edge=True, dtype=np.complex)

    image_l2 = \
        pylops.optimization.leastsquares.RegularizedInversion(KOp, [D2op],
                                                              y, epsRs=[5e-1],
                                                              show=True,
                                                              **dict(iter_lim=20))
    image_l2 = np.real(image_l2.reshape(nx, ny))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image_l2[pad // 2:-pad // 2, pad // 2:-pad // 2], cmap='gray',
               vmin=0, vmax=1)
    ax1.set_title("Reconstruction\nL2 Regularized inversion")
    ax1.axis('tight')
    ax2.imshow(image_l2[pad // 2:-pad // 2, pad // 2:-pad // 2] -
               image[pad // 2:-pad // 2, pad // 2:-pad // 2],
               cmap='gray', vmin=-0.2, vmax=0.2)
    ax2.set_title("Reconstruction error\nL2 Regularized inversion")
    ax2.axis('tight')

    # Solve TV inverse problem
    Dop = [pylops.FirstDerivative(ny * nx, dims=(nx, ny), dir=0, edge=True,
                                  kind='backward', dtype=np.complex),
           pylops.FirstDerivative(ny * nx, dims=(nx, ny), dir=1, edge=True,
                                  kind='backward', dtype=np.complex)]

    mu = 0.5
    lamda = [.05, .05]
    niter = 5
    niterinner = 1

    image_tv, niter = \
        pylops.optimization.sparsity.SplitBregman(KOp, Dop, y,
                                                  niter,
                                                  niterinner,
                                                  mu=mu,
                                                  epsRL1s=lamda,
                                                  tol=1e-4,
                                                  tau=1.,
                                                  show=True,
                                                  **dict(iter_lim=10,
                                                         damp=1e-2))
    image_tv = np.real(image_tv.reshape(nx, ny))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image_tv[pad // 2:-pad // 2, pad // 2:-pad // 2], cmap='gray',
               vmin=0, vmax=1)
    ax1.set_title("Reconstruction\nTV Regularized inversion")
    ax1.axis('tight')
    ax2.imshow(image_tv[pad // 2:-pad // 2, pad // 2:-pad // 2] -
               image[pad // 2:-pad // 2, pad // 2:-pad // 2],
               cmap='gray', vmin=-0.2, vmax=0.2)
    ax2.set_title("Reconstruction error\nTV Regularized inversion")
    ax2.axis('tight')