# Solutions for Deblurring tutorial - PyData Global 2020 Tutorial

def Unsharp_Mask():
    """Unsharp Mask operator
    """
    # load image from scikit-image
    image = data.microaneurysms()
    ny, nx = image.shape

    # %load -s Diagonal_timing solutions/intro_sol.py

    # Define matrix kernel
    kernel_unsharp = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    # PyLops Operator
    unsharp_op = Convolve2D(N=ny*nx,
                            h=kernel_unsharp,
                            offset=(
                                kernel_unsharp.shape[0]//2, kernel_unsharp.shape[1]//2),
                            dims=(ny, nx))

    img_unsharp = unsharp_op * image.flatten()
    img_unsharp = img_unsharp.reshape(image.shape)

    # PLOTTING
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Image')

    ax[1].imshow(kernel_unsharp, cmap='gray')
    ax[1].set_title('Kernel')

    ax[2].imshow(img_unsharp, cmap='gray')
    ax[2].set_title('Blurred Image')


def Noisy_Inversion():
    """Solve Inverse problems for a noisy image
    """
    scale = 5.8
    noise = np.random.normal(0, scale, img_gauss_.shape)
    img_gauss_noisy_ = img_gauss_ + noise
    img_gauss_noisy = img_gauss_noisy_.flatten()


    # PLOTTING
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img_gauss_, cmap='gray')
    ax[0].set_title('Blurred Image')

    ax[1].imshow(img_gauss_noisy_, cmap='gray')
    ax[1].set_title('Blurred Image + Noise')

    him2 = ax[2].imshow(img_gauss_noisy_ - img_gauss_, cmap='gray')
    ax[2].set_title('Noise level')
    plt.show()
    
    
    # least squares inversion
    deblur_l2 = leastsquares.NormalEquationsInversion(Op=Gauss_op,
                                                      Regs=None,
                                                      data=img_gauss_noisy,
                                                      maxiter=40)

    # least squares - regularized inversion
    deblur_l2_reg = leastsquares.RegularizedInversion(Op=Gauss_op,
                                                      Regs=[D2op],
                                                      data=img_gauss_noisy,
                                                      epsRs=[1e0],
                                                      show=True,
                                                      **dict(iter_lim=20))

    # TV inversion
    deblur_tv = sparsity.SplitBregman(Op=Gauss_op,
                                      RegsL1=Dop,
                                      data=img_gauss_noisy,
                                      niter_outer=10,
                                      niter_inner=2,   
                                      mu=1.8,
                                      epsRL1s=[8e-1, 8e-1],
                                      tol=1e-4,
                                      tau=1.,
                                      ** dict(iter_lim=5, damp=1e-4, show=True))[0]

    # FISTA inversion
    deblur_fista = sparsity.FISTA(Op=Gauss_op,
                                  data=img_gauss_noisy,
                                  eps=1e-1,
                                  niter=100,
                                  show=True)[0]

    # Reshape images
    deblur_l2 = deblur_l2.reshape(image.shape)
    deblur_l2_reg = np.real(deblur_l2_reg.reshape(image.shape))
    deblur_tv = deblur_tv.reshape(image.shape)
    deblur_fista = deblur_fista.reshape(image.shape)


    # PLOTTING
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))

    ax[0, 0].imshow(image, aspect='auto', cmap='gray')
    ax[0, 0].set_title('Original Image')

    ax[0, 1].imshow(deblur_l2, aspect='auto', cmap='gray')
    ax[0, 1].set_title('LS-Inversion')

    ax[0, 2].imshow(deblur_l2_reg, aspect='auto', cmap='gray')
    ax[0, 2].set_title('Regularized LS-Inversion')

    ax[1, 0].imshow(img_gauss_, aspect='auto', cmap='gray')
    ax[1, 0].set_title('Blurred Image')

    ax[1, 1].imshow(deblur_tv, aspect='auto', cmap='gray')
    ax[1, 1].set_title('TV-Inversion')

    ax[1, 2].imshow(deblur_fista, aspect='auto', cmap='gray')
    ax[1, 2].set_title('FISTA Inversion')
    fig.tight_layout()
