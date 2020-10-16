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
                        offset=(kernel_unsharp.shape[0]//2, kernel_unsharp.shape[1]//2),
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

