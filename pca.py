import numpy as np
import utilities as utl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Preparing Data
    data, x, y = utl.load_dataset()

    feat_mean, data = utl.zero_mean(data)

    vectors = utl.pca(data)

    # PCA
    dims = [2, 3]
    pca_fig = plt.figure()
    pca_fig.suptitle('PCA For Olivetti Faces')
    for i, dim in enumerate(dims):
        vector = vectors[:dim]
        proj = utl.project(data, vector)
        if dim != 3:
            pca_ax = pca_fig.add_subplot(1, 2, i + 1)
            pca_ax.scatter(proj[:, 0], proj[:, 1], c=y)
        else:
            pca_ax = pca_fig.add_subplot(1, 2, i + 1, projection='3d')
            pca_ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=y)
            pca_ax.set_zlabel('PC-3')
        pca_ax.set_title('Faces With ' + str(dim) + ' Principal Components')
        pca_ax.set(xlabel='PC-1', ylabel='PC-2')
    pca_fig.tight_layout()

    # Reconstruction
    components = [1, 20, 50, 150]
    recon_fig = plt.figure()
    recon_fig.suptitle('Reconstruct From PCA')
    for i, component in enumerate(components):
        vector = vectors[:component]
        reconstruct = np.dot(data, np.dot(vector.T, vector)) + feat_mean
        recon_ax = recon_fig.add_subplot(2, 2, i + 1)
        recon_ax.imshow(reconstruct[20].reshape(64, 64),cmap='gray')
        recon_ax.set_title(f'K = {component}')
    recon_fig.tight_layout()

    plt.show()
