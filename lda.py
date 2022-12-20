import numpy as np
import utilities as utl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Preparing Data
    data, x, y = utl.load_dataset()
    
    eigenvectors = utl.LDA(data, y)
    
    n_components = [1, 40, 60]

    image = []
    for i in n_components:
        w = eigenvectors[0:i]
        lda = np.dot(data, w.T)
        X_reconstructed = np.dot(lda, w) + (np.mean(data, axis=0))
        im = X_reconstructed[0].reshape(64, 64)
        plt.imshow(im,)
        plt.title('Reconstructed with K = ' + str(i), fontsize=16)
        plt.show()
