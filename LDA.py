import utilities as utl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
      
    #Preparing Data
    x,y=utl.load_dataset()
    
    eigenvectors=utl.LDA(x,y)
    
    Ncomponents=[1,40,60]
    image=[]
    for i in Ncomponents:
        w=eigenvectors[0:i]
        lda=np.dot(x,w.T)
        X_reconstructed = np.dot(lda,w) + (np.mean(x,axis=0))
        im = X_reconstructed[0].reshape(64,64)
        plt.imshow(im,cmap = 'gray')
        plt.title('Reconstructed with K = ' + str(i), fontsize=16)
        plt.show()
        
     
