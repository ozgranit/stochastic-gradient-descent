# stochastic-gradient-descent
handwritten digits classifier (using the MNIST dataset, separates '0' and '8')

The MNIST dataset consists of images of handwritten digits, 
along with their labels. Each image has 28×28 pixels, where each pixel is in grayscale
scale, and can get an integer value from 0 to 255. Each label is a digit between 0 and 9. The
dataset has 70,000 images. Althought each image is square, we treat it as a vector of size 28×28 = 784.

In the file there is a helper function. The function reads the examples
labelled 0, 8 and returns them with 0−1 labels. 

in SGD we have to select external parameters η0 and C,
we will use the training set to train classifiers for various parameter values
and use the validation set to check how well they do.

we try to optimize η0 and C.

### SGD Algorithm:
- at each iteration t = 1,...T we sample i uniformly; 
  - if yi*wt*xi < 1, we update:
      - wt+1 = (1 − ηt)wt + ηt*C*yi*xi
  - else :
      - wt+1 = (1 − ηt)wt 
  - where ηt = η0/t, and η0 is a constant.
