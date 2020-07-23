# viola-jones
The code is the implementation of Face Detection using Viola-Jones Algorithm described at https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf.
# Data
The data is described at http://cbcl.mit.edu/software-datasets/FaceData2.html, and downloaded from www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz and compiled into pickle files.

Each image is 19x19 and greyscale. There are Training set: 2,429 faces, 4,548 non-faces Test set: 472 faces, 23,573 non-faces

- training.pkl
  - An array of tuples. The first element of each tuple is a numpy array representing the image. The second element is its clasification (1 for face, 0 for non-face)
  - 2429 face images, 4548 non-face images

- test.pkl
  - An array of tuples. The first element of each tuple is a numpy array representing the image. The second element is its clasification (1 for face, 0 for non-face)
  - 472 faces, 23573 non-face images
  
# Result
- For the value of hyperparameter T=10, the model achieved 85.5% accuracy on the training set and 78% accuracy on the test set.
