## Weight Initilization of Neural Network 

- Never set the weights to be the same. (10% acc at MNIST dataset)
- Random uniform distribution is a good initialization method. (acc 60% on MNIST)
  - The range of the distribution should be ranging from negative number to positive number. 
  - The scale of the distribution could be [1/sqrt(number of neurons of previous layer)]
  - Reasonable range of our initialization. 
    - [-0.1, 0.1) - Almost the same with 1/sqrt(number of neurons of previous layer)
    - [-0.01, 0.01), [-0.001, 0.001) is bit so small. 
 
 - Normal Distribution to initialize weights. 
  - More likely to choose around zero. 
  - Compared with [-0.1, 0.1), normal distribution with std 0.1 performs a little bit better. 
  - Normal distribution usually performs better than random uniform distribution. 
  - But sometimes we still get fairly large number via normal distribution, we hope to pull those larger closer to zero. 
  - That's when truncated normal distribution comes in, who cuts off the tail at both sides. Values have more than two standard 
  deviation from the mean are throwed out. 
  - Usually, truncated normal distribution is slightly better. For small network, it doesn't really matter. But for large netowrk
  with lots of neurons, we are more likely to initialized with large extreme numbers. 
  
  - It's REALLY important to get your weights initialized correctly!!!!!!!!


## Autoencoder

1. Autoencoder find its own way in image denosing and image compression. 
2. Using resize function to upsample image in a network might be better than doing transpose convolution, which will lead to checkboard artifact. 
