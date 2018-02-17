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
  - ii

## Autoencoder

1. Autoencoder find its own way in image denosing and image compression. 
2. Using resize function to upsample image in a network might be better than doing transpose convolution, which will lead to checkboard artifact. 


## RNN

- Perform the same task for each element in the input sequence. 

### History
- Feed Forward network is limited because they are unable to capture temperal dependency. 


### RNN Structure
- There are two main differences between FFNNs and RNNs. The Recurrent Neural Network uses:
    - sequences as inputs in the training phase, and
    - memory elements
- Memory is defined as the output of hidden layer neurons, which will serve as additional input to the network during next training step.


## Word Embedding

### Word2Vec [link](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

- Overview
    - We’re going to train the neural network to do the following. Given a specific word in the middle of a sentence (the input word), look at the words nearby and pick one at random. The network is going to tell us the probability for every word in our vocabulary of being the “nearby word” that we chose.

    - We’ll train the neural network to do this by feeding it word pairs found in our training documents. 

    - The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). 

- Model Detail

    - First of all, We’re going to represent an input word like “ants” as a one-hot vector.
    - When training this network on word pairs, the input is a one-hot vector representing the input word and the training output is also a one-hot vector representing the output word.
    - the end goal of all of this is really just to learn this hidden layer weight matrix – the output layer we’ll just toss when we’re done!

- Negative Sampling

  - In the example I gave, we had word vectors with 300 components, and a vocabulary of 10,000 words. Recall that the neural network had two weight matrices–a hidden layer and output layer. Both of these layers would have a weight matrix with 300 x 10,000 = 3 million weights each!
  - There are three innovations in this second paper:

    - Treating common word pairs or phrases as single “words” in their model.
    - Subsampling frequent words to decrease the number of training examples.
    - Modifying the optimization objective with a technique they called “Negative Sampling”, which causes each training sample to update only a small percentage of the model’s weights.

  - Word2Vec implements a “subsampling” scheme to address this. For each word we encounter in our training text, there is a chance that we will effectively delete it from the text. The probability that we cut the word is related to the word’s frequency.
  - When training the network on the word pair (“fox”, “quick”), recall that the “label” or “correct output” of the network is a one-hot vector. That is, for the output neuron corresponding to “quick” to output a 1, and for all of the other thousands of output neurons to output a 0.
  - With negative sampling, we are instead going to randomly select just a small number of “negative” words (let’s say 5) to update the weights for. (In this context, a “negative” word is one for which we want the network to output a 0 for). We will also still update the weights for our “positive” word (which is the word “quick” in our current example).

## Generative Adversarial Network

### Application of GAN

- GAN is used for generating realistic data. 
  - STACKGAN model: Taking a textual description of bird and generate real image of bird. 
  - Pix2pix: Draw very crude sketch using the mouse, iGAN draws the nearest possible realistic image. 
  - Transferring a photo of a face into cartoon of a face. 
  - CycleGAN: unsupervised image to image translation. 
  
- GAN is also used for generating realistic simulated training set. 
- Imitation learning: GAN can learn the action taken by human expert. 

### How GAN works

- Instead of generating one pixel at a time, GAN generate the whole image in parallel. 


### Game and equilibria

- Equilibria: Neither player can improve their payoff by changing strategy. In other words, no player can improve their payoff without changing other players stretegy. 

- To understand GAN, we need to think about how payoff and equilibria work in the context of machine learning. The optimization algorithm of GAN is different from other NN (SGD to minimize cost function.)

### Practical tips and tricks for training GAN

- Structure:
  - If we use NN as generator & discriminator, it's important for both of them to have at least one hidden layer so that they both have universal approximation property. 

  - The discriminator can use leaky relu as activation function, it's important for the generator to receive gradient from discriminator. 

  - The tanh is a good activation function for the output of generator network, where the output is scale to between -1 and 1. 

- Optimization
  - Two Simultaneous Optimization. 
  - Trick: Replace label of 1 with 0.9, and keep the label 0. This is a specific GAN label moving stretgy used for regularize classifier. It helps the discriminator to generalize better, and avoid learning to make extreme prediction when extrapolating. 
  - For the generator optimizer, the labels are flipped. In other words, the generator will maximize the log probability of wrong label. 
  - But in practice, it's much better for the generator to minize cross-entropy with the labels flipped, rather than maximize the discriminator cross-entropy. 

- Scale up network to work for large image. 
  - Use CNN to construct GAN. 
