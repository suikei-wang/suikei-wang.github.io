---
title:  "cs231n-Image Classification & Linear Classification & Loss Function"
layout: post
categories: cs231n
tags:  cs231n
author: Suikei Wong
mathjax: true
---

* content
{:toc}


# Image Classification
<br>
**Motivation & Example.** As a core task in Computer Vision, Image Classification is actually a hard problem for machine, and many seemingly distinct Computer Vision tasks can be reduced to image classification. To a computer, an image is represented as one large  **3-dimensional array of numbers**: [width, height, depth], of size Width * Height * 3. The 3 represents the **three color channels Red, Green, Blue**.
<br><br>
**Data-driven approach.** Rather than writing and algorithm that can classify images into distinct categories, we're going to provide the computer with **many examples of each class** and then **develop learning algorithms that look at these examples** and **learn about the visual appearance of each class**. This approach is referred to as a *data-driven approach*, since it relies on first accumulating a *training dataset of labeled images*.
<br><br>
**Pipeline.** The task in Image Classification is to **take an array of pixels that represents a single image and assign a label to it**. Proceed as follows:

* **Input:** A set of *N* images, each labeled with one of *K* different classes. This data is referred to *training set*.
* **Learning:** Using the training set learn what every one of the classes looks like. The step is refereed to *training a classifer*, or *learning a model*.
* **Evaluation:** Evaluate the quality of the classifer by asking it to predict labels for a new set of images that it has never seen before, and compare the true labels(*ground truth*) of these images to its results.

<br><br>
# Nearest Neighbor Classifier
<br>This classifier has nothing to do with Convolutional Neural Networks(CNN), but it will allow us to get an idea about the basic approach to an image classification problem. It just memorize all data and labels during training and find the most similiar(nearest) one in prediction.
<br><br>
With *N* examples, training is very fast since it only needs to memorize all data and labels: **Train O(1)**. However, the prediction is very slow since it needs to loop over all test rows and find the nearest training image to the i'th test image: **Predict O(N)**. This is bad: **we want classifier that are fast at prediction; slow for training is ok**.
<br><br>
**Example dataset: CIFAR-10.** This dataset consists of 60,000 tiny images with size of 32 * 32 and each image is labeled with one of 10 classes:
![nn](http://cs231n.github.io/assets/nn.jpg) 
<span style="color:grey">Left: Example images from the [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html). Right: first column shows a few test images and next to each are top 10 nearest neighbors in the training set according to pixel-wise difference.
</span>
<br><br>
**Distance Metric.** One of the simplest possibilities is to compare the images *pixel by pixel* and add up all the differents. In other words, given two images and representing them as vectors $$ I_{1}, I_{2} $$, a reasonable choice for comparing them might be the **L1(Manhattan) distance(Taxicab Geometry):**
<br> 
<center>$$ d_{1}\left(I_{1}, I_{2}\right)=\sum_{p}\left|I_{1}^{p}-I_{2}^{p}\right| $$ </center>
<br>
If two images are identical the result will be zero, but if images are very different the result will be large.
<br><br>
Another common choise could be to instead use the **L2(Euclidean) distance**, which has the geometric interpretation of computing the euclidean distance between two vectors. The distance takes the form:
<br>
<center>$$ d_{2}\left(I_{1}, I_{2}\right)=\sqrt{\sum_{p}\left(I_{1}^{p}-I_{2}^{p}\right)^{2}} $$</center>
<br>
<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Manhattan_distance.svg/1280px-Manhattan_distance.svg.png" alt="drawing" width="250"/></center>
<span style="color:grey">Taxicab geometry versus Euclidean distance: In taxicab geometry, the red, yellow, and blue paths all have the same shortest path length of 12. In Euclidean geometry, the green line has length $$ 6 \sqrt{2} \approx 8.49 $$ and is the unique shortest path.
</span>
<br><br>
**L1 vs. L2.** In particular, the L2 distance is much more unforgiving than the L1 distance when it comes to differences between two vectors.  That is, the L2 distance prefers many medium disagreements to one big one.
<br>
**L1 distance depends on the choice of coordinates system.** If the input features, the individual entries in the vector have some important meaning for the task, then maybe somehow L1 distancemight be a more natural fit, since if we rotate the coordinate frame, that would actually change the L1 distance between the points, while L2 distance doesn't changed. **In L1 distance, decision boundaries tend to follow the coordinate axis.**
<br>
**L2 distance doesn't depends on coordinates system.** If it's just a genetic vector in some space and we don't know which of the different elements, L2 distance is slightly more natural. **In L2 distance, it just put the boundaries they should fall naturally.**
<br><br>
**Evaluation criterion.** It is common to use **accuracy**, which measures the fraction of predictions that were correct.
<br><br>
**k-Nearest Neighbors.** Instead of copying label from nearest neighbor, k-Nearest Neighbors takes **majority vote** from **k** cloest points. The choose of k tends to **smooth out the decision boundaries** and leads to **better results**. If we use a larger value of **k**, voting top 3 or 5, that would be end up being a lot robust. 

<img src="http://cs231n.github.io/assets/knn.jpeg" alt="knn">

<span style="color:grey">An example of the difference between Nearest Neighbor and a 5-Nearest Neighbor classifier, using 2-dimensional points and 3 classes (red, blue, green). </span>
<br><br>
**Pros & Cons.** Actually if we expect the k-Nearest Neighbor classifier to work well, we kind of need our training example to cover the space quite densely, so this classifier on images never used.

Pros:

* simple to implement and understand
* takes no time to train

Cons:

* pay that computational cost at test time
* require a coparison to every single training example

<br><br>

# Validation sets for Hyperparameters tuning
<br>The value of **k** and the choice of **distance** are tuning of **hyperparameters:** choices about the algorithm that we **set** rather than learn from the training data. And this is very **problem-dependent** and we must try them all out and see what works best. In general, we try both L1 and L2 distance and see what works better. 
<br><br>
**Validation set.** As we cannot use the test set for the purpose of tweaking hyperparameters, and we also need to prevent **overfit** to the training set, we split the training set in two: a slightly smaller training set and the validation set. We don't care about the accuracy on training set and the real performance of the model depends on its accuracy on the test set, which is only used once at the end. So we need to validate the choices of hyperparameters on the validation set and pick the best set of hperparameters, then test it on the test set.
<br><br>
**Cross-validation.** When the size of training data is **small**, we sometimes use a more sophisticated technique for hyperparameter tuning called **cross-validation**. In this method, we can get a better and less noisy estimate of how well a certain value of *k* works by **iterating over different validation sets and averaging the performance across these**. In practice, people prefer to avoid cross-validation in favor of having a single validation split, since it can be **computationally expensive**. If we have a small data, using cross-validation. If we have a big data, using big validation splits is enough.

<br><br>
# Linear Classification
<br>To image classification that we will eventually naturally extend to entire Neural Networks and Convolutional Neural Networks, we develop a more powerful approach, which has two major components: a **score function( $$ f $$ )** that maps the raw data to class scores, and a **loss function( $$ L $$)** that quantifies the agreement between the predicted scores and the ground truth labels. We will then cast this as an **optimization problem** in which we will **minimize the loss function** with respect to the parameters of the score function.
<br><br>
**Score function.** We assume a training dataset of images $$ x_{i} \in R^{D} $$, each associated with a label $$ y_{i} $$. Here $$ i=1 \ldots N $$ and $$ y_{i} \in 1 \ldots K $$. That is, we have **N** examples (each with a dimensionality **D**) and **K** distinct cagegories. We define the score function $$ f : R^{D} \mapsto R^{K} $$ that maps the raw image pixels to class scores.
<br><br>
**Linear classifier.** In practice, the score function $$ f $$ can be complex, however in this model we will start out with arguably the simplest possible function, a single linear mapping:
<center>$$ f\left(x_{i}, W, b\right)=W x_{i}+b $$</center>
In the above equation, the parameters in matrix **W** (of size [K * D]) are often called the **weights**. Vector **b** (of size [K * 1] is called the **bias vector** since it influences the output scores without interacting with the actual data $$ x_{i} $$. In matrix **W**, it contains **N** classifiers, where each classifier is a row of **W**, we call each row of **W** corresponds to a *template*, also called *prototype*. 
<br>
Training data is used to learn parameters **W,b** instead of the original images data, so once the learning is complete we can discard the entire training set and only keep the learned parameters. Then we only need to do **a single matrix multiplication** to classify the test image, which is significantly faster than comparing a test image to all training images.
<br>

<img src="http://cs231n.github.io/assets/imagemap.jpg" alt="imagemap">
<span style="color:grey">An example of mapping an image to class scores. </span>

A linear classifier computes the score of a class as a **weighted sum of all of its pixel values across all 3 of its color channels** and depends on what values we set for these weights. Images are stretched into **high-dimensional column vectors** and we can interpret **each image as a single point in this space**. With this terminology, the linear classifier is **doing template matching**, where the templates are learned. 
<br>
In CNN, we implied different weights in different layers, so neural network in different layers can **extract different features** of images, which allows us to distinguish different classes of images.
<br><br>
**Bias trick.** Bias in the score function: combine the two sets of parameters(weights and bias) into a single matrix that holds both of them by extending the vector $$ x_{i} $$ with one additional dimension that always holds the constant $$ 1 $$ - a default *bias dimension*. So the new score function can simplify to a single matrix multiply:
<center>$$ f\left(x_{i}, W\right)=W x_{i} $$</center>

<img src="http://cs231n.github.io/assets/wb.jpeg" alt="wb">
<span style="color:grey">Illustration of the bias trick. </span>
<br><br>
**Image data preprocessing.** It is important to **center your data** by subtracting the mean from every feature. In the case of images,this corresponds to computing a *mean image* across the training images.

<br><br>
# Loss Function
<br>In order to control over the weights and measure our unhappiness with outcomes such as this one with a **loss function(cost function/objective)**. 
<br><br>
<h4>Multiclass Support Vector Machine loss(SVM loss)</h4>
The SVM loss is set up to that the SVM "wants" **the correct class for each image to a have a score higher than the incorrect classes by some fixed margin $$ \Delta $$**. We are given the pixels of image $$ x_{i} $$ and the label $$ y_{i} $$ that specifies the index of the correct class. The score function $$ f $$, takes the pixels and computes the vector $$ f\left(x_{i}, W\right) $$ of class scores, which we will abbreviate to $$ s $$. So the score for the j-th class is the j-th elements: $$ s_{j}=f\left(x_{i}, W\right)_{j} $$ and the Multiclass SVM loss for the i-th example is formalized as:
<center>$$ L_{i}=\sum_{j \neq y_{i}} \max \left(0, s_{j}-s_{y_{i}}+\Delta\right) $$</center>

$$ s_{j} $$ is the score of each incorrect class and $$ s_{y_{i}} $$ is the score of the correct class.

Working with linear score function $$ f $$, the loss function is in this equivalent form:
<center>$$ L_{i}=\sum_{j \neq y_{i}} \max \left(0, w_{j}^{T} x_{i}-w_{y_{i}}^{T} x_{i}+\Delta\right) $$</center>

The threshold at zero $$ \max (0,-) $$ function is often called the **hinge loss**. 
<br><br>
**Regularization.** In order to avoid overfitting, we introduce a **regularization penalty** $$ R(W) $$ to make it cannot fit the training data very well. We wish to encode some preference for a certain set of weights **W** over others to remove this ambiguity. So we extend the loss function with this **regularization penalty** $$ R(W) $$. The most common regularization penalty is the **L2** norm that discourages large weights through an elementwise quadratic penalty over all parameters:
<center>$$ R(W)=\sum_{k} \sum_{l} W_{k, l}^{2} $$</center>

It is **summing up all the squared elements of** $$ W $$ . Notice that the regularization function is not a function of the data, **it is only based on the weights**.
<br><br>
Now the loss function is made up of two components: the **data loss** and the **regularzation loss**, so the Multiple SVM loss becomes:
<center>$$ L=\underbrace{\frac{1}{N} \sum_{i} L_{i}}_{\text { data loss }}+\underbrace{\lambda R(W)}_{\text { regularization loss }} $$</center>
Where $$ N $$ is the number of training examples and the regularization penalty is weighted by a hyperparameter $$ \lambda $$. L2 penatly leads to the appealing **max margin** property in SVMs. It prefers smaller and more diffuse weight vectors, the final classifier is encouraged to take into account **all input dimensions** to **small amounts** rather than a few input dimensions and very strongly. This effect can **improve the generalization performance** of the classifiers on test images and lead to less *overfitting*.
<br><br>
Note that **biases** do not have the same effect since they do not control the strength of influence of an input dimension. Therefore, it is common to **only regularize the weights $$ W $$** but not the biases $$ b $$. And we can **never** achieve loss of exactly 0.0 on all examples since this would only be possible in the pathological setting of $$ W=0 $$.
<br><br>
**Setting delta.** $$ \Delta $$ can safely be set to $$ \Delta=1.0 $$ in all calss. The hyperparameters $$ \Delta $$ and $$ \lambda $$ in fact **control the same tradeoff**: the tradeoff between the data loss and the regularization loss in the objective. And the magnitude of the weights $$ W $$ has direct effect on the scores: the exact value of the margin between the scores is in some sense meaningless because the weights can shrink or stretch the differences arbitrarily. The Multiple SVM loss doesn't changed significantly but it may influence the regularizataion loss a lot.
<br><br>
<h4>Softmax classifier</h4>
Compared with the binary Logistic Regression classifier(Sigmoid), Softmax classifier is its generlization to multiple classes. It gives a slightly more intuitive output(normalized class probabilities) and also has a probabilitistic interpretation that we will describe shortly. 
<br><br>
In Softmax classifier, the function mapping $$ f\left(x_{i} ; W\right)=W x_{i} $$ stays unchanged, but we now interpret these scorese as the **unnormalized log probabilities for each class** and replace the *hinge loss* with a **corss-entropy loss**.
The function $$ f_{j}(z)=\frac{e^{z j}}{\sum_{k} e^{z^{z} k}} $$ is called the **softmax function**: it takes a vector of arbitrary real-valued scores (in $$ z $$) and squashes it to a vector of values between zero and one that sum to one (normalization). It takes over all of scores, **exponentiate them** so that now they become positive, then renormalizes them by the **sum of those exponents** and ends up with **probability distribution**. 
<br><br>
**cross-entropy loss.** For the loss function of Softmax classifier, the target is that the probability of true class is high and close to one. So it wants to maximize the log likelihoood(**Maximum Likelihood Estimation MLE**), or to minimize the negative log likelihood of the correct class since it will be easier to maximize the log function than the raw probability.(maximize $$ \log P $$ of correct class, so minimize the $$ - \log P $$):
<center>$$ L_{i}=-\log \left(\frac{e^{f_{j_{i}}}}{\sum_{j} e^{f_{j}}}\right) $$</center>
where $$ f_{j} $$ is the j-th element of the vector of class score $$ f $$. Usually at initialization **W** is small so all $$ s \approx 0 $$, the loss is $$ - \log \frac{1}{C} = \log C$$, where C is the number of categories.
<br><br>
So the step of Softmax classifier is:

* calculate the score (raw probabilities, also called **unnormalized log probabilities**).
* exponentiate these score, get the **unnormalized probabilities**.
* normalize them between zero and one that sum to one, get the **probabilities**.
* then use these probabilities to calculate the loss function and minimize the result.

**MLE&MAP.** Minimizing the negative log likelihood of the correct class can also be interpreted as performing **Maximum Likelihood Estimation (MLE)**. We can also interpret the regularization term $$ R(W) $$ in the full loss function as coming from a Gaussian prior over the weight matrix $$ W $$, where we are performing the **Maximum a posteriori (MAP)** estimation. 
<br><br>
