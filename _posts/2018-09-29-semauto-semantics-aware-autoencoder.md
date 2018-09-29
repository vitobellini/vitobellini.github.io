---
layout: post
title:  "SEMAUTO: Semantics-Aware Autoencoder in Recommendation Scenario"
date:   2018-09-29 13:00:00 +0200
categories: posts
published: false
---

## Introduction

Artificial Neural Networks are very powerful model that can obtain an high discrimination power but the price you pay for better accuracy is the lack of interpretability. That's the reason why they are called black-boxes; they just work but you cannot understand how they compute the predictions. Neural Networks can approximate any kind of function but the study of their structure won't give us any insight about the function being approximated, because there is no simple link between the weights and the function to approximate.

Autoencoders are widely and successfully used in collaborative filtering settings. The most common configuration consists in having input and output neurons that represent catalog's items. This network is then trained with user ratings and it learns how to reconstruct them by using a latent representation which is encoded in the hidden layer. Even with this simple model, its possible to outperform many state of the art algorithms but the price you pay for better accuracy is the lack of interpretability.

Users, on the other hand, would like to know why a certain item has been recommended. This may be important to users from different perspectives. For example, it can help them to better understand how the system is working and it can increase their trust in the recommender.

### Idea

If it could be possible to label every neuron in hidden layers and force the neural network to be aware of the hidden nodes meaning, we could address the problem of deep learning models interpretability. In a recommendation scenario, if we think at an autoencoder, usually we have input and output units representing items while hidden units encode a latent representation of users' ratings once the model has been trained. What if we find a way to replace a latent representation of user ratings with items' attributes and force the gradient to flow only through those attributes that belong to items?

Therefore, we came up with a not fully connected architecture based on an autoencoder model, in which input and output neurons that represent all the items in the catalog are connected only with those neurons that represent items’ attributes related to them.

![SEMAUTO](/assets/2018-09-29/semauto.png)

Autoencoders are capable of encoding a latent representation of their input data within the hidden layer and they exploit it to reconstruct the original input data at the output layer. In our case, we have a not fully connected architecture that allows us to assign an explicit meaning to the hidden neurons. This means that at the end of training, we have encoded an explicit representation of the input data in the feature space.
