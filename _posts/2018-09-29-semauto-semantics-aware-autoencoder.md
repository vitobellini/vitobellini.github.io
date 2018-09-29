---
layout: post
title:  "SEMAUTO: Semantics-Aware Autoencoder in Recommendation Scenario"
date:   2018-09-29 13:00:00 +0200
categories: posts
published: false
---

## Introduction

Artificial Neural Networks are very powerful model that can obtain an high discrimination power but the price you pay for better accuracy is the lack of interpretability. That's the reason why they are called black-boxes; they just work but you cannot understand how they compute the predictions. Neural Networks can approximate any kind of function but the study of their structure won't give us any insight about the function being approximated, because there is no simple link between the weights and the function to approximate.

### Idea

If it could be possible to label every neuron in hidden layers and force the neural network to be aware of the hidden nodes meaning, we could address the problem of deep learning models interpretability. In a recommendation scenario, if we think at an autoencoder, usually we have input and output units representing items while hidden units encode a latent representation of users' ratings once the model has been trained. What if we find a way to replace a latent representation of user ratings with items' attributes and force the gradient to flow only through those attributes that belong to items?
