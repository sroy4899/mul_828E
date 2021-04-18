# About

This is a repository dedicated to the investigation of machine unlearning techniques.  

As a start, we intend to create some optimal architectures specifically for the seminal machine learning datasets MNIST and CIFAR and involve the usage of a convolutional neural net. 

From there, we intend to investigate some of the most popular techniques for machine learning present in the literature. Exact methods will not be employed. 

In full, we intend to test  
1) naive unlearning via retraining from scratch  
2) SISA untraining 
3) Newton-based methods à la Guo et al. 
4) DeltaGrad 
5) Linear filtration à la Baumhauer et al.  
6) Projective residual update 
7) Obfuscation 

This is all definitely bounded by the amount of time that we have in order to complete the project. 

To further stratify, there is also the question of how we measure success in this case. To this end, we intend to stratify the number of datapoints we delete (or simplify down to deleting only a single datapoint or subset of datapoints). In terms of success, we could rely on the theoretical guarantees provided by the papers, but future work might involve generating threat models and evaluating these models against those threats. 


