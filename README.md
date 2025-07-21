# Enabler--A summer project by Ben Kurzion

## Problem Statement
A lot of people are colorblind and it makes crafting an outfit a pain in the butt. Sometimes, you get lucky and choose cohesive items, but more often than not, your pants clash with your shirt and you have no idea how silly you look. 
Rather than have to contend with random chance, Enabler helps colorblind people identify colors and calculates whether two items clash with one another!

## Technical Description
When the user submits an image to the Enabler, the most computationally challenging task is to separate the foreground from the background. Given an image of a shirt, an ideal algorithm will be able to separate each pixel in the image as part of the shirt or not part of the shirt. This way, when evaluating whether or not two items clash, the Enabler will know the true color of each item without noise from the background pixels. This is called image segmentation, and I have implemented two approaches to try to segment effectively and accurately.  


### The Min-Cut Approach
Given a connected graph network with weighted edges between nodes, an s-t cut is defined as a partition of the nodes in the graph such that a source node $s$ and a sink node $t$ are not in the same partition. A result of such cut is the set of edges that connect nodes in separate partitions. The capacity of such cut is the sum of the edge weights for all the edges crossing the cut. A minimum cut is an s-t cut with minimum capactiy. 

If we model each pixel as a node in a graph with edges formed between adjacent nodes, we can run a minimum cut algorithm (Ford-Fulkerson) on the image. Pairs of pixels with similar RGB values will have higher edge weights than pixels with dissimilar RGB values. With enough difference between the foreground and the background, edges between the foreground and the background will be cut, leaving a clean segmentation. 

### The Clustering Approach
In k-means clustering, $k$ clusters are randomly placed in among the data points. Iteratively, data points are assigned to their nearest cluster and the cluster center is updated to the center of all of the points assigned to it. This continues until there is no change in cluster membership. 

Since a cursory glance at RGB values offers no indication of where the background versus the foreground pixels might be, we can't just randomly place 2 cluster centers in the image and run the clustering algorithm. Instead, I initialized every pixel to be its own cluster. Then, using the same similarity measure as discussed above, the two most similar clusters are combined and their RGB values averaged. This process is repeated iteratively until there are only 2 clusters remaining. Ideally, these would be the foreground and the background. 

## Technical Challenges
A phone camera will typically take a 12 megapixel image which contains $4000$ x $3000$ pixels. For any algorithm that runs in $O(n^2)$ or worse, this is an unfeasible number of pixels. Even when the image is scaled down to $128$ x $128$ pixels, these algorithms are very cumbersome and ineffectual. 

Furthermore, given any situation when the background is not starkly different from the foreground, the algorithms fail to separate the object. Realisitically, images that users captured will be far from optimal, likely closer to maximally adversarial. 

## Next Steps
Companies like Meta have released zero-shot segmentation models that work very well. As expected, deep learning defeats classical algorithms. To produce a truly effective product, I will need to play around with publically available models. 