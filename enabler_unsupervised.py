'''
Rather than do a fancy algorithm that straight up sucks, let's try and unsupervised, clustering based approach to segmentation.
Under the assumption that there exists exactly one object in the foreground and the rest of the image is background, we can do some type of agglomerative clustering
until there exist only 2 discrete classes. 
'''