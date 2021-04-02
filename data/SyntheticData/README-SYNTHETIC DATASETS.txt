

In the root directory, there exist two child directories namely, "synthetic data" and "synthetic data(limited)". 

Files in "synthetic data" directory are devoted to data sets as described in our paper. 

Files in "synthetic data (limited)" directory are devoted to the case that we tried to limit the number of between communities edges, at most, equal to the total number of within-community edges in a complete graph multiplied by the probability of between community edges. 

In each child directory, you will find a .py file. This file is the main file for generating the synthetic datasets. This file is called in jupyter notebook for 1) to generate and save the data 
a folder, 2) to reload and visualize the generated data.

Note: More descriptions can be found inside the jupyter notebooks.
Note: To load the data set one does not need this files and it is provide for the sake of completeness.


Datasets are stored in .pickle files. Each .pickle file pursues the following naming convention.

Naming convention: 
	First letter S/M denotes the network size i.e Small/Medium;
	Second letter Q/C/M denotes the features type i.e Quantitative,
	Categorical, Mix of Quantitative and Categorical

	Within the parenthesis, three numbers are shown separated by commas. 
	The first number shows the total number of nodes, the second one shows
	the number of features and the third number represents the number of communities. 



Each .pickle file contains 8 different settings and each setting is stored as a dict.
Moreover, each separate setting is a dict of dict which is repeated 10 times.
And each of these repeats also stored as dict containing the four pairs of keys and values.
	
	1) ['GT'] := A list integer, containing the number of nodes in the first to last community summing to total number of nodes, e.g [30, 40, 50, 50, 30]
	
	2) ['Y']  := A numpy array, entity-to-feature matrix

	3) ['Yn'] := A numpy array, noisy version of entity-to-feature matrix 

	4) ['P']  := A numpy array, adjacency matrix

The code below can be used to access datasets in a .pickle file:


# name_of_datasets.pickle :="SC(200, 5, 5)" or SQ(200, 5, 5) or any other ones
with open(os.path.join('path_to_datasets', 'name_of_datasets.pickle'), 'rb') as fp:
    SAN = pickle.load(fp)

	for setting, repeats in SAN.items():

	     for repeat, matrices in repeats.items():

	     	GT = matrices['GT']    # Ground Truth
        
          	Yin  = matrices['Y']   # Entity-to-feature matrix
          
         	Ynin = matrices['Yn']  # Noisy Entity-to-feature matrix
        
         	Pin  = matrices['P']   # Adjacency matrix


Each setting consists of three numbers separated with commas.
The first number represents the probability of drawing an edge within a community (p in the paper). The second number represents the probability of drawing an edge between communities (q in the paper). And the third number shows the cluster intermix in quantitative feature case (alpha in the paper) or the feature homogeneity threshold (epsilon in the paper).


Finally, it is noteworthy to mention that the structure of the saved ground truth is not appropriate for applying Sklearn metrics, e.g ARI or NMI. As It is mentioned previously, this is a list of integers denoting the number of nodes in the first community, in the second community to the last community. To preprocess this list one can run the below-mentioned function (flat_ground_truth(ground_truth)) to format it in a way that is appropriate for applying the ARI, NMI or other sklearn metrics. This function can be found at the end of the main file for generating synthetic data as well as the below.



def flat_ground_truth(ground_truth):
    """
    :param ground_truth: the clusters/communities cardinality
                        (output of cluster cardinality from synthetic data generator)
    :return: two flat lists, the first one is the list of labels in an appropriate format
             for applying sklearn metrics. And the second list is the list of lists of
              containing indices of nodes in the corresponding cluster.
    """
    k = 1
    interval = 1
    labels_true, labels_true_indices = [], []
    for v in ground_truth:
        tmp_indices = []
        for vv in range(v):
            labels_true.append(k)
            tmp_indices.append(interval+vv)

        k += 1
        interval += v
        labels_true_indices += tmp_indices

    return labels_true, labels_true_indices
