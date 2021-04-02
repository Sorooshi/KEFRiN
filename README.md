# KEFRiN

The source codes, experimental test-beds and, the datasets of our paper titled "K-Means Clustering Extended to Community Detection in Feature-Rich Networks" by Soroosh Shalileh and, Boris Mirkin submitted to the Journal of Pattern Recognition.

For more information on how to call our algorithms "KEFRiNe or "KEFRiNc" one can refer to any the demo jupyter notebooks "Demo_clustering_results_Lawyers".

Also this algorithm can be run through the terminal by calling: python KEFRiNe.py/KEFRiNc.py --Name="name of dataset" --PreProcessing="z-m" --Run=1

To restore the save results, run the above comments and set --Run to zero.

Note that the above method for calling our proposed algorithms requires the dataset to in .pickle format as it provided in data directory.

For generating similar synthetic data sets, One should call "synthetic_data_generator.py" as this is demonstrated in Jupyter notebook "MediumSize_synthetic_data.ipynb".
