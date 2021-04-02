# KEFRiN

The source codes, experimental test-beds and, the datasets of our paper titled "K-Means Clustering Extended to Community Detection in Feature-Rich Networks" by [Soroosh Shalileh](https://www.hse.ru/en/org/persons/316426865) and [Boris Mirkin](https://www.hse.ru/en/staff/bmirkin), submitted to the Journal of Pattern Recognition.



For more information on how to call our algorithms "KEFRiNe or "KEFRiNc," one can refer to the demo jupyter notebooks "demo.ipynb".

Also thess algorithms can be run through the terminal by calling: 
            
    For using Euclidean distance:       
        python KEFRiN.py --Name="name of dataset in data dir" --PreProcessing="z-m" --Run=1 --Euclidean=1  --Cosine=0
        
    For using Cosine distance:    
        python KEFRiN.py --Name="name of dataset in data dir" --PreProcessing="rng-u" --Run=1 --Euclidean=0  --Cosine=1

To restore the save results, run the above comments and set --Run to zero.

Note that the above method for calling our proposed algorithms requires the dataset to in .pickle format as provided in the data directory.

For generating similar synthetic data sets, One should call "synthetic_data_generator.py" as this is demonstrated in Jupyter notebook "MediumSize_synthetic_data.ipynb".
