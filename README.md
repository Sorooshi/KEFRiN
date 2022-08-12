# KEFRiN

The source codes, experimental test-beds and, the datasets of our paper entitled "Community Partitioning over Feature-Rich Networks Using an Extended K-Means Method" by [Soroosh Shalileh](https://www.hse.ru/en/staff/srshalileh) and [Boris Mirkin](https://www.hse.ru/en/staff/bmirkin), submitted to the Journal of Entropy.



For more information on how to call our algorithms "KEFRiNe or "KEFRiNc or KEFRiNm" one can refer to the demo jupyter notebooks "demo.ipynb".

Also thess algorithms can be run through the terminal by calling: 
            
    For using Euclidean distance:       
        python KEFRiN.py --Name="name of dataset in data dir" --PreProcessing="z-m" --Run=1 --Euclidean=1  --Cosine=0  --Manhattan=0
        
    For using Cosine distance:    
        python KEFRiN.py --Name="name of dataset in data dir" --PreProcessing="rng-u" --Run=1 --Euclidean=0  --Cosine=1  --Manhattan=0
        
    For using Manhattan distance:    
        python KEFRiN.py --Name="name of dataset in data dir" --PreProcessing="rng-u" --Run=1 --Euclidean=0  --Cosine=0  --Manhattan=1

To restore the save results, run the above comments and set --Run to zero.




Remark 1: I will provide a pip installation of this software.

Remark 2:  I will add this algorithm to CDI Lib.
