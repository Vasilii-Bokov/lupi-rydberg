# lupi-rydberg
To run the transformer: 

0. In all the files below set the same N and the corresponding M (1000 or 5000 or more)
1. python3 generate_multiple_datasets(_bd).py
2. ./submit_all_datasets(_bd).sh
3. ./submit_accuracy_jobs(_bd).sh
4. python3 collect_accuracy_results(_bd).py


To check the progress, do:

squeue -u user_name
