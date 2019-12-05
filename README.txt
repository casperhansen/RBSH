This contains the code for the paper:
Casper Hansen, Christian Hansen, Jakob Grue Simonsen, Stephen Alstrup, and Christina Lioma. 2019. Unsupervised Neural Generative Semantic Hashing. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'19). ACM, New York, NY, USA, 735-744. DOI: https://doi.org/10.1145/3331184.3331255

Due to space limitations on github, we cannot upload our processed data, so we provide this google drive link instead: https://drive.google.com/drive/folders/1rwMfV7oOk6P0O3qSgu1LznJa1WyKx6FK?usp=sharing

Code structure.
Using our provided processed datasets you can easily rerun the experiments within our paper. The main file, "our_method.py", runs everything and you should essentially only change the following parameter lines (described in our paper):
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument("--rank_inc", default=10, type=float)
    parser.add_argument("--rank_val", default=0.5, type=float)
    parser.add_argument("--KL_inc", default=5, type=int) 

Our neural network model is implemented in graphs_weak.py, and the used generators for feeding the network with data is in generators.py
