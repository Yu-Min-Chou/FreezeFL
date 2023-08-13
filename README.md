# FreezeFL
Federated learning is an emerging paradigm that enables edge devices to collaboratively train a model without sharing private data. There are two key challenges in the setting. 1)~system heterogeneity, the significant variability in hardware resources on edge devices. 2)~statistical heterogeneity, non-identically distributed data on edge devices. In this work, we propose a framework, FreezeFL, which alleviates the system heterogeneity by applying layer freezing and a fairness guaranteed client selection algorithm. We further show that FreezeFL does not exacerbate statistical heterogeneity while solving system heterogeneity and keeps client-oblivious; namely, clients do not need to share information besides model weights. Extensive experiment results on FEMNIST and Shakespeare show that FreezeFL outperforms various existing methods. Relative to FedAvg, FreezeFL shortens round length by 33\% and 46\%, respectively, while not hurting test accuracy.

## Overview
The overview of the proposed framework is shown as below figure. During the training process, the server maintain soft deadline $T$, which means the estimated average model exchange time. For the clients that take longer than the soft deadline, we freeze partial layers of their local model to save computation and communication costs. The slower the clients are, we freeze more layers for them. The key challenge in our approach is to decide the number of frozen layers of each client during the learning process. Since statistical heterogeneity will deteriorate and degrade model performance if we freeze too many layers rashly. On the other hand, system heterogeneity will not be improved if we freeze too few layers. In other words, the decision on the number of frozen layers is the trade-off between statistical and system heterogeneity. Therefore, the goal of our work is to propose a layer freezing strategy that can minimize additional statistical heterogeneity while solving the system heterogeneity problem. For the freezing strategy, we take multiple factors into consideration, including the system heterogeneity, quality of local data, and the importance of model layers.
![image](https://github.com/Yu-Min-Chou/FreezeFL/assets/42434345/98e85f0a-191a-4f0f-91dc-d646887af872)

## Implementation
1. Data generation
    1. Use [LEAF](https://github.com/TalwalkarLab/leaf/tree/master) to generate data
    
    ```bash
    ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample // for femnist
    ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8 // for shakespeare
    ```

2. Training
    1. There are three baselines(FedAvg, Fjord, RBCSF). All of them are implemented with LEAF framework. In this section, we will first go through LEAF framework, then detail the implementation of the baselines and FreezeFL
    2. LEAF framework (Take FedAvg as an example)
        1. Server operations
            1. The class is defined in ../Freezing/FedAvg/server.py
            2. Sever perform **random client selection** with the below function
                1. The parameter ***possible_clients*** refers to the client ready to perform local training. In FedAvg, we assume all clients are always online.
                2. The parameter ***num_clients*** refers to the number of clients should be selected in each round.
                
                ```bash
                def select_clients(self, cur_round, possible_clients, num_clients=20)
                ```
                
            3. Lets clients perform **local training**
                1. Here, the server let clients perform training sequentially with the GPU with the specific ***GPU_id***. The detail implementation of local training are described in the next subsection(Create clients)
                2. The parameter ***num_local_epoch*** refers to the number of local epochs
                
                ```bash
                def train_model(self, num_local_epochs=1)
                ```
                
            4. **Aggregate all updates** 
                1. This function implements basic weighted averaging
                
                ```bash
                def update_model(self)
                ```
                
            5. **Evaluate the accuracy of the global model**
                1. The parameter clients are used to pass testing dataset to the server.
                
                ```bash
                def test_model(self, clients)
                ```
                
        2. Client operations
            1. The class is defined in ../Freezing/FedAvg/client.py, but the functions used to setup and create clients are defined in ../Freezing/FedAvg/main.py
            2. **Setup clients**
                1. Distribute data
                    1. In the function ***setup_clients,*** we should specify the path to the training dataset and testing dataset(created with LEAF)
                2. Assign computation and communication capabilities
                    1. In the function create_clients, we assign capabilities to the clients, the capabilities follow uniform distribution and bound by three parameters
                        1. CAP_TIERS: the number of tiers of capabilities, the clients belong the same tier have same capabilities.
                        2. CAP_MAX
                            1. The maximum value of capabilities
                        3. CAP_MIN
                            1. The minimum value of capabilities
            3. **Local training**
                1. Just call the training function defined in cnn.py or rnn.py
                2. In addition, the time for computation and communication are also recorded by the client and return it to the server.
                    1. The training time of a model is simulated by the profiling result
    3. All the training procedure are implemented in ../Freezing/FedAvg/main.py
        1. The args of main.py include
            1. dataset
            2. model
            3. clients-per-round: number of selected clients of each round
            4. num-rounds: total training round
            5. num-local-epochs: number of local epochs
            6. GPU-id: the GPU used for training
        2. The hyper-parameters of the main.py are defined in ../Freezing/…/baseline_constants.py
            1. Learning rate, batch_size, num_classes for each dataset
            2. Besides CAP_TIERS, CAP_MIN, CAP_MAX, there are also BASIC_COMPUTATION and BASIC_BANDWIDTH. The two parameters are used to adjust the ratio of time for computation and communication. The default setting let the time for computation and communication equal.
        
        ```bash
        python main.py -dataset femnist -model cnn
        ```

3. FreezeFL
    1. Adaptive Layer Freezing
        1. This optimization is defined in ../Freezing/Layerwise/model/cnn.py or rnn.py
        2. In the below function, we first set all layers trainable and perform a single round of training to calculate the utility of each layer. Then use the formulation in the FreezeFL paper to decide the number of frozen layers. Finally, frozen the partial layers and finish the local training. 
    
        ```bash
        def freeze_train(self, dataset_train, num_local_epochs, soft_deadline, comp_ability, comm_ability, num_samples, profile_result)
        ```
    
    2. Layerwise aggregation
        1. This optimization is defined in ../Freezing/Layerwise/model/server.py
        2. This optimization only aggregate the unfrozen layers
      
        ```bash
        def update_model(self)
        ```
      
    3. Fairness-guarantee client selection
        1. This optimization is defined in ../Freezing/Layerwise/model/server.py
        2. Use the formulation in the FreezeFL paper to update the utility of clients and select clients based on the utilities
      
        ```bash
        def update_utility(self)
        ```
