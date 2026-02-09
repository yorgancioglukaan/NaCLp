# The public repository containing the code for the paper "Network Augmentation for Cold-Start Link Prediction"

 - The conda environment used is provided in the environment.yaml file
 - Network augmentation can be performed as follows:

```
# for a pytorch geometric data and a list of cold-start query node ids, you can perform cold-start test/train/val splits

train_data, val_data, test_data = node_query_based_split(data, query=query, train_ratio=train_ratio,val_ratio=val_ratio)

# then use the augmentation module

enhancer = NetworkEnhancer(train_data)
enhancer.train(epochs=enhance_epocs, neg_ratio=5, train_ratio=0.1, sampling_scheme='random', lr=0.01, query=train_data.query_nodes)
enhanced_train_data = enhancer.enhance(target_nodes=train_data.query_nodes, enhance_k=k)

# a user determined list of training set can be provided to train the augmentation model on

enhancer.train(train_data=user_train, epochs=enhance_epocs, neg_ratio=5, train_ratio=0.1, sampling_scheme='random', lr=0.01, query=train_data.query_nodes)

```
