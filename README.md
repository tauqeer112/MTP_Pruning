### Exploring the Impact of Network Pruning Techniques on Model Efficiency and Layer Similarity in Medical Image Analysis

## Authors

- [Tauqeer Akhtar](https://github.com/tauqeer112)
- [Dr. Deepti R. Bathula](https://www.iitrpr.ac.in/deepti-r-bathula)

## Abstract
Network pruning is a technique used to reduce the size of deep neural networks by removing unimportant or redundant connections and parameters. This technique make large models more computationally efficient without sacrificing accuracy. Inspired by this technique, we explore the performance of different network pruning techniques that are one-shot, layerwise, and iterative pruning, on state-of-the-art models VGG16 and DenseNet169. In our work, we present an approach in which we observe that iterative pruning outperforms one-shot and layer-wise pruning both in terms of qualitative and quantitative performance. We also visualized the features learned by the layers of the model while implementing these pruning techniques and observed that the features learned by one-shot and iterative pruning are similar but different compared to layerwise pruning. Extensive experiments were conducted to evaluate the performance of our proposed approach, using two datasets: CIFAR100 and HAM10000.

## Problem Statement

In this study, we aim to investigate the effectiveness of three different pruning methods,namely iterative, layerwise, and one-shot pruning, on the efficiency of neural networks.

#### Itarative Pruning
Pruning is done in steps, where after each step, the network is fine-
tuned to recover its lost performance. This process is repeated until the desired sparsity
level is achieved.

![iterative](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/Iterative.png?raw=true)

#### One-Shot pruning
The entire network is pruned at once to the desired sparsity level
then fine-tuned only once to recover its lost performance.

![oneshot](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/Oneshot.png?raw=true)

#### Layerwise pruning
Each layer is pruned to the desired level of sparsity, and the
model is finetuned after each layer pruning. The overall desired global pruning level is achieved when all layers have been pruned.

![layerwise](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/Layerwise.png?raw=true)



## Datasets and Models
### Datasets
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW8)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)


### Model and Dataset Combinations
![Models](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/dataset_prung.png?raw=true)


VGG16 was chosen due to its large number of parameters and robustness to L1 unstructured pruning, making it suitable for comparing different pruning methods.

DenseNet169 was chosen as a smaller network with less than 10% of VGG16's parameters, making it a suitable comparison model.

One Computer vision dataset CIFAR100 and one medical Image dataset HAM10000 was taken.

## Results
### Accuracies with different datasets, models and pruning technique
![Results](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/results.png?raw=true)

Our findings indicate that iterative pruning generally outperformed the other two pruning
methods, consistently achieving better results across various sparsity levels

### Layerwise pruning effect on accuracy

![layerwise](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/layerwise_effect.png?raw=true)

We observed a significant decrease in the performance of the pruned models as we in-
creased the number of layers being pruned

### Comparison of features learned by layers with different pruning techniques

![cka](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/cka.png?raw=true)

Our findings reveal that layerwise pruning resulted in maximum feature similarity be-
tween the layers, suggesting that this type of pruning may result in the model learning fewer
distinct features compared to iterative and one-shot pruning as shown in CKA above.

![vs](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/vs.png?raw=true)

Our results showed that the layers of the models pruned by one-shot and iterative pruning learned similar types of features, but differed significantly from those learned by the layers of the models pruned by layerwise pruning

![vggprun](https://github.com/tauqeer112/MTP_pruning/blob/main/Images/vgg_prun.png?raw=true)

Our experiments also revealed that the pruned VGG16 model performs better during
training than the unpruned model, regardless of the dataset used (HAM10000 and CIFAR100)




## Conclusions

In conclusion, we conducted a comparative study on different pruning methods, and evaluated their effects on both performance and feature similarity. Our experiments demonstrated that iterative pruning is a promising technique for model compression, outperforming one-shot and layer-wise pruning in terms of efficiency. Moreover, we found that layer-wise pruning resulted in the least diverse feature representation among different layers. We also observed that Pruned models showed better training performance compared to unpruned models if it has a large number of parameters. However, it is important to note that as more layers were pruned, we observed a significant decrease in the model’s performance, highlighting the need for careful selection of pruning techniques and compression ratios.

## Reproducing the results

### Requirement

- Pytorch

### How to run the code:

- There is folder for each dataset and model combination.
- First download the dataset, then use `Train_base_models.py` to get the base model without any pruning.
- Then run different types of pruning with files   `Iterative_pruning.py` ,`Layerwise_pruning.py` , and `One_Shot_Pruning.py`
- `Prune_then_distill.py` prunes the base model then distills the student using pruned teacher.
