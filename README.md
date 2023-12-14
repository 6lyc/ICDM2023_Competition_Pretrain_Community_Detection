# ICDM 2023 图学习挑战赛：基于预训练模型的社区发现和团伙挖掘 
# —— Top 6

## ComDetector: 节点-子结构判别能力是社区发现预训练任务的关键

### About
本文档关注如何将预训练技术应用于社区发现任务，并详细阐述了一种切实可行的方案***ComDetector***。在本方案中，我们发现让模型学会判别图中的节点与其对应子结构（社区）是解决社区发现预训练任务的关键。具体的，我们在预训练阶段采用互信息最大化的方式让模型学会将开源数据集ogbn-arxiv中的节点与其对应子结构绑定的知识，并通过微调将这个能力泛化到其他数据集上。

官方推送: [IEEE ICDM 2023 图学习挑战赛落幕！探索深度图学习应用潜力](http://mp.weixin.qq.com/s?__biz=MzkyNDI4Njc5NA==&mid=2247489323&idx=1&sn=f0c9196dcd248648c216fe3df98fda15&chksm=c1d97d0ef6aef418838d1bf07f5824fbcc6f22631ab5122d6030e12e3a869ca9d29eed2069e2&mpshare=1&scene=24&srcid=1213tekVSF2f73bmFu64BJtY&sharer_shareinfo=be4a247a16a5a21ec88df87d7ee8e4b4&sharer_shareinfo_first=be4a247a16a5a21ec88df87d7ee8e4b4#rd)

### Setup
该脚本在`python 3.9.17`上运行，并需要安装以下第三方库以及他们的依赖：

    dgl==0.6.1

    networkx==3.1

    numpy==1.26.0

    ogb==1.3.6

    pandas==2.1.0

    scikit-learn==1.0.2

    scipy==1.11.2

    torch==1.13.1

    torchaudio==0.13.1

    torchvision==0.14.1
    
### Script
· 在 ogbn-arxiv (in dataset folder) 上预训练，并在 icdm2023_session1_test (将icdm数据放在`./icdm2023_session1_test/`路径下)上测试，结果见result文件夹

(1) 预训练数据预处理

`python ./ogbn_preprocess.py`

(2) 测试数据下载

    cd ./icdm2023_session1_test
    wget https://icdm2023dataset.oss-rg-china-mainland.aliyuncs.com/testdata/icdm2023_session1_test.zip
    unzip icdm2023_session1_test.zip
    
(3) 预训练以及测试

`sh ./script/pretrain.sh`

· 在 Cora / Citeseer (in data folder) 上进行本地测试 

`sh ./script/local_ft.sh`
