# ETRI - 2023 휴먼이해 인공지능 경진대회
## 멀티 센서 행동 학습을 위한 그래프 기반 라이프로그 세그멘테이션 프레임워크 
<!-- A Graph-based Lifelog Segmentation Framework for Multi-Sensor Activity Recognition -->
[2023 휴먼이해 인공지능 논문경진대회](https://aifactory.space/competition/detail/2234)
## 1. Abstract
본 논문에서는 라이프로그 데이터에 기반의 행동 분류 모델에 대한 프레임워크를 제안한다. 기존의 논문들과는 다르게, 본 논문은 그래프 신경망 (GNN) 기반의 모델을 활용하여 멀티 센서 간의 종속성을 고려하고 세그멘테이션을 통해 행동 예측을 수행한다. 실험을 통해 제안하는 모델은 라이프로그 데이터에 대해 효과적인 행동 예측을 수행할 수 있음을 확인하였다.


## 2. Model Architecture
<img width="80%" src=model.PNG>

## 3. Code

## 3.1 Environment
> - Python 3.X
> - Pytorch 1.13
> - Numpy
> - Pandas
> - pyyaml
> - torch-geometric 2.2.0
> - torch-scatter 2.1.1
> - torch-sparse 0.6.17
> - tensorboard (for loss visualization: Optional)

## 3.2 파일 설명
### 3.2.1 전처리 ("./preprocess/")
> ```data_preprocess.py``` 데이터 전처리
>   - GNN 모델 학습을 위한 timewise data (--type timewise)
>   - Segmentation 모델 학습을 위한 daywise data (--type daywise)
> 
> ```augmentation.py``` 데이터 augmentation 
> 
> ```augmentation_guide.ipynb``` augmentation guide 노트북 파일. 상황에 맞춰 augmentation 진행하면 됨. 

### 3.2.2 Body-Action GNN Model  ("./code/")
> ```config.yaml``` configuraion 파일 (GNN initial edge 및 모델 파라미터 등)
> 
> ```gnn.py``` GNN 모델
> 
> ```loader.py``` GNN data loader (timewise data)
> 
> ```pretrain.py``` train 함수
> 
> ```main.py``` main 함수
> 

### 3.2.3 Lifelog Segmentation Model ("./code/")
> ```segmentation_loader.py``` Segmentation loader (daywise data)
> 
> ```segmentation.py``` semgentation train 함수
> 
> ```tmse.py``` Temporal MSE 손실함수
> 
> ```main_seg.py``` main 함수

## 3.3 사용 방법
### 3.3.1 데이터 전처리
Data는 [ETRI 라이프로그 데이터셋 (2020-2018)](https://nanum.etri.re.kr/share/schung1/ETRILifelogDataset2020?lang=ko_KR)를 활용한다. './data/original/'에 원본 데이터를 다운 받은후 preprocessing.py를 통해 전처리 이후 학습을 위한 디렉토리는 아래와 같다 ("./data/daywise/", "./data/timewise/".)
```
+--data
  +--original
    +--user01
    +--user02
      ...
    +--user30
    
  +--daywise
    +--train
      +--user01_1598759880.npz
      +--user01_1598832660.npz
        …
      +--user30_1600988460.npz
      
    +--test
      +--user01_1600013400.npz
      +--user01_1600527600.npz
        …
      +--user30_1600902660.npz
      
  +--timewise
      +--user01_1598759880_1598777040.npz
      +--user01_1598759880_1598779980.npz
        …
      +--user30_1600988460_1601033040.npz 
```
Timewise 데이터는 클래스별 8000개씩으로 구성하고 8000개 미만의 class에 대한 데이터는 augmentation.py를 통해 8000개로 맞춘다.
Daywise train/test 분리 별도. 
```
python preprocessing.py --type daywise --file_dir ./data/original --save_dir ./data/daywise # daywise 파일 저장 -> GNN 모델 학습
python preprocessing.py --type timewise --file_dir ./data/original --save_dir ./data/timewise # timewise 파일 저장 -> Segmentation 모델 학습
```

### 3.3.2 Body-Action GNN 모델 학습
```
python main.py --epoch 50 --root_dir ./data/timewise --mode conbarlow --barlow_epoch 30 --save_dir <save_directory> --exp_name <save_exp_name>
```

### 3.3.3 Lifelog Segmentation 모델 학습
```
python main_seg.py --epoch 50 --root_dir ./data/daywise/train --mode segmentation --save_dir <save_directory> --exp_name <save_exp_name>
```

## 4. Experiments
|방법론|Accuracy|Weighted F1|
|----|----|----|
|SVM|0.29|0.24|
|Logistic Regression|0.30|0.24|
|MLP|0.24|0.19|
|K-Means|0.13|0.04|
|FINCH|0.20|0.14|
|OURS|0.42|0.29|



## Contact
- Eunseon Seong : emilyseong@hanyang.ac.kr
- Harim Lee : hrimlee@hanyang.ac.kr

## Reference
[1] Seungeun Chung, Chi Yoon Jeong, Jeong Mook Lim, Jiyoun Lim, Kyoung Ju Noh, Gague Kim, Hyuntae Jeong,
Real-world multimodal lifelog dataset for human behavior study. ETRI Journal 43(6), 2021 

[2] Mutegeki, Ronald, and Dong Seog Han, A CNN-LSTM approach to human activity recognition. 2020 international conference on artificial intelligence in information and communication (ICAIIC). IEEE, 2020.

[3] Khatun, Mst Alema, et al. Deep CNN-LSTM with self-attention model for human activity recognition using wearable sensor. IEEE Journal of Translational Engineering in Health and Medicine 10 : 1-16, 2022
[4] 
Eldele, Emadeldeen, et al. Time-series representation learning via temporal and contextual contrasting. arXiv preprint arXiv:2106.14112, 2021.

[5] Lishan Qiao, Limei Zhang, Songcan Chen, and Dinggang Shen. 2018. Data-driven graph construction and graph learning: A review. Neurocomputing 312 336-351, 2018.	

[6] Zbontar, Jure, et al. Barlow twins: Self-supervised learning via redundancy reduction. International Conference on Machine Learning. PMLR, 2021.	

[7] Ishikawa, Yuchi, et al. Alleviating over-segmentation errors by detecting action boundaries. Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021.

[8] M. Saquib Sarfraz, Vivek Sharma, and Rainer Stiefelhagen. Efficient parameter-free clustering using first neighbor relations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 8934–8943, 2019.
