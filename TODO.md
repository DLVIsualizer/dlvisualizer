## TO LEARN
#### 1. 딥러닝 공부 + Tensorflow에 익숙해지자 
-> 무엇을 어떻게 visualization 하는 것이 의미있는지를 알기 위해
<br></br>
#### 2. 모두를 위한 딥러닝
#### 3. udacity의 deep learning 코스
#### 4. kaggle 

## TODO

#### 1. 시각화를 지원할 모델을 선정
Googlenet, alexnet, resnet 등 주로 많이 사용되는 모델들을 선정하도록
Or user defined model에 대한 visualization을 지원할 방법에 대해서도 조사
<br></br>
####  2. tf.saver 모델 파싱 및 data(이미지) load
tf.saver()로 저장한 모델을 어떻게 파싱할 것인지
<br></br>

#### 3. 뉴런을 통과한 이미지 처리
사용자가 선택한 이미지가 모델을 통과할 때 변화하는 이미지를 어디서 계산할지
##### - dlvisualizer에서 계산
    - dlvisualizer
            1. 기존에 tensorflow api를 이용하여 neuron ouput들을 계산
            - 어떻게 저장?
                - 파일로 저장
            2. 저장 없이 특정 API를 호출했을 때 return 해줌
     -  dlvisualizer-web
            1. 어떻게 불러오기?
                -  gpu.js or turbo.js로 이미지파일 로딩?
            2. API호출
<br></br>

##### - dlvisualizer-web에서 계산
    - dlvisualizer
        - 할거 없음
    -  dlvisualizer-web
        - gpu.js or turbo.js를 이용해서 neuron output들을 계산
        - 어떻게 불러오기?
            -  계산한 이미지들 로딩
<br></br>

####  4. Django로 웹서버 개발
