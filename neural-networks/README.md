## 신경망 

### 신경망 추론

$\mathbf{h} = \mathbf{xW} + \mathbf{b}$

$\mathbf{h,x,W,b}$ 는 각각 입력, 은닉층 뉴런, 가중치, 편향을 뜻하는 행렬

`forward_net.py` 에 구현된 `TwoLayerNet` 클래스를 통해 신경망 추론:

```{.python}
x = np.random.randn(10,2)
model = TwoLayerNet(2,4,3)
s = model.predict(x)
```



### 신경망 학습

