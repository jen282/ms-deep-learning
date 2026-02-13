# coding: utf-8
import sys, os
import numpy as np
sys.path.insert(0, r'C:\Users\USER\ms-deep-learning\deep-learning\notebooks\deep-learning-basic')
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    

    def __init__(self, input_size, hidden_size, output_size, wegiht_init_std=0.01):
        # 가중치 초기화 
        self.params = {}
        self.params['W1'] = wegiht_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = wegiht_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성 
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    
    # 출력값 계산
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    # 손실함수 x: 입력 데이터, t: 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    

    # 정확도 측정 함수
    def accuracy(self, x, t):
        y = self.predict(x) 
        y = np.argmax(y, axis=1) # 확률값이 가장 높은 인덱스 
        if t.ndim != 1 : t=np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0]) #예측=정답 일치하는 비율
        return accuracy

    # 가중치 수치 미분
    def numerical_gradient(self, x , t):
        loss_W = lambda W : self.loss(x, t) 
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    # 가중치 학습, 역전파 (back propagation)
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
  
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads