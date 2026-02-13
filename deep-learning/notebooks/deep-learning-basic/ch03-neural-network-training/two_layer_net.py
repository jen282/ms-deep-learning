# coding: utf-8
import sys, os
import numpy as np
sys.path.insert(0, r'C:\Users\USER\ms-deep-learning\deep-learning\notebooks\deep-learning-basic')
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    
    # 가중치 초기화 
    def __init__(self, input_size, hidden_size, output_size, wegiht_init_std=0.01):
        self.params = {}
        self.params['W1'] = wegiht_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = wegiht_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    # 출력값 계산
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # 입력 -> 히든 가중합 계산
        a1 = np.dot(x, W1) + b1
        # 입력 -> 히든 활성화 함수(sigmoid)
        z1 = sigmoid(a1)
        # 히든 -> 출력 가중합 계산
        a2 = np.dot(z1, W2) + b2 
        # 히든 -> 출력 활성화 함수(softmax : 확률로 변환)
        y = softmax(a2)

        return y
    
    # 손실함수 x: 입력 데이터, t: 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    

    # 정확도 측정 함수
    def accuracy(self, x, t):
        y = self.predict(x) 
        y = np.argmax(y, axis=1) # 확률값이 가장 높은 인덱스 
        t = np.argmax(t, axis=1) # 정답 인덱스

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
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y-t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads