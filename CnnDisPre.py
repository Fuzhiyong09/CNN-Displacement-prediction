from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

##-1-------数据预处理------##

#数据切分
stratPoint = 0
endPoint = 124
#-读取数据
Data = pd.read_excel(r'C:\Users\Administrator\Desktop\Trail.xlsx')
Feature = Data.iloc[:endPoint,:7].values
Label = Data.iloc[:endPoint,8:9].values

#-标准化处理 
StandPFeature = preprocessing.StandardScaler().fit_transform(Feature)
StandLabelPara = preprocessing.StandardScaler().fit(Label)
StandLabel = preprocessing.StandardScaler().fit_transform(Label)

#-构造训练集和测试集
xTrain = StandPFeature[1:112,:] #训练集特征
xTest = StandPFeature[len(xTrain)+1:len(StandPFeature),:] #测试集特征
yTrain = StandLabel[1:112,:]
yTest = StandLabel[len(xTrain)+1:len(StandPFeature),:]

##2----卷积神经网络权重预定义和网络结构定义----##

#2.1-----初始化卷积核权重
n_input = 7
n_output = 1

#网络结构为2层卷积+2层全连接+1层输出层
#卷积核采用2个[1,2,1]
#参数分别为[Height,Width,Channel,预定义维度]
weights = {'wc1':tf.Variable(tf.random_normal([1,2,1,10],stddev = 0.1)),
		   'wc2':tf.Variable(tf.random_normal([1,2,1,16],stddev = 0.1)),
			'wf1':tf.Variable(tf.random_normal([7*1*16, 20],stddev = 0.1)),
           'wf2':tf.Variable(tf.random_normal([20, 10],stddev = 0.1)),
           'wf3':tf.Variable(tf.random_normal([10, n_output],stddev = 0.1)),
          }
		  
#--卷积核偏差，每一个维度都存在一个偏差b
biases = {'bc1': tf.Variable(tf.random_normal([10], stddev = 0.1)),
		  'bc2': tf.Variable(tf.random_normal([16], stddev = 0.1)),
          'bf1': tf.Variable(tf.random_normal([20], stddev = 0.1)),
           'bf2': tf.Variable(tf.random_normal([10], stddev = 0.1)),
		  'bf3': tf.Variable(tf.random_normal([n_output], stddev = 0.1))

}

#2.2-----卷积网络结构搭建

def convBasic(OriginInput, _w, _b, keepratio):
	
	#整合输入数据shape，-1可理解占位符，
	ShapeInput = tf.reshape(OriginInput,[-1,1,7,1])
	#第一层卷积
	conv1 = tf.nn.conv2d(ShapeInput,_w['wc1'],strides = [1,1,1,1], padding = 'SAME')
	conv1Acti = tf.nn.relu(tf.nn.bias_add(conv1,_b['bc1']))
	#第一层池化，池化核参数[batch,Height,Width,Channel]
	pool1 = tf.nn.max_pool(conv1Acti, ksize = [1,1,2,1], strides = [1,1,2,1], padding = 'SAME')
	#第一层失活
	pool1Drop = tf.nn.dropout(pool1, keepratio)
	
	#第二层卷积
	conv2 = tf.nn.conv2d(ShapeInput,_w['wc2'],strides = [1,1,1,1], padding = 'SAME')
	conv2Acti = tf.nn.relu(tf.nn.bias_add(conv2,_b['bc2']))
	#第二层池化
	pool2 = tf.nn.max_pool(conv2Acti, ksize = [1,1,2,1], strides = [1,1,1,1], padding = 'SAME')
	#第二层失活
	pool2Drop = tf.nn.dropout(pool2, keepratio)
	
	#第三层全连接
	dense1 = tf.reshape(pool2Drop, [-1, _w['wf1'].get_shape().as_list()[0]])
	fullC1 = tf.nn.relu(tf.add(tf.matmul(dense1, _w['wf1']), _b['bf1']))
	fullC1Drop = tf.nn.dropout(fullC1, keepratio)
	
	#第四层全连接
	fullC2 = tf.nn.relu(tf.add(tf.matmul(fullC1Drop, _w['wf2']), _b['bf2']))
	fullC2Drop = tf.nn.dropout(fullC2, keepratio)
    
    #第五层输出
	out = tf.add(tf.matmul(fullC2Drop, _w['wf3']), _b['bf3'])
	
	#输出倒数第二层网络结构和最后回归结果
	allOut = { 'out': out,'fullC2':fullC2Drop}
	return allOut

##3--卷积网络输入预定义---##

#3.1-输入预定
x = tf.placeholder(tf.float32, [n_input])
y = tf.placeholder(tf.float32, [n_output])
keepratio = tf.placeholder(tf.float32)

#3.2-损失函数和优化器预定义
pred = convBasic(x, weights, biases, keepratio)['out']
cost = tf.square(y - pred)
optm = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

##4--卷积网络预定义计算

#-4.1初始化tensorflow
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#-4.2预定义参数
trainingEpoch = 800


#-4.3卷积网络计算
for epoch in range(trainingEpoch):
    avgCost = 0 #初始化损失
    
    for i in range(len(xTrain)):
        batchX = xTrain[i,:]
        batchY = yTrain[i]
		
		#传入数据进行计算，计算误差
        feed = {x:batchX, y: batchY, keepratio:0.7}
        sess.run(optm, feed_dict = feed)
        avgCost += sess.run(cost, feed_dict = feed) 
  
	#展示数据	
    if epoch % 100 == 0:
        print('avgerage cost is %f， epoch is %d' % (avgCost, epoch))
		
print('finish')	

##-----------5模型测试-----------##

#5.1获取测试集
TotalCost = 0
cnnPreDisT = []
for i in range(len(xTest)):
    batchX = xTest[i,:]
    batchY = yTest[i]
	
    feed = {x:batchX, y: batchY, keepratio:0.7}
    TotalCost = sess.run(cost, feed_dict = feed) + TotalCost
	
	#预测结果是张量，转为普通numpy格式
    cnnPreDisT.append(sess.run(pred, feed_dict = feed)[0][0])

#平均误差	
AverCost = TotalCost / len(xTest)
CnnMseT = mean_squared_error(yTest, cnnPreDis)

# 或许训练好后的训练集预测结果
for i in range(len(xTrain)):
    batchX = xTrain[i,:]
    batchY = yTrain[i]
	
    feed = {x:batchX, y: batchY, keepratio:0.7}
    cnnPreDisT.append(sess.run(pred, feed_dict = feed)[0][0])




##----6 基于卷积网络+机器学习回归模型预测-----##

#6.1获取卷积网络倒数第二层特征
fullC2 = convBasic(x, weights, biases, keepratio)['fullC2']
CovFe = []
for i in range(len(StandPFeature)):
    batchX = StandPFeature[i,:]
    batchY = StandLabel[i]
	
	#获取训练集卷积后特征
    feed = {x:batchX, y: batchY, keepratio:0.7}
    CovFe.append(sess.run(fullC2, feed_dict = feed)[0])

#6.2PCA降维
#pca = PCA(n_components = 0.90, svd_solver = 'full')
#PFeature = pca.fit_transform(CovFe)
PFeature = np.array(CovFe)

#构造降维后训练集和测试集
xPTrain = PFeature[:len(xTrain),:]
xPTest = PFeature[len(xTrain)+1:len(StandLabel),:]

#--7融合传统机器学习算法----##

#--7融合传统机器学习算法----##

#7.1建立两层bp神经网络神经网络，神经元个数10,5
bp = MLPRegressor(hidden_layer_sizes = [10,5], activation = 'relu',\
					solver = 'adam', learning_rate_init = 0.001, max_iter = 100000)
			
#建立随机森林算法，初始决策树棵树500
rf = RandomForestRegressor(n_estimators = 500,oob_score = True)

#拟合
bp.fit(xPTrain,yTrain)
rf.fit(xPTrain,yTrain)

#7.2模型预测
bpPreDis = bp.predict(xPTrain)
rfPreDis = rf.predict(xPTrain)
bpPreDisT = bp.predict(xPTest)
rfPreDisT = rf.predict(xPTest)

#7.3模型评价
#训练集评价结果
bpAccu = explained_variance_score(yTrain, bpPreDis)
rfAccu = explained_variance_score(yTrain, rfPreDis)
bpMse = mean_squared_error(yTrain, bpPreDis)
rfMse = mean_squared_error(yTrain, rfPreDis)

#测试集评价结果
bpAccuT = explained_variance_score(yTest, bpPreDisT)
rfAccuT = explained_variance_score(yTest, rfPreDisT)
cnnAccuT = explained_variance_score(yTest, cnnPreDisT)
bpMseT = mean_squared_error(yTest, bpPreDisT)
rfMseT = mean_squared_error(yTest, rfPreDisT)

print('the train variance bp is %f, and rf is %f' % (bpAccu, rfAccu))
print('the train mse bp is %f, and rf is %f' % (bpMse, rfMse))
print('the test variance bp is %f, rf is %f and CNN is %f' % (bpAccuT, rfAccuT, cnnAccuT))
print('the test mse bp is %f, rf is %f and CNN is %f' % (bpMseT, rfMseT,CnnMseT))

#反标准化预测结果
actualPreTest = bpPreDisT*np.sqrt(StandLabelPara.var_) + StandLabelPara.mean_


##------------8预测结果绘制------##

#8.1训练集预测结果
plt.plot(range(len(yTrain)), yTrain, color = 'b', marker = 'o',alpha = 0.5)
plt.scatter(range(len(yTrain)), bpPreDis,marker = 'o', color = 'red')
plt.scatter(range(len(yTrain)), rfPreDis, marker = 'o', color = 'black')
plt.show()

#8.2测试集预测结果绘制
plt.plot(range(len(yTest)), yTest, color = 'b', marker = 'o',alpha = 0.5)
plt.scatter(range(len(yTest)), bpPreDisT,marker = 'o', color = 'red')
plt.scatter(range(len(yTest)), rfPreDisT, marker = 'o', color = 'black')
plt.scatter(range(len(yTest)), cnnPreDis, marker = 'o', color = 'green')
plt.show()

##---9传统机器学习算法预测------##

#9.1模型构建
bpA = bp.fit(xTrain,yTrain)
rfA = rf.fit(xTrain,yTrain)

#9.2模型预测
bpAPreDis = bp.predict(xTrain)
rfAPreDis = rf.predict(xTrain)
bpAPreDisT = bp.predict(xTest)
rfAPreDisT = rf.predict(xTest)

#9.3模型评价
#训练集评价结果
bpAAccu = explained_variance_score(yTrain, bpAPreDis)
rfAAccu = explained_variance_score(yTrain, rfAPreDis)
bpAMse = mean_squared_error(yTrain, bpAPreDis)
rfAMse = mean_squared_error(yTrain, rfAPreDis)

#测试集评价结果
bpAccuT = explained_variance_score(yTest, bpAPreDisT)
rfAccuT = explained_variance_score(yTest, rfAPreDisT)

bpAMseT = mean_squared_error(yTest, bpAPreDisT)
rfAMseT = mean_squared_error(yTest, rfAPreDisT)

print('the train variance bp is %f, and rf is %f' % (bpAccu, rfAccu))
print('the train mse bp is %f, and rf is %f' % (bpAMse, rfAMse))
print('the test variance bp is %f, rf is %f' % (bpAccuT, rfAccuT))
print('the test mse bp is %f, rf is %f ' % (bpAMseT, rfAMseT))

##----10参数反标准化----##

# 不同函数预测值反标准化
[bpARealDis, rfARealDis, bpRealDis, rfRealDis,cnnRealDis] =  np.sqrt(StandLabelPara.var_)*[bpAPreDisT, rfAPreDisT, bpPreDisT, rfPreDisT,cnnPreDisT] +StandLabelPara.mean_
[bpARealDisTr, rfARealDisTr, bpRealDisTr, rfRealDisTr] =  np.sqrt(StandLabelPara.var_)*[bpAPreDis, rfAPreDis, bpPreDis, rfPreDis] +StandLabelPara.mean_

# 原函数值标准化
yRealDis =Label[len(xTrain)+1:len(StandPFeature),:]


# 预测结果绘制
plt.plot(range(len(yTest)), yRealDis, color = 'b', marker = 'o',alpha = 0.5)
plt.plot(range(len(yTest)), bpRealDis,marker = 'o', color = 'red')
plt.plot(range(len(yTest)), rfRealDis, marker = 'o', color = 'black')
plt.plot(range(len(yTest)), cnnRealDis, marker = 'o', color = 'green')
plt.plot(range(len(yTest)), bpARealDis, marker = 'o', color = 'purple')
plt.plot(range(len(yTest)), rfARealDis, marker = 'o', color = 'grey')
plt.show()

#测试集预测结果指标
CnnBpRMSET = np.sqrt(mean_squared_error(yRealDis, bpRealDis))
CnnRfRMSET = np.sqrt(mean_squared_error(yRealDis, rfRealDis))
CnnABpRMSET = np.sqrt(mean_squared_error(yRealDis, bpARealDis))
CnnARfRMSET = np.sqrt(mean_squared_error(yRealDis, rfARealDis))
 
#构建整体数据集
bpPreDisToal = np.hstack((bpRealDisTr, bpRealDis))
rfPreDisToal = np.hstack((rfRealDisTr, rfRealDis))
bpAPreDisToal = np.hstack((bpARealDisTr, bpARealDis))
rfAPreDisToal = np.hstack((rfARealDisTr, rfARealDis))

#整体数据集预测结果指标
CnnBpRMSE = np.sqrt(mean_squared_error(Label, bpPreDisToal))
CnnRfRMSE = np.sqrt(mean_squared_error(Label, rfPreDisToal))
CnnABpRMSE = np.sqrt(mean_squared_error(Label, bpAPreDisToal))
CnnARfRMSE = np.sqrt(mean_squared_error(Label, rfAPreDisToal))
