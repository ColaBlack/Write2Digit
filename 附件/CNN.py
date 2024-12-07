import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import models, layers
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

x, y = load_digits(return_X_y=True)  # 载入数据
x = x.reshape(-1, 8, 8, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=202406)  # 划分训练集与测试集
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 对标签进行独热编码
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 定义网络结构
CNN = models.Sequential()

# 定义卷积层
# 该有32个3x3的卷积核，步长为1
CNN.add(layers.Convolution2D(input_shape=(8, 8, 1), filters=32, kernel_size=3, strides=1, padding='same',
                             activation='relu'))
# 定义池化层
CNN.add(layers.MaxPool2D(pool_size=2, strides=2, padding='same'))
CNN.add(layers.Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
CNN.add(layers.MaxPool2D(pool_size=2, strides=2, padding='same'))
# 扁平化
CNN.add(layers.Flatten())
# 全连接层
CNN.add(layers.Dense(units=512, activation='relu'))
CNN.add(layers.Dropout(0.5))
CNN.add(layers.Dense(units=10, activation='softmax'))

# 绘制网络结构示意图
plot_model(CNN, to_file='CNN.png', show_shapes=True, show_layer_names=False, rankdir='TB')
plt.figure(figsize=(10, 10))
img = plt.imread('CNN.png')
plt.imshow(img)
plt.axis('off')
plt.show()

# 设置训练参数并进行训练
# 设置学习率为0.001
CNN.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 进行20次训练，每批64个数据
CNN.fit(x_train, y_train, epochs=20, batch_size=64)

# 进行模型评估
train_loss, train_acc = CNN.evaluate(x_train, y_train, batch_size=64)
print(f"训练集损失函数{train_loss:.4f},准确率{train_acc * 100:.2f}%")
test_loss, test_acc = CNN.evaluate(x_test, y_test, batch_size=64)
print(f"测试集损失函数：{test_loss:.4f},准确率：{test_acc * 100:.2f}%")
y_pred = CNN.predict(x_test).argmax(axis=-1)
y_true = y_test.argmax(axis=-1)
print("-----分类报告如下-----")
print(classification_report(y_true, y_pred))
print("混淆矩阵如下")
cm = confusion_matrix(y_true, y_pred)
print(cm)
print("-----网络报告如下-----")
print(CNN.summary())

# 绘制混淆矩阵的热力图
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 随机展示一个样例
index = np.random.randint(0, len(x_test))
sample = x_test[index]
sample = sample.reshape(8, 8)
plt.imshow(sample, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f"True Label:{y_true[index]},Predicted Label:{y_pred[index]}")
plt.show()
print(np.argmax(y_pred[index]))

# 测试完毕，保存网络模型
CNN.save('CNN.h5')
