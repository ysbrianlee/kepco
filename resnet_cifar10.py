import tensorflow as tf
from residual import make_residual_block_layer, make_bottleneck_layer

class ResNetWithResidualBlock(tf.keras.Model) :
    def __init__(self, NUM_CLASSES, layer_params) :
        super(ResNetWithResidualBlock, self).__init__()

        # 첫 번째 Convolution layer
        # Kernel size : 3x3
        # Output channel : 16
        # Stride : 1
        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                           kernel_size=(3, 3),
                                           strides=1,
                                           padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        # Residual block 을 사용한 Conv layer block
        # layer_params 를 통해 반복횟수를 모델별로 변경하여 ResNet 구현
        # 기존 ResNet 과 달리 Conv layer block 은 4개 -> 3개로 변경
        self.layer1 = make_residual_block_layer(filter_num=16,
                                               blocks=layer_params[0])
        
        self.layer2 = make_residual_block_layer(filter_num=32,
                                               blocks=layer_params[1],
                                               stride=2)
        
        self.layer3 = make_residual_block_layer(filter_num=64,
                                               blocks=layer_params[2],
                                                stride=2)

        # Global Average Pooling  
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # Fully connected layer with softmax
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                       activation=tf.keras.activations.softmax)
        
    def call(self, inputs, training=None, mask=None) :
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)
        
        return output
        
class ResNetWithBottleneck(tf.keras.Model) :
    def __init__(self, NUM_CLASSES, layer_params) :
        super(ResNetWithBottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                           kernel_size=(3, 3),
                                           strides=2,
                                           padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.layer1 = make_bottleneck_layer(filter_num=16,
                                           blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=32,
                                           blocks=layer_params[1],
                                           stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=64,
                                           blocks=layer_params[2],
                                           stride=2)
        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                       activation=tf.keras.activations.softmax)
        
    def call(self, inputs, training=None, mask=None) :
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)
        
        return output
    
def resnet_20(NUM_CLASSES):
    return ResNetWithResidualBlock(NUM_CLASSES, layer_params=[6, 6, 6])

def resnet_32(NUM_CLASSES):
    return ResNetWithResidualBlock(NUM_CLASSES, layer_params=[10, 10, 10])

def resnet_44(NUM_CLASSES):
    return ResNetWithBottleneck(NUM_CLASSES, layer_params=[14, 14, 14])

def resnet_56(NUM_CLASSES):
    return ResNetWithBottleneck(NUM_CLASSES, layer_params=[18, 18, 18])

def resnet_110(NUM_CLASSES):
    return ResNetWithBottleneck(NUM_CLASSES, layer_params=[36, 36, 36])