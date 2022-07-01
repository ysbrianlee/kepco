import tensorflow as tf
from residual import make_residual_block_layer, make_bottleneck_layer

class ResNetWithResidualBlock(tf.keras.Model) :
    def __init__(self, NUM_CLASSES, layer_params) :
        super(ResNetWithResidualBlock, self).__init__()
        
        # 첫 번째 Convolution layer
        # Kernel size : 7x7
        # Output channel : 64
        # Stride : 2
        # Pooling window : 3x3
        # Pooling stride : 2
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2,
                                              padding='same')
        
        # Residual block 을 사용한 Conv layer block
        # layer_params 를 통해 반복횟수를 모델별로 변경하여 ResNet-18 / ResNet-34 구현

        self.layer1 = make_residual_block_layer(filter_num=64,
                                               blocks=layer_params[0])
        
        self.layer2 = make_residual_block_layer(filter_num=128,
                                               blocks=layer_params[1],
                                               stride=2)
        
        self.layer3 = make_residual_block_layer(filter_num=256,
                                               blocks=layer_params[2],
                                                stride=2)
        
        self.layer4 = make_residual_block_layer(filter_num=512,
                                               blocks=layer_params[3],
                                               stride=2)

        # Global Average Pooling        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

        # Fully connected layer with softmax
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                       activation=tf.keras.activations.softmax)
        
    # Constructor 에서 각각의 layer 를 정의하고 call() 메소드에서 layer 를 연결하여 네트워크를 구성한다.       
    def call(self, inputs, training=None, mask=None) :
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.avgpool(x)
        output = self.fc(x)
        
        return output
        
class ResNetWithBottleneck(tf.keras.Model) :
    def __init__(self, NUM_CLASSES, layer_params) :
        super(ResNetWithBottleneck, self).__init__()

        # 첫 번째 Convolution layer
        # Kernel size : 7x7
        # Output channel : 64
        # Stride : 2
        # Pooling window : 3x3
        # Pooling stride : 2
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2,
                                              padding='same')
        
        # Bottleneck block 을 사용한 Conv layer block
        # layer_params 를 통해 반복횟수를 모델별로 변경하여 ResNet-50 / ResNet-101 / ResNet-152 구현
        self.layer1 = make_bottleneck_layer(filter_num=64,
                                           blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                           blocks=layer_params[1],
                                           stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                           blocks=layer_params[2],
                                           stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                           blocks=layer_params[3],
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
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)
        
        return output
    
def resnet_18(NUM_CLASSES):
    return ResNetWithResidualBlock(NUM_CLASSES, layer_params=[2, 2, 2, 2])

def resnet_34(NUM_CLASSES):
    return ResNetWithResidualBlock(NUM_CLASSES, layer_params=[3, 4, 6, 3])

def resnet_50(NUM_CLASSES):
    return ResNetWithBottleneck(NUM_CLASSES, layer_params=[3, 4, 6, 3])

def resnet_101(NUM_CLASSES):
    return ResNetWithBottleneck(NUM_CLASSES, layer_params=[3, 4, 23, 3])

def resnet_152(NUM_CLASSES):
    return ResNetWithBottleneck(NUM_CLASSES, layer_params=[3, 8, 36, 3])