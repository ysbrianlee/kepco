import tensorflow as tf

# Residual block 구현 Class
# tf.keras.layers.Layer 를 상속받아 생성자와 call() 메소드를 overriding 한다. 
class ResidualBlock(tf.keras.layers.Layer) :
    # Constructor 구현
    def __init__(self,
                 filter_num,    # Residual block 의 feature map channel 개수
                 stride=1       # convolution layer 의 stride(default=1)
                ):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,    # 출력 feature map channel 개수
                                           kernel_size=(3, 3),     # convolution kernel size
                                           strides=stride,         # convolution stride
                                           padding='same')         # zero-padding 추가
        self.bn1 = tf.keras.layers.BatchNormalization()            # Batch normalization
        
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,    # 출력 feature map channel 개수
                                           kernel_size=(3, 3),     # convolution kernel size
                                           strides=1,              # convolution stride
                                           padding='same')         # zero-padding 추가
        self.bn2 = tf.keras.layers.BatchNormalization()            # Batch normalization
        
        # Convolution stride 값에 따른 조건문 처리
        # Stride = 1 이 아닌 경우 conv1 을 수행하면 Feature map의 크기가 변경된다.
        # [B, H, W, Cin] -> [B, H/stride, W/stride, Cout]
        # 따라서, shortcut 으로 연결된 입력 또한 동일 크기로 변경해주어야하며, 이 때 1x1 convolution 사용
        if stride != 1 :
            self.shortcut = tf.keras.Sequential()    # tf.keras.Sequential() 모델 생성
            self.shortcut.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                      kernel_size=(1, 1),
                                                      strides=stride))
            self.shortcut.add(tf.keras.layers.BatchNormalization())
        # stride = 1 일 경우 Identity shortcut 을 그대로 사용한다.
        else :
            self.shortcut = lambda x : x
            
    # Constructor 에서 각각의 layer 를 정의하고 call() 메소드에서 layer 를 연결하여 네트워크를 구성한다. 
    def call(self,
             inputs,            # 입력 Tensor
             training=None,     # Training 여부
             **kwargs):         # 기타 인자 (이번 실습에선 사용하지 않음)
        residual = self.shortcut(inputs)    # Shortcut 생성
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        
        return output
    
# Residual block 을 연결한 Conv layer 구현 함수
def make_residual_block_layer(filter_num, blocks, stride=1):
    # blocks 개수만큼 Residual block을 생성
    res_block = tf.keras.Sequential()
    res_block.add(ResidualBlock(filter_num, stride=stride))
    
    for _ in range(1, blocks) :
        res_block.add(ResidualBlock(filter_num, stride=1))
        
    return res_block


# Bottleneck block 구현 Class
# tf.keras.layers.Layer 를 상속받아 생성자와 call() 메소드를 overriding 한다. 
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self,
                 filter_num,    # Bottleneck 의 첫 번째 convolution layer 의 출력 feature map 채널 개수
                 stride=1       # Convolution layer 의 stride
                ):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                           kernel_size=(1, 1),        # 1x1 convolution 으로 feature map 채널 개수 조절
                                           strides=1,
                                           padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                           kernel_size=(3, 3),
                                           strides=stride,
                                           padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,    # 첫 번째 layer 의 출력 feature map 채널 개수의 4배
                                           kernel_size=(1, 1),         # 1x1 convolution 으로 feature map 간 채널 개수 조절
                                           strides=1,
                                           padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        # 1x1 convolution 을 이용하여 shortcut 으로 연결된 입력 또한 동일 출력 feature map 채널 개수를 갖게 함
        self.shortcut = tf.keras.Sequential()
        
        self.shortcut.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                  kernel_size=(1, 1),
                                                  strides=stride))
        self.shortcut.add(tf.keras.layers.BatchNormalization())
        
    # Constructor 에서 각각의 layer 를 정의하고 call() 메소드에서 layer 를 연결하여 네트워크를 구성한다.
    def call(self, inputs, training=None, **kwargs):
        residual = self.shortcut(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        output = tf.nn.relu(tf.keras.layers.add([residual,  x]))
        
        return output
    
# Bottleneck 을 연결한 Conv layer 구현 함수
def make_bottleneck_layer(filter_num, blocks, stride=1):
    # blocks 개수만큼 Bottleneck 을 생성
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))
    
    for _ in range(1, blocks) :
        res_block.add(BottleNeck(filter_num, stride=1))
        
    return res_block