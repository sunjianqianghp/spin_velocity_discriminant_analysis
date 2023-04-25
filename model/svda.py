import tensorflow as tf

def Q(rotations):
    '''
    计算各运动点对应的组内误差平方和占总误差平方和的比重Q_i, 再将它们相加并返回
    rotations: List[np.array]
    '''
    nums_samples = [rotation.shape[0] for rotation in rotations]                       # 各类的样本量
    bars_within_class = [tf.reduce_mean(rotation, axis=0, keepdims=True) for rotation in rotations]   # 组内样本平均值: [[[bar_p1_class1, bar_p2_class1,...]], [[bar_p1_class2, bar_p2_class2,...]], ...]
    bar_total = tf.reduce_sum(tf.concat(rotations, axis=0), axis=0, keepdims=True)/sum(nums_samples)  # 总平均值: [[bar_p1, bar_p2, ...]]
    # [num_moving_points]
    sse = tf.reduce_sum(
        tf.concat(
            [(rotation-bar_cls)**2 for rotation, bar_cls in zip(rotations, bars_within_class)],  #组内样本与组内样本平均值差的平方
            axis=0), # 将不同类的“误差平方”concat起来
        axis=0)      # 求和

    # [num_moving_points]
    ssa = tf.reduce_sum(
        tf.concat(
            [n*(bar_cls-bar_total)**2 for n, bar_cls in zip(nums_samples, bars_within_class)], 
            axis=0), 
        axis=0)
    
    sse_sum = tf.reduce_sum(sse)
    ssa_sum = tf.reduce_sum(ssa)
    print(sse_sum.numpy(), ssa_sum.numpy(), sse_sum.numpy()/ssa_sum.numpy(), sse_sum.numpy()-ssa_sum.numpy())
    return sse_sum/ssa_sum

def Cov(X):
    '''
    计算X中协方差矩阵，X的每一列为一个变量
    X: shape=[None]
    '''
    n = X.shape[0]   # 样本量
    means   = tf.reduce_mean(X, axis=0, keepdims=True) # 各特征均值
    X_means = X-means
    cov = tf.matmul(tf.transpose(X_means), X_means)/n
    return cov

def corr(X):
    '''
    计算X的相关系数矩阵，X的每一列为一个变量
    X: shape=[None]
    '''
    cov = Cov(X) # 协方差
    std_vec = tf.math.reduce_std(X, axis=0, keepdims=True) # 各特征标准差
    correlation_pearson = cov/tf.matmul(tf.transpose(std_vec), std_vec)
    return correlation_pearson

def distM(x, G):
    '''
    desc: 返回点x到点集的马氏距离
    Args:
        x: 样本点集, shape=[dim] or [None, dim]
        G: 一组样本集, shape=[None, dim]
    '''
    if len(x.shape)==1: # 如果阶数只有1，对其扩充
        x = tf.expand_dims(x, 0)

    if int(x.shape[-1]) != int(G.shape[-1]):
        raise Exception("x and G have different column num")

    mu = tf.reduce_mean(G, axis=0, keepdims=True) # 集群中心点
    Sigma = Cov(G) # 集群协方差
    dd = x - mu
    return tf.expand_dims(tf.linalg.diag_part(tf.sqrt(tf.matmul(tf.matmul(dd, tf.linalg.inv(Sigma)), tf.transpose(dd)))), -1)

def classifier(x, Gs):  
    '''
    根据马氏距离返回x所属的类
    '''
    distances = tf.concat([distM(x, G) for G in Gs], axis=-1)
    indice = tf.argmin(distances,    axis = 1, output_type=tf.int32)
    return indice

def obj(rotations, alpha1=0.01):
    Q_sum = Q(rotations)
    Corr_2_sum = tf.reduce_sum(tf.concat([corr(rotation)**2 for rotation in rotations], axis=0))
#     print(Q_sum.numpy(), Corr_2_sum.numpy())
    return Q_sum+alpha1*Corr_2_sum

def accuracy(xs, ys):
    xs=tf.squeeze(xs)
    ys=tf.squeeze(ys)
    return 1.0-tf.reduce_mean(
        tf.cast(
            tf.sign(
                tf.abs(xs - ys)
            ), 
            tf.float32)
    )


class RotationModel(tf.keras.Model):
    def __init__(self, sample_dim, num_moving_points=1, uniform_len=1.0):
        super(RotationModel, self).__init__()
        self.num_moving_points = num_moving_points
        self.moving_points_positions = []
        self.moving_points_velocities = []
        
        for i in range(self.num_moving_points):
            self.moving_points_positions.append(
                tf.Variable( 
                    tf.random.uniform(shape=[1, sample_dim], minval=0.0,  maxval=uniform_len, dtype=tf.float32), 
                    dtype=tf.float32, 
                    name='postion'+str(i)
                )
            )
            self.moving_points_velocities.append(
                tf.Variable(
                    tf.random.uniform(shape=[1, sample_dim], minval=0.0, maxval=uniform_len, dtype=tf.float32), 
                    dtype=tf.float32, 
                    name='velocity'+str(i)
                )
            )
        
    def rotate_velocity(self, v, x, xt):              ##定义角速度函数
        '''
        xt: 参考点位置, shape=[-1, dim], dim为样本维度
        x: 移动点位置, shape=[1, dim]
        v：移动点速度, shape=[1, dim]
        返回角速度
        '''
        s = xt - x # [None, dim]
        n = int(s.get_shape()[0]) # 样本量
        numerator = tf.sqrt( tf.maximum(tf.reshape( ( tf.norm(v)*tf.norm(s, axis = 1) )**2, [n,1]) - tf.matmul(s, tf.transpose(v))**2, 0.0001) ) # [None, 1]
        denominator = tf.reshape(   tf.norm(s, axis = 1)**2+0.001, [n, 1])  # [None, 1]
        r = numerator/denominator # [None, 1]
        return  r
    
    def call(self, inputs):
        '''
        inputs: List[numpy.ndarray] or numpy.ndarray
        返回转速张量（列表）
        '''
        if type(inputs) == list: # 训练时
            num_classes = len(inputs)
            rotations = []
            for i in range(num_classes):
                rotations.append(
                    tf.concat(
                        [self.rotate_velocity(self.moving_points_velocities[j], self.moving_points_positions[j], inputs[i]) for j in range(self.num_moving_points)], 
                        axis=1)
                )
            return rotations
        else: # 预测时
            return tf.concat([self.rotate_velocity(self.moving_points_velocities[j], self.moving_points_positions[j], inputs) for j in range(self.num_moving_points)], axis=1)
    
    def predict_cate(self, x, Gs):  
        '''
        根据马氏距离返回x所属的类
        '''
        distances = tf.concat([self.distM(x, G) for G in Gs], axis=-1)
        indice = tf.argmin(distances,    axis = 1)
        return indice
    
