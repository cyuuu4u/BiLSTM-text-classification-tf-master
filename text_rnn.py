#coding:utf-8
import tensorflow as tf
import numpy as np

class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度

    rnn = 'gru'             # lstm 或 gru

    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

class TextRNN(object):
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, l2_reg_lambda, num_layers= 2, hidden_dim = 64, rnn="lstm", trainable=True):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") # 为输入的数据创建占位符，在任何阶段都可使用它向模型输入数据。第二个参数是输入张量的形状，none表示该维度可以为任意值。而在我们模型中该维度表示批处理大小默认为64。
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y") # 标记好的输出的分类
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") # 保存数据用的，第一个参数是数据类型第二个参数是数据结构

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        def lstm_cell():  # lstm核
            return tf.contrib.rnn.BasicLSTMCell(hidden_dim,activation=tf.nn.softsign, forget_bias=1.0, state_is_tuple=True)
            # hidden_dim一个lstm阵列的隐藏单元个数
        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(hidden_dim)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if (rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

        # Embedding layer
        # 这一层的作用是将词汇索引映射到低维度的词向量进行表示。它本质是一个我们从数据中学习得到的词汇向量表。
        with tf.device('/cpu:0'), tf.name_scope("embedding"):  # name_scope创建一个新的名称范围，用于TenosorBoard
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),  # 随机初始化参数w，也就是词向量的矩阵，最大句子长度*词向量维度
                trainable=True, name="W")
            # 把随机初始化的句子中的数字看做索引，寻找对应的词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # 选取词向量矩阵里索引对应的元素，input_x为索引
            print("embedd:", self.embedded_chars)

        ################## 多层单向lstm网络 ##################

        # with tf.name_scope("multiRNN"):
        #     cells = [dropout() for _ in range(num_layers)]
        #     rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        #     # self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        #     _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embedded_chars, dtype=tf.float32)# 输出outputs,last_states，其中outputs是[batich,max_length,embedding_size]，也就是每一个迭代隐状态的输出,
        #     print("_outputs", _outputs)
        #     _outputs_expand = tf.expand_dims(_outputs, -1)
        #     print("_outputs_expand:", _outputs_expand)
        #     # last_states是由(c_state,h_state)组成的tuple，均为[batch,embedding_size]。
        #     # tf.nn.dynamic_rnn(cell,inputs)
        #     # tf.nn.static_bidirectional_rnn()的outputs的shape就是[steps,batch_size,dim*2,m](注意，是list类型)最后的输出是前向和后向输出的concat
        #     # last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
        #     # print("rnn_output", last)
        #
        # # # rnn输出再接一个全连接层用relu激活送入再一个全连接层分类
        # # with tf.name_scope("score"):
        # #     # 全连接层，后面接dropout以及relu激活
        # #     fc = tf.layers.dense(last, hidden_dim, name='fc1')
        # #     fc = tf.contrib.layers.dropout(fc, self.dropout_keep_prob)
        # #     fc = tf.nn.relu(fc)
        # #
        # #     # 分类器
        # #     self.logits = tf.layers.dense(fc, num_classes, name='fc2')
        # #     self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
        #
        # # mean pooling是对所有时间单位上的ht的平均
        # with tf.name_scope("mean_pooling_layer"):
        #     pool = tf.nn.avg_pool(_outputs_expand,
        #                           [1, sequence_length, 1, 1],
        #                           [1, 1, 1, 1],
        #                           'VALID',
        #                           name='pool')
        #     pool_squeeze = tf.squeeze(pool)
        #     # out_put = tf.reduce_mean(_outputs, 1)
        #     print("pool", pool_squeeze)
        #
        # # mean pooling之后直接用全连接层分类
        # with tf.name_scope('output'):
        #     # 双向
        #     # W = tf.Variable(tf.truncated_normal([2*hidden_unit, num_classes], stddev=0.1), name='W')
        #     # 单向
        #     W = tf.Variable(tf.truncated_normal([int(hidden_dim), 2], stddev=0.1), name='W')
        #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
        #     l2_loss += tf.nn.l2_loss(W)
        #     l2_loss += tf.nn.l2_loss(b)
        #     self.logits = tf.nn.xw_plus_b(pool_squeeze, W, b, name='scores')
        #     self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')


        #################### 双向lstm #####################

        # with tf.name_scope("biRNN"):
        #     x = tf.unstack(self.embedded_chars, sequence_length, 1)
        #     # Define lstm cells with tensorflow
        #     # Forward direction cell
        #     lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, forget_bias=1.0)
        #     # Backward direction cell
        #     lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, forget_bias=1.0)
        #
        #     (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_chars, dtype=tf.float32)
        #     _outputs = tf.concat([output_fw, output_bw], axis=-1)
        #
        #     # # Get lstm cell output
        #     # _outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
        #     #                                                dtype=tf.float32)
        #     # shape[steps,batch_size,dim*2,m]
        #     print("outputs:", _outputs)
        #     _outputs_reshape = tf.reshape(_outputs, [-1, sequence_length, 2 * hidden_dim])
        #     _outputs_expand = tf.expand_dims(_outputs_reshape, -1)
        #     print("_output_expand:", _outputs_expand)

        ################### 双向多层lstm ###################

        with tf.name_scope("multi-biRNN"):
            # x = tf.unstack(self.embedded_chars, sequence_length, 1)
            _inputs = self.embedded_chars
            for _ in range(num_layers):
                with tf.variable_scope(None, default_name="bidirectional-rnn"):
                    rnn_cell_fw = lstm_cell()
                    rnn_cell_bw = lstm_cell()
                    # initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
                    # initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
                    (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs,
                                                                      dtype=tf.float32)
                    print("output:", output)
                    _inputs = tf.concat(output, 2)
            _outputs = _inputs

            # shape[steps,batch_size,dim*2,m]
            print("outputs:", _outputs)
            # _outputs_reshape = tf.reshape(_outputs, [-1, sequence_length, 2 * hidden_dim])
            # _outputs_expand = tf.expand_dims(_outputs_reshape, -1)
            # print("_output_expand:", _outputs_expand)

        with tf.name_scope("mean_pooling_layer"):
            # pool = tf.nn.avg_pool(_outputs_expand,
            #                       [1, sequence_length, 1, 1],
            #                       [1, 1, 1, 1],
            #                       'VALID',
            #                       name='pool')
            # pool_squeeze = tf.squeeze(pool)
            self.out_put = tf.reduce_mean(_outputs, 1) # simply求平均？
            avg_pool = tf.nn.dropout(self.out_put, keep_prob=self.dropout_keep_prob)
            print("pool", avg_pool)

        with tf.name_scope('output'):
            # 双向
            W = tf.Variable(tf.truncated_normal([int(2*hidden_dim), 2], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[2]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(avg_pool, W, b, name='scores')
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)+l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
