from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import workspace, brew


def inception_block(model, input_blob, prefix, input_channel,
                    pool_channel, conv5_reduce_channel, conv5_channel,
                    conv3_reduce_channel, conv3_channel, conv1_channel):
    # pool
    brew.max_pool(
        model,
        input_blob,
        prefix + '_pool',
        kernel=3,
        stride=1,
        pad=1
    )
    brew.conv(
        model,
        prefix + '_pool',
        prefix + '_pool_proj_conv',
        input_channel,
        pool_channel,
        kernel=1,
        stride=1,
        pad=0,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        prefix + '_pool_proj_conv',
        prefix + '_pool_proj_bn',
        pool_channel
    )
    brew.relu(
        model,
        prefix + '_pool_proj_bn',
        prefix + '_pool_proj_relu'
    )
    # 5x5
    brew.conv(
        model,
        input_blob,
        prefix + '_5x5_reduce_conv',
        input_channel,
        conv5_reduce_channel,
        kernel=1,
        stride=1,
        pad=0,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        prefix + '_5x5_reduce_conv',
        prefix + '_5x5_reduce_bn',
        conv5_reduce_channel
    )
    brew.relu(
        model,
        prefix + '_5x5_reduce_bn',
        prefix + '_5x5_reduce_relu'
    )
    brew.conv(
        model,
        prefix + '_5x5_reduce_relu',
        prefix + '_5x5_conv',
        conv5_reduce_channel,
        conv5_channel,
        kernel=5,
        stride=1,
        pad=2,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        prefix + '_5x5_conv',
        prefix + '_5x5_bn',
        conv5_channel
    )
    brew.relu(
        model,
        prefix + '_5x5_bn',
        prefix + '_5x5_relu'
    )
    # 3x3
    brew.conv(
        model,
        input_blob,
        prefix + '_3x3_reduce_conv',
        input_channel,
        conv3_reduce_channel,
        kernel=1,
        stride=1,
        pad=0,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        prefix + '_3x3_reduce_conv',
        prefix + '_3x3_reduce_bn',
        conv3_reduce_channel
    )
    brew.relu(
        model,
        prefix + '_3x3_reduce_bn',
        prefix + '_3x3_reduce_relu'
    )
    brew.conv(
        model,
        prefix + '_3x3_reduce_relu',
        prefix + '_3x3_conv',
        conv3_reduce_channel,
        conv3_channel,
        kernel=3,
        stride=1,
        pad=1,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        prefix + '_3x3_conv',
        prefix + '_3x3_bn',
        conv3_channel
    )
    brew.relu(
        model,
        prefix + '_3x3_bn',
        prefix + '_3x3_relu'
    )
    # 1x1
    brew.conv(
        model,
        input_blob,
        prefix + '_1x1_conv',
        input_channel,
        conv1_channel,
        kernel=1,
        stride=1,
        pad=0,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        prefix + '_1x1_conv',
        prefix + '_1x1_bn',
        conv1_channel
    )
    brew.relu(
        model,
        prefix + '_1x1_bn',
        prefix + '_1x1_relu'
    )
    # concat
    brew.concat(
        model,
        [prefix + '_pool_proj_relu',
         prefix + '_5x5_relu',
         prefix + '_3x3_relu',
         prefix + '_1x1_relu'],
        prefix + '_output'
    )
    return sum([pool_channel, conv5_channel,
                conv3_channel, conv1_channel])


def loss_block(model, input_blob, prefix,
               input_channel, num_labels, label=None):
    brew.conv(
        model,
        input_blob,
        prefix + '_conv',
        input_channel,
        128,
        kernel=1,
        stride=1,
        pad=0,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        prefix + '_conv',
        prefix + '_conv_bn',
        128
    )
    brew.relu(
        model,
        prefix + '_conv_bn',
        prefix + '_conv_relu'
    )
    brew.average_pool(
        model,
        prefix + '_conv_relu',
        prefix + '_pool',
        global_pooling=True
    )
    brew.fc(
        model,
        prefix + '_pool',
        prefix + '_fc',
        128,
        2048,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.relu(
        model,
        prefix + '_fc',
        prefix + '_fc_relu'
    )
    brew.dropout(
        model,
        prefix + '_fc_relu',
        prefix + '_fc_drop',
        ratio=0.5
    )
    brew.fc(
        model,
        prefix + '_fc_drop',
        prefix + '_pred_' + str(num_labels),
        2048,
        num_labels
    )
    if label == None:
        brew.softmax(
            model,
            prefix + '_pred_' + str(num_labels),
            prefix + '_softmax'
        )
    else:
        softmax, loss = model.SoftmaxWithLoss(
            [prefix + '_pred_' + str(num_labels), label],
            [prefix + '_softmax', prefix + '_loss']
        )


def create_inception_bn(
        model, data, input_channel,
        num_labels, label=None):
    # data-bn
    brew.spatial_bn(
        model,
        data,
        'data_bn',
        input_channel
    )
    # conv-1
    brew.conv(
        model,
        'data_bn',
        'conv1',
        input_channel,
        64,
        kernel=7,
        stride=2,
        pad=3,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        'conv1',
        'conv1_bn',
        64
    )
    brew.relu(
        model,
        'conv1_bn',
        'conv1_relu'
    )
    # pool1
    brew.max_pool(
        model,
        'conv1_relu',
        'pool1',
        kernel=3,
        stride=2
    )
    # conv2 reduce
    brew.conv(
        model,
        'pool1',
        'conv2_reduce',
        64,
        64,
        kernel=1,
        stride=1,
        pad=0,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        'conv2_reduce',
        'conv2_reduce_bn',
        64
    )
    brew.relu(
        model,
        'conv2_reduce_bn',
        'conv2_reduce_relu'
    )
    # conv2
    brew.conv(
        model,
        'conv2_reduce_relu',
        'conv2',
        64,
        192,
        kernel=3,
        stride=1,
        pad=1,
        bias_init=('ConstantFill', {'value': 0.2}),
        weight_init=('MSRAFill', {})
    )
    brew.spatial_bn(
        model,
        'conv2',
        'conv2_bn',
        192
    )
    brew.relu(
        model,
        'conv2_bn',
        'conv2_relu'
    )
    # pool2
    brew.max_pool(
        model,
        'conv2_relu',
        'pool2',
        kernel=3,
        stride=2
    )
    # inception_3a
    block_3a_channel = inception_block(
        model,
        'pool2',
        prefix='inception_3a',
        input_channel=192,
        pool_channel=32,
        conv5_reduce_channel=16,
        conv5_channel=32,
        conv3_reduce_channel=96,
        conv3_channel=128,
        conv1_channel=64
    )
    # inception_3b
    block_3b_channel = inception_block(
        model,
        'inception_3a_output',
        prefix='inception_3b',
        input_channel=block_3a_channel,
        pool_channel=64,
        conv5_reduce_channel=32,
        conv5_channel=96,
        conv3_reduce_channel=128,
        conv3_channel=192,
        conv1_channel=128
    )
    # pool3
    brew.max_pool(
        model,
        'inception_3b_output',
        'pool3',
        kernel=3,
        stride=2
    )
    # inception_4a
    block_4a_channel = inception_block(
        model,
        'pool3',
        prefix='inception_4a',
        input_channel=block_3b_channel,
        pool_channel=64,
        conv5_reduce_channel=16,
        conv5_channel=48,
        conv3_reduce_channel=96,
        conv3_channel=208,
        conv1_channel=192
    )
    # inception_4b
    block_4b_channel = inception_block(
        model,
        'inception_4a_output',
        prefix='inception_4b',
        input_channel=block_4a_channel,
        pool_channel=64,
        conv5_reduce_channel=24,
        conv5_channel=64,
        conv3_reduce_channel=112,
        conv3_channel=224,
        conv1_channel=160
    )
    # inception_4c
    block_4c_channel = inception_block(
        model,
        'inception_4b_output',
        prefix='inception_4c',
        input_channel=block_4b_channel,
        pool_channel=64,
        conv5_reduce_channel=24,
        conv5_channel=64,
        conv3_reduce_channel=128,
        conv3_channel=256,
        conv1_channel=128
    )
    # inception_4d
    block_4d_channel = inception_block(
        model,
        'inception_4c_output',
        prefix='inception_4d',
        input_channel=block_4c_channel,
        pool_channel=64,
        conv5_reduce_channel=32,
        conv5_channel=64,
        conv3_reduce_channel=144,
        conv3_channel=288,
        conv1_channel=112
    )
    # inception_4e
    block_4e_channel = inception_block(
        model,
        'inception_4d_output',
        prefix='inception_4e',
        input_channel=block_4d_channel,
        pool_channel=128,
        conv5_reduce_channel=32,
        conv5_channel=128,
        conv3_reduce_channel=160,
        conv3_channel=320,
        conv1_channel=256
    )
    # pool4
    brew.max_pool(
        model,
        'inception_4e_output',
        'pool4',
        kernel=3,
        stride=2
    )
    # inception_5a
    block_5a_channel = inception_block(
        model,
        'pool4',
        prefix='inception_5a',
        input_channel=block_4e_channel,
        pool_channel=128,
        conv5_reduce_channel=32,
        conv5_channel=128,
        conv3_reduce_channel=160,
        conv3_channel=320,
        conv1_channel=256
    )
    # inception_5b
    block_5b_channel = inception_block(
        model,
        'inception_5a_output',
        prefix='inception_5b',
        input_channel=block_5a_channel,
        pool_channel=128,
        conv5_reduce_channel=48,
        conv5_channel=128,
        conv3_reduce_channel=192,
        conv3_channel=384,
        conv1_channel=384
    )
    # loss 1
    loss_block(
        model,
        'inception_4a_output',
        prefix='cls1',
        input_channel=block_4a_channel,
        num_labels=num_labels,
        label=label
    )
    # loss 2
    loss_block(
        model,
        'inception_4d_output',
        prefix='cls2',
        input_channel=block_4d_channel,
        num_labels=num_labels,
        label=label
    )
    # loss 3
    loss_block(
        model,
        'inception_5b_output',
        prefix='cls3',
        input_channel=block_5b_channel,
        num_labels=num_labels,
        label=label
    )

    if label == None:
        return (
                ['cls1_softmax', 'cls2_softmax', 'cls3_softmax'],
               )
    else:
        return (
                ['cls1_softmax', 'cls2_softmax', 'cls3_softmax'],
                ['cls1_loss', 'cls2_loss', 'cls3_loss']
               )