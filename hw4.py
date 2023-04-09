"""

HW4

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
import math

"""

function view_as_windows

"""

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).
    Parameters
    ----------
    arr_in : Pytorch tensor
        N-d Pytorch tensor.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.
    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.
    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle
    Examples
    --------
    >>> import torch
    >>> A = torch.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])
    >>> A = torch.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = torch.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not torch.is_tensor(arr_in):
        raise TypeError("`arr_in` must be a pytorch tensor")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = torch.tensor(arr_in.shape)
    window_shape = torch.tensor(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    # window_strides = torch.tensor(arr_in.stride())
    window_strides = arr_in.stride()

    indexing_strides = arr_in[slices].stride()

    win_indices_shape = torch.div(arr_shape - window_shape
                          , torch.tensor(step), rounding_mode = 'floor') + 1
    
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = torch.as_strided(arr_in, size=new_shape, stride=strides)
    return arr_out

#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    # out_ch_size: output channel size(==number of filters)
    def __init__(self, f_height, f_width, input_size, in_ch_size, out_ch_size):
        
        # Xavier init

        # W가 filter임!
        self.W = torch.normal(0, 1 / math.sqrt((in_ch_size * f_height * f_width / 2)),
                                  size=(out_ch_size, in_ch_size, f_height, f_width))
        self.b = 0.01 + torch.zeros(size=(1, out_ch_size, 1, 1))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()

    #######
    # Q1. Complete this method
    # Performs a forward pass of convolutional layer.
    # Input parameter x: tensor
    # 4-dimensional torch.tensor input map to the convolutional layer. Each dimension of x has meaning
    #       x.shape=(batch_size, input_channel_size, in_width, in_height)
    # For example, if x.shape returns (16, 3, 32, 32), it means that x is an image/activation map of size 32X32, with three input channels(like RGB),
    # and there are 16 of them treated as one batch (as in mini-batch SGD).
    # Output is out: tensor
    # 4-dimensional torch.tensor output feature map as a result of convolution. Each dimension of out has meaning
    #       out.shape=(batch_size, num_filter, out_width, out_height)
    # For example, if out.shape will return (16, 8, 28, 28), for the input x with shape (16, 3, 32, 32): 
    #       if the convolutional layer has each filter size 5*5*3, and there are a total of 8 filters.
    # Note that the input width and height changes from 32 to 28. "Batch size remains the same!"
    # The forward pass of convolution will be done "without" zero-padding and "stride of 1".
    # So, if the input map has width N, and the filter size is given by F, "the width of output map" will be "N-F+1"
    #######
    def forward(self, x):
        # x.shape = torch.Size([8, 3, 32, 32])
        # y.shape = torch.Size([1, 1, 30, 30, 8, 3, 3, 3])
        
        # out_ch_size = 8
        out_ch_size = x.shape[0]

        # out: 최종 리턴할 것
        out = []
        
        # 이미지가 8개니까 8번 반복문 돌림
        for i in range(out_ch_size):
            # 이미지 한장
            img = x[i]  # img.shape: (3,32,32)
            
            # 이미지에서 3*3*3을 자른 거를 30*30만큼 가지고 있는 y
            y = view_as_windows(img, self.W[0].shape)
            y = torch.squeeze(y, axis=0)
            y = y.reshape(30, 30, -1)   # --> (30, 30, 27)
            y = y.reshape(-1, 27)   # --> (30x30, 27)
            y = np.transpose(y)     # --> (27, 30x30)
            
            # 필터를 (8, 27)짜리로 변형하기
            filt = self.transform_filter(self.W)
            
            # 이제 내적해서 최종 (8,30,30) 짜리를 만들기
            tmp = torch.matmul(filt, y)
            tmp = tmp.reshape(8, 30, -1)
            
            # 해당 tmp를 out에 append하기
            out.append(tmp)
        
        # out의 shape은 (8,8,30,30)이어야 함.
        out = torch.stack(out, 0)
        return out
    
    
    #######
    ## If necessary, you can define additional class methods here
    #######
    def transform_filter(self, W):
        out_ch_size = W.shape[0]
        ret = W.reshape(out_ch_size, -1)
        return ret

    

# pool_size: window size of pooling
# stride: stride of pooling

class nn_max_pooling_layer:
    def __init__(self, pool_size, stride):
        # pool_size == 2
        # stride == 2
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q2. Complete this method
    # Input parameter x: tensor
    # 4-dimensional torch.tensor input map to the convolutional layer. Each dimension of x has meaning
    #       x.shape=(batch_size, input_channel_size, in_width, in_height)
    # 4-dimensional torch.tensor output feature map as a result of convolution. Each dimension of out has meaning
    #       out.shape=(batch_size, input_channel_size, out_width, out_height)    
    # For example, if out.shape will return (16,3,16,16), for the input x with shape (16,3,32,32),
    # assuming that the stride=2 and pool_size=2.
    # 
    # Tips for maxpool operations
    # It is convenient to use view_as_windows function with parameter "step", which is equivalent to the stride.
    #######
    def forward(self, x):
        # x.shape = torch.Size([8, 3, 32, 32])
        
        out_ch_size = x.shape[0]
        
        # out: 최종 리턴할 것
        out = torch.zeros(8,3,16,16)
        
        # 이미지가 8개니까 반복문을 8번 돌림
        for i in range(out_ch_size):
            # 이미지 한장
            img = x[i]  # img.shape: (3,32,32)
            
            # 윈도우는 (1,2,2)
            y = view_as_windows(img, (1,2,2), step=(1,2,2)) #y.shape: torch.Size([3,16,16,1,2,2])
            y = y.reshape(3,16,16,4)
            
            # 채널 하나마다 max_pooling을 해주고 다시 합쳐준다.
            ch_size = y.shape[0] #3
            temp = torch.zeros(3,16,16) #여기에 이미지 한장 당 max_pooling 된 결과를 넣어줄 거임.
            
            # 이미지 하나의 채널 하나마다 max_pooling을 해준다.
            for j in range(ch_size):
                channel = y[j]  #channel shape: (16,16,4)
                channel = channel.reshape(-1, 4) #channel shape: (16*16, 4)
                temp2 = torch.zeros(16*16, 1) #(16*16, 4)짜리 temp 하나 더 만든다.
                
                #16*16번만큼 반복문을 돌려서 max를 뽑아낸다.
                for k in range(channel.shape[0]):
                    temp2[k] = max(channel[k])
                    
                temp2 = temp2.reshape(16, -1)   #(16, 16)으로 reshape해준다.
                temp[j] = temp2                 #이제 이 결과가 temp이미지의 채널 하나가 된다.
        
            out[i] = temp   #이제 이 결과가 out의 이미지 하나가 된다.
        # (N - F) / stride + 1 == (32-2)/2 + 1 == 16
        
        # out의 shape은 (8, 3, 16, 16)이다.
        return out
    
    #######
    ## If necessary, you can define additional class methods here
    #######

"""
TESTING 
"""

if __name__ == "__main__":

    # data sizes
    batch_size = 8
    input_size = 32
    filter_width = 3
    filter_height = filter_width
    in_ch_size = 3
    num_filters = 8

    std = 1e0
    dt = 1e-3

    # number of test loops
    num_test = 50

    # error parameters
    err_fwd = 0
    err_pool = 0


    # for reproducibility
    # torch.manual_seed(0)

    # set default type to float64
    torch.set_default_dtype(torch.float64)

    print('conv test')
    for i in range(num_test):
        # create convolutional layer object
        cnv = nn_convolutional_layer(filter_height, filter_width, input_size,
                                   in_ch_size, num_filters)

        # test conv layer from torch.nn for reference
        test_conv_layer = nn.Conv2d(in_channels=in_ch_size, out_channels=num_filters,
                                kernel_size = (filter_height, filter_width))
        
        # test input
        x = torch.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
        
        with torch.no_grad():
            
            out = cnv.forward(x)
            W,b = cnv.get_weights()
            test_conv_layer.weight = nn.Parameter(W)
            test_conv_layer.bias = nn.Parameter(torch.squeeze(b))
            test_out = test_conv_layer(x)
            
            err=torch.norm(test_out - out)/torch.norm(test_out)
            err_fwd+= err
    
    stride = 2
    pool_size = 2
    
    print('pooling test')
    for i in range(num_test):
        # create pooling layer object
        mpl = nn_max_pooling_layer(pool_size=pool_size, stride=stride)
        
        # test pooling layer from torch.nn for reference
        test_pooling_layer = nn.MaxPool2d(kernel_size=(pool_size,pool_size), stride=stride)
        
        # test input
        x = torch.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
        
        with torch.no_grad():
            out = mpl.forward(x)
            test_out = test_pooling_layer(x)
            
            err=torch.norm(test_out - out)/torch.norm(test_out)
            err_pool+= err

    # reporting accuracy results.
    print('accuracy results')
    print('forward accuracy', 100 - err_fwd/num_test*100, '%')
    print('pooling accuracy', 100 - err_pool/num_test*100, '%')