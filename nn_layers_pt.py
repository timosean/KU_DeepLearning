"""

nn_layers_pt.py

PyTorch version of nn_layers

"""

import torch
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

    def __init__(self, f_height, f_width, input_size, in_ch_size, out_ch_size):
        
        # Xavier init
        self.W = torch.normal(0, 1 / math.sqrt((in_ch_size * f_height * f_width / 2)),
                                  size=(out_ch_size, in_ch_size, f_height, f_width))
        self.b = 0.01 + torch.zeros(size=(1, out_ch_size, 1, 1))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.v_W = torch.zeros_like(self.W)
        self.v_b = torch.zeros_like(self.b)
        
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()

    def forward(self, x):
        
        ###################################
        # Q4. Implement your layer here
        # Takes a batch of MNIST images as input x.
        # This convolutional layer has 28 filters.
        
        # x.shape = (60000, 1, 28, 28) = (batch_size, in_ch_size, in_width, in_height)
        # y.shape = (60000, 28, 26, 26) = (batch_size, num_filters, out_width, out_height)
        # W.shape = (28, 1, 3, 3) = (out_ch_size, in_ch_size, f_height, f_width)
        
        # f_height = 3
        # f_width = 3
        # input_size = 28
        # in_ch_size = 1
        # out_ch_size = 28 (필터개수)
        
        batch_size = x.shape[0]

        # out: 최종 리턴할 것
        out = []
        
        # 이미지가 batch_size개니까 batch_size번 반복문 돌림
        for i in range(batch_size):
            # 이미지 한장
            img = x[i]  # img.shape: (1,28,28)
            
            # 이미지에서 1*3*3을 자른 거를 26*26만큼 가지고 있는 y
            y = view_as_windows(img, self.W[0].shape)   # --> (1,26,26,1,3,3)
            y = torch.squeeze(y, axis=0)    # --> (26,26,1,3,3)
            y = y.reshape(26, 26, -1)   # --> (26, 26, 9)
            y = y.reshape(-1, 9)   # --> (26x26, 9)
            y = np.transpose(y)     # --> (9, 26x26)
            
            # 필터를 (28, 9)짜리로 변형하기
            filt = self.transform_filter(self.W)
            
            # 이제 내적해서 최종 (28,26,26) 짜리를 만들기
            tmp = torch.matmul(filt, y)
            tmp = tmp.reshape(28, 26, -1)
            
            # 해당 tmp를 out에 append하기
            out.append(tmp)
        
        # out의 shape은 (batch_size,28,26,26)이어야 함.
        out = torch.stack(out, 0)
        return out + self.b
        ###################################
        pass
    
    def transform_filter(self, W):
        out_ch_size = W.shape[0]
        ret = W.reshape(out_ch_size, -1)
        return ret
    
    def step(self, lr, friction):
        with torch.no_grad():
            self.v_W = friction*self.v_W + (1-friction)*self.W.grad
            self.v_b = friction*self.v_b + (1-friction)*self.b.grad
            self.W -= lr*self.v_W
            self.b -= lr*self.v_b
            
            self.W.grad.zero_()
            self.b.grad.zero_()

# max pooling
class nn_max_pooling_layer:
    def __init__(self, pool_size, stride):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        ###################################
        # Q5. Implement your layer here
        # A 2x2 maxpooling is done with stride 2.
        
        # pool_size = 2
        # stride = 2
        
        # x.shape = torch.Size([batch_size, 28, 26, 26])
        # y.shape = torch.Size([batch_size, 28, 13, 13])
        
        batch_size = x.shape[0]
        
        y = view_as_windows(x, (1,1,2,2), step=(1,1,2,2))   # --> y = (batch_size, 28, 13, 13, 1, 1, 2, 2)
        y = y.reshape(batch_size, 28, 13, 13, -1)
        y = y.reshape(-1, 4)
        
        # batch_size*28*13*13 만큼 반복문을 돌려서 max풀링한다.
        y = torch.max(y, axis=1).values
            
        y = y.reshape(batch_size,28,13,13)
        
        out = y
        
        #############################################################################################
        #############################################################################################
        # 이미지가 batch_size개니까 반복문을 batch_size번 돌림
        # for i in range(batch_size):
        #     # 이미지 한장
        #     img = x[i]  # img.shape: (28,26,26)
            
        #     # 윈도우는 (1,2,2)
        #     y = view_as_windows(img, (1,2,2), step=(1,2,2)) # y.shape: torch.Size([28,13,13,1,2,2])
        #     y = y.reshape(28,13,13,4)
            
        #     # 채널 하나마다 max_pooling을 해주고 다시 합쳐준다.
        #     ch_size = y.shape[0] # 28
        #     temp = torch.zeros(28,13,13) #여기에 이미지 한장 당 max_pooling 된 결과를 넣어줄 거임.
            
        #     # 이미지 하나의 채널 하나마다 max_pooling을 해준다.
        #     for j in range(ch_size):
        #         channel = y[j]  #channel shape: (13,13,4)
        #         channel = channel.reshape(-1, 4) #channel shape: (13*13, 4)
        #         temp2 = torch.zeros(13*13, 1) #(13*13, 4)짜리 temp 하나 더 만든다.
                
        #         #13*13번만큼 반복문을 돌려서 max를 뽑아낸다.
        #         for k in range(channel.shape[0]):
        #             temp2[k] = max(channel[k])
                    
        #         temp2 = temp2.reshape(13, -1)   #(13, 13)으로 reshape해준다.
        #         temp[j] = temp2                 #이제 이 결과가 temp이미지의 채널 하나가 된다.
        
        #     out[i] = temp   #이제 이 결과가 out의 이미지 하나가 된다.
            #############################################################################################
            #############################################################################################
        
        return out
        ###################################
        pass

# relu activation
class nn_activation_layer:

    # linear layer. creates matrix W and bias b
    # W is in by out, and b is out by 1
    def __init__(self):
        pass

    def forward(self, x):
        return x.clamp(min=0)

# fully connected (linear) layer
class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        
        # Xavier/He init
        self.W = torch.normal(0, std/math.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+torch.zeros((output_size))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.v_W = torch.zeros_like(self.W)
        self.v_b = torch.zeros_like(self.b)

    ## Q1
    def forward(self,x):
        # compute forward pass of given parameter
        # output size is batch x output_size x 1 x 1
        # input size is batch x input_size x filt_size x filt_size
        output_size = self.W.shape[0]
        batch_size = x.shape[0]
        #Wx=x.reshape((batch_size,-1))@(self.W.reshape(output_size,-1)).T
        Wx = torch.mm(x.reshape((batch_size, -1)),(self.W.reshape(output_size, -1)).T)
        out = Wx+self.b
        return out

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()
    
    def step(self, lr, friction):
        with torch.no_grad():
            self.v_W = friction*self.v_W + (1-friction)*self.W.grad
            self.v_b = friction*self.v_b + (1-friction)*self.b.grad
            self.W -= lr*self.v_W
            self.b -= lr*self.v_b
            self.W.grad.zero_()
            self.b.grad.zero_()


# softmax layer
class nn_softmax_layer:
    def __init__(self):
        pass

    def forward(self, x):
        s = x - torch.unsqueeze(torch.amax(x, axis=1), -1)
        return (torch.exp(s) / torch.unsqueeze(torch.sum(torch.exp(s), axis=1), -1)).reshape((x.shape[0],x.shape[1]))


# cross entropy layer
class nn_cross_entropy_layer:
    def __init__(self):
        self.eps=1e-15

    def forward(self, x, y):
        # first get softmax
        batch_size = x.shape[0]
        num_class = x.shape[1]
        # onehot = torch.zeros((batch_size, num_class))
        # onehot[range(batch_size), list(y.reshape(-1))] = 1
        onehot = np.zeros((batch_size, num_class))
        onehot[range(batch_size), (np.array(y)).reshape(-1, )] = 1
        onehot = torch.as_tensor(onehot)

        # avoid numerial instability
        x[x<self.eps]=self.eps
        x=x/torch.unsqueeze(torch.sum(x,axis=1), -1)

        return sum(-torch.sum(torch.log(x.reshape(batch_size, -1)) * onehot, axis=0)) / batch_size
