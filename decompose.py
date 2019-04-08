import torch
import torch.nn as nn
import torchvision

import numpy as np
from collections import OrderedDict

__all__ = ['VH_decompose_model', 'channel_decompose', 'network_decouple'\
           'EnergyThreshold', 'LinearRate', 'ValueThreshold']

# different criterions for sigma selection

class EnergyThreshold(object):

    def __init__(self, threshold, eidenval=True):
        """
        :param threshold: float, threshold to filter small valued sigma:
        :param eidenval: bool, if True, use eidenval as criterion, otherwise use singular
        """
        self.T = threshold
        assert self.T < 1.0 and self.T > 0.0
        self.eiden = eidenval

    def __call__(self, sigmas):
        """
        select proper numbers of singular values
        :param sigmas: numpy array obj which containing singular values
        :return: valid_idx: int, the number of sigmas left after filtering
        """
        if self.eiden:
            energy = sigmas**2
        else:
            energy = sigmas

        sum_e = torch.sum(energy)
        valid_idx = sigmas.size(0)
        for i in range(energy.size(0)):
            if energy[:(i+1)].sum()/sum_e >= self.T:
                valid_idx = i+1
                break

        return valid_idx

class LinearRate(object):

    def __init__(self, rate):
        """
        filter out small valued singulars according to given proportion
        :param rate: value, left proportion
        """
        self.rate = rate

    def __call__(self, sigmas):
        return int(sigmas.size(0)*self.rate)

class ValueThreshold(object):

    def __init__(self, threshold):
        """
        filter out small valued singulars according to a given value threshold
        :param threshold: float, value threshold
        """
        self.T = threshold

    def __call__(self, sigmas):
        # input sigmas should be a sorted array from large to small
        valid_idx = sigmas.size(0)
        for i in range(len(sigmas)):
            if sigmas[i] < self.T:
                valid_idx = i
                break
        return valid_idx

def _set_model_attr(field_name, att, obj):
    '''
    set a certain filed_name like 'xx.xx.xx' as the object att
    :param field_name: str, 'xx.xx.xx' to indicate the attribution tree
    :param att: input object to replace a certain field attribution
    :param obj: objective attribution
    '''

    field_list = field_name.split('.')
    a = att

    # to achieve the second last level of attr tree
    for field in field_list[:-1]:
        a = getattr(a, field)

    setattr(a, field_list[-1], obj)

def pd_conv(cin, cout, kernel, stride, pad, bias):
    return nn.Sequential(
        OrderedDict([
            ('pw', nn.Conv2d(cin, cout, 1, 1, 0, bias=False)),
            ('dw', nn.Conv2d(cout, cout, kernel, stride, pad, groups=cout, bias=bias))
            ])
        )

class MultiPathConv(nn.Module):

    '''
    a sub module structure used for network decouple as follows
         
         /--conv 1--\   
        /            \
    --> ----conv 2--->+--->
        \            /
         \--conv n--/
    '''

    def __init__(self, n, cin, cout, kernel, pad, stride, bias):
        super(MultiPathConv, self).__init__()

        self.path_num = n
        self.path = nn.ModuleList([pd_conv(cin, cout, kernel, stride, pad, bias=(i==0 and bias)) for i in range(n)])

    def forward(self, x):
        output = 0.0
        for m in self.path:
            output += m(x)
        return output


# different low-rank decomposition scheme

def network_decouple(model_in, look_up_table, criterion, train=True, lambda_=0.0003, truncate=None):
    '''
    decouple a input pre-trained model under nuclear regularization
    with singular value decomposition

    a single NxCxHxW low-rank filter is decoupled
    into a parrallel path composed of point-wise conv followed by depthwise conv

    :param model_in: object of derivated class of nn.Module, the model is initialized with pre-trained weight
    :param look_up_table: list, containing module names to be decouple
    :param criterion: object, a filter to filter out small valued simgas, only valid when train is False
    :param train: bool, whether decompose during training, if true, function only compute corresponding
           gradient w.r.t each singular value and do not apply actual decouple
    :param lambda_: float, weight for regularization term, only valid when train is True
    :return: model_out: a new nn.Module object initialized with a decoupled model
    '''
    if train:
        model_in.train()
    else:
        model_in.eval()

    for name, m in model_in.named_modules():

        if name in look_up_table:
            param = m.weight.data
            dim = param.size()
            
            if m.bias:             
                hasb = True
                b = m.bias.data
            else:
                hasb = False
            
            try:
                valid_idx = []
                # compute average rank according to criterion
                for i in range(dim[0]):
                    W = param[i, :, :, :].view(dim[1], -1)
                    U, sigma, V = torch.svd(W, some=True)
                    valid_idx.append(criterion(sigma))
                item_num = min(max(valid_idx), min(dim[2]*dim[3], dim[1]))
                
                pw = [param.new_zeros((dim[0], dim[1], 1, 1)) for i in range(item_num)]
                dw = [param.new_zeros((dim[0], 1, dim[2], dim[3])) for i in range(item_num)]

                # svd decoupling
                for i in range(dim[0]):
                    W = param[i, :, :, :].view(dim[1], -1)
                    U, sigma, V = torch.svd(W, some=True)
                    V = V.t()
                    U = U[:, :item_num].contiguous()
                    V = V[:item_num, :].contiguous()
                    sigma = torch.diag(torch.sqrt(sigma[:item_num]))
                    U = U.mm(sigma)
                    V = sigma.mm(V)
                    V = V.view(item_num, dim[2], dim[3])
                    for j in range(item_num):
                        pw[j][i, :, 0, 0] = U[:, j]
                        dw[j][i, 0, :, :] = V[j, :, :]

               
            except Exception as e:
                print(e)
                raise Exception('svd failed during decoupling')

            new_m = MultiPathConv(item_num, cin=dim[1], cout=dim[0], kernel=m.kernel_size, stride=m.stride, pad=m.padding, bias=hasb)
    
            state_dict = new_m.state_dict()
            for i in range(item_num):
                dest = 'path.%d.pw.weight' % i
                src = '%s.weight' % name
                print(dest+' <-- '+src)
                state_dict[dest].copy_(pw[i])

                dest = 'path.%d.dw.weight' % i
                print(dest+' <-- '+src)
                state_dict[dest].copy_(dw[i])

                if i == 0 and hasb:
                    dest = 'path.%d.dw.bias' % i
                    src = '%s.bias' % name
                    print(dest+' <-- '+src)
                    state_dict[dest].copy_(b)

            new_m.load_state_dict(state_dict)
            _set_model_attr(name, att=model_in, obj=new_m)

    return model_in.cuda()

def channel_decompose(model_in, look_up_table, criterion, train=True, lambda_=0.0003, truncate=None):
    '''
    decouple a input pre-trained model under nuclear regularization
    with singular value decomposition

    a single NxCxHxW low-rank filter is decoupled
    into a NxRx1x1 kernel following a RxCxHxW kernel

    :param model_in: object of derivated class of nn.Module, the model is initialized with pre-trained weight
    :param look_up_table: list, containing module names to be decouple
    :param criterion: object, a filter to filter out small valued simgas, only valid when train is False
    :param train: bool, whether decompose during training, if true, function only compute corresponding
           gradient w.r.t each singular value and do not apply actual decouple
    :param lambda_: float, weight for regularization term, only valid when train is True
    :return: model_out: a new nn.Module object initialized with a decoupled model
    '''
    if train:
        model_in.train()
    else:
        model_in.eval()

    for name, m in model_in.named_modules():

        if name in look_up_table:
            param = m.weight.data
            dim = param.size()
            
            if m.bias:             
                hasb = True
                b = m.bias.data
            else:
                hasb = False
            
            NC = param.view(dim[0], -1) # [N x CHW]

            try:
                N, sigma, C = torch.svd(NC, some=True)
                C = C.t()
                # remain large singular value
                if not train:
                    valid_idx = criterion(sigma)
                    N = N[:, :valid_idx].contiguous()
                    sigma = sigma[:valid_idx]
                    C = C[:valid_idx, :]
                else:
                    subgradient = torch.mm(N, C)
                    subgradient = subgradient.contiguous().view(dim[0],dim[1],dim[2],dim[3])
            except:
                if train:
                    subgradient = 0.0
                else:
                    raise Exception('svd failed during decoupling')

            if train:
                m.weight.grad.data.add_(lambda_ * subgradient)
            elif m.stride == (1, 1):  # when decoupling, only conv with 1x1 stride is considered
                r = int(sigma.size(0))
                C = torch.mm(torch.diag(torch.sqrt(sigma)), C)
                N = torch.mm(N,torch.diag(torch.sqrt(sigma)))

                C = C.view(r,dim[1],dim[2], dim[3])
                N = N.view(dim[0], r, 1, 1)

                new_m = nn.Sequential(
                    OrderedDict([
                        ('C', nn.Conv2d(dim[1], r, dim[2], 1, 1, bias=False)),
                        ('N', nn.Conv2d(r, dim[0], 1, 1, 0, bias=hasb))
                    ])
                )
        
             
                state_dict = new_m.state_dict()
                print(name+'.C.weight'+' <-- '+name+'.weight')
                state_dict['C.weight'].copy_(C)
                print(name + '.N.weight' + ' <-- ' + name + '.weight')

                state_dict['N.weight'].copy_(N)
                if hasb:
                    print(name+'.N.bias'+' <-- '+name+'.bias')
                    state_dict['N.bias'].copy_(b)

                new_m.load_state_dict(state_dict)
                _set_model_attr(name, att=model_in, obj=new_m)

    return model_in.cuda()


def VH_decompose_model(model_in, look_up_table, criterion, train=True, lambda_=0.0003, truncate=1.0):
    '''
    decouple a input pre-trained model under nuclear regularization
    with singular value decomposition

    a single NxCxHxW low-rank filter is decoupled
    into a RxCxVxW kernel and a NxRxWxH kernel

    :param model_in: object of derivated class of nn.Module, the model is initialized with pre-trained weight
    :param look_up_table: list, containing module names to be decouple
    :param criterion: object, a filter to filter out small valued simgas, only valid when train is False
    :param train: bool, whether decompose during training, if true, function only compute corresponding
           gradient w.r.t each singular value and do not apply actual decouple
    :param lambda_: float, weight for regularization term, only valid when train is True
    :return: model_out: a new nn.Module object initialized with a decoupled model
    '''

    if train:
        model_in.train()
    else:
        model_in.eval()

    for name, m in model_in.named_modules():
        if name in look_up_table:
            # the module should be decoupled
            param = m.weight.data
            if m.bias:
                hasb = True
                b = m.bias.data # Tensor size N
            else:
                hasb = False

            dim = param.size()
            VH = param.permute(1, 2, 0, 3).contiguous().view(dim[1] * dim[2], -1)

            try:
                V, sigma, H = torch.svd(VH, some=True)
                H = H.t()
                # remain large singular value
                if train:
                    subgradient = torch.mm(V, H)
                    subgradient = subgradient.contiguous().view(dim[1], dim[2], dim[0], dim[3]).permute(2, 0, 1, 3)
                    #print(sigma)
                else:
                    valid_idx = criterion(sigma)
                    V = V[:, :valid_idx].contiguous()
                    sigma = sigma[:valid_idx]
                    H = H[:valid_idx, :]
            except:
                if train:
                    subgradient = 0.0
                else:
                    raise Exception('svd failed during decoupling')

            if train:
                m.weight.grad.data.add_(lambda_*subgradient)
            elif m.stride == (1,1): # when decoupling, only conv with 1x1 stride is considered
                r = int(sigma.size(0))
                H = torch.mm(torch.diag(sigma), H).contiguous()

                H = H.view(r, dim[0], dim[3], 1).permute(1,0,3,2)
                V = V.view(dim[1], 1, dim[2], r).permute(3,0,2,1)

                new_m = nn.Sequential(
                OrderedDict([
                    ('V', nn.Conv2d(dim[1], r, kernel_size=(int(dim[2]),1),stride=(1, 1),padding=(m.padding[0],0),  bias=False)),
                    ('H', nn.Conv2d(r, dim[0], kernel_size=(1,int(dim[3])),stride=(1, 1),padding=(0,m.padding[1]),  bias=hasb))])
                )

                state = new_m.state_dict()
                print(name+'.V.weight' + ' <-- ' + name+'.weight')
                state['V.weight'].copy_(V)
                print(name+'.H.weight' + ' <-- ' + name+'.weight')
                state['H.weight'].copy_(H)

                if m.bias:
                    print(name+'.H.bias' + ' <-- ' + name+'.bias')
                    state['H.bias'].copy_(b)

                new_m.load_state_dict(state)

                _set_model_attr(name, att=model_in, obj=new_m)

    return model_in.cuda()
 
