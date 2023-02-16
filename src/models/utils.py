import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_emb_extractor(emb_extractor_model, checkpoint_path, *args, **kwargs):
    """
    return embedding extractor model
    """
    emb_extractor = emb_extractor_model(*args, **kwargs)
    #checkpoint = torch.load(checkpoint_path, map_location=device)
    #emb_extractor.load_state_dict(checkpoint['model'])
    emb_extractor.to(device)

    return emb_extractor


def get_model(mymodel, emb_extractor, *args, **kwargs):
    model = mymodel(emb_extractor, *args, **kwargs).to(device)
    return model


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out
    

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


def forward(model, generator, return_input=False, 
    return_target=False):
    """Forward data to a model.
    
    Args: 
      model: object
      generator: object
      return_input: bool
      return_target: bool

    Returns:
      audio_name: (audios_num,)
      clipwise_output: (audios_num, classes_num)
      (ifexist) segmentwise_output: (audios_num, segments_num, classes_num)
      (ifexist) framewise_output: (audios_num, frames_num, classes_num)
      (optional) return_input: (audios_num, segment_samples)
      (optional) return_target: (audios_num, classes_num)
    """
    output_dict = {}
    device = next(model.parameters()).device
    time1 = time.time()

    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        print(n)
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)

        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

        append_to_dict(output_dict, 'clipwise_output', 
            batch_output['clipwise_output'].data.cpu().numpy())

        if 'segmentwise_output' in batch_output.keys():
            append_to_dict(output_dict, 'segmentwise_output', 
                batch_output['segmentwise_output'].data.cpu().numpy())

        if 'framewise_output' in batch_output.keys():
            append_to_dict(output_dict, 'framewise_output', 
                batch_output['framewise_output'].data.cpu().numpy())
            
        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])
            
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])

        if n % 10 == 0:
            print(' --- Inference time: {:.3f} s / 10 iterations ---'.format(
                time.time() - time1))
            time1 = time.time()

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, audio_length):
    """Count flops. Code modified from others' implementation.
    """
    multiply_adds = True
    list_conv2d=[]
    def conv2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_conv2d.append(flops)

    list_conv1d=[]
    def conv1d_hook(self, input, output):
        batch_size, input_channels, input_length = input[0].size()
        output_channels, output_length = output[0].size()
 
        kernel_ops = self.kernel_size[0] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_length
 
        list_conv1d.append(flops)
 
    list_linear=[] 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
 
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
 
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
 
    list_bn=[] 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)
 
    list_relu=[] 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement() * 2)
 
    list_pooling2d=[]
    def pooling2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_pooling2d.append(flops)

    list_pooling1d=[]
    def pooling1d_hook(self, input, output):
        batch_size, input_channels, input_length = input[0].size()
        output_channels, output_length = output[0].size()
 
        kernel_ops = self.kernel_size[0]
        bias_ops = 0
        
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_length
 
        list_pooling2d.append(flops)
 
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, nn.Conv2d):
                net.register_forward_hook(conv2d_hook)
            elif isinstance(net, nn.Conv1d):
                net.register_forward_hook(conv1d_hook)
            elif isinstance(net, nn.Linear):
                net.register_forward_hook(linear_hook)
            elif isinstance(net, nn.BatchNorm2d) or isinstance(net, nn.BatchNorm1d):
                net.register_forward_hook(bn_hook)
            elif isinstance(net, nn.ReLU):
                net.register_forward_hook(relu_hook)
            elif isinstance(net, nn.AvgPool2d) or isinstance(net, nn.MaxPool2d):
                net.register_forward_hook(pooling2d_hook)
            elif isinstance(net, nn.AvgPool1d) or isinstance(net, nn.MaxPool1d):
                net.register_forward_hook(pooling1d_hook)
            else:
                print('Warning: flop of module {} is not counted!'.format(net))
            return
        for c in childrens:
            foo(c)

    # Register hook
    foo(model)
    
    device = device = next(model.parameters()).device
    input = torch.rand(1, audio_length).to(device)

    out = model(input)
 
    total_flops = sum(list_conv2d) + sum(list_conv1d) + sum(list_linear) + \
        sum(list_bn) + sum(list_relu) + sum(list_pooling2d) + sum(list_pooling1d)
    
    return total_flops




class Interpolator(nn.Module):
    def __init__(self, ratio, interpolate_mode='nearest'):
        """Interpolate the sound event detection result along the time axis.
        Args:
            ratio: int
            interpolate_mode: str
        """
        super(Interpolator, self).__init__()

        if interpolate_mode == 'nearest':
            self.interpolator = NearestInterpolator(ratio)

        elif interpolate_mode == 'linear':
            self.interpolator = LinearInterpolator(ratio)
        
    def forward(self, x):
        """Interpolate the sound event detection result along the time axis.
        
        Args:
            x: (batch_size, time_steps, classes_num)
        Returns:
            (batch_size, new_time_steps, classes_num)
        """
        return self.interpolator(x)


class NearestInterpolator(nn.Module):
    def __init__(self, ratio):
        """Nearest interpolate the sound event detection result along the time axis.
        Args:
            ratio: int
        """
        super(NearestInterpolator, self).__init__()

        self.ratio = ratio

    def forward(self, x):
        """Interpolate the sound event detection result along the time axis.
        
        Args:
            x: (batch_size, time_steps, classes_num)
        Returns:
            upsampled: (batch_size, new_time_steps, classes_num)
        """
        (batch_size, time_steps, classes_num) = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, self.ratio, 1)
        upsampled = upsampled.reshape(batch_size, time_steps * self.ratio, classes_num)
        return upsampled


class LinearInterpolator(nn.Module):
    def __init__(self, ratio):
        """Linearly interpolate the sound event detection result along the time axis.
        Args:
            ratio: int
        """
        super(LinearInterpolator, self).__init__()

        self.ratio = ratio
    
        weight = torch.zeros(ratio * 2 + 1)

        for i in range(ratio):
            weight[i] = i / ratio

        for i in range(ratio, ratio * 2 + 1):
            weight[i] = 1. - (i - ratio) / ratio

        weight = weight[None, None, :]

        self.register_buffer('weight', weight, persistent=False)
        
    def forward(self, x):
        """Interpolate the sound event detection result along the time axis.
        
        Args:
            x: (batch_size, time_steps, classes_num)
        Returns:
            upsampled: (batch_size, new_time_steps, classes_num)
        """
        batch_size, time_steps, classes_num = x.shape
        x = x.transpose(1, 2).reshape(batch_size * classes_num, 1, time_steps)

        upsampled = F.conv_transpose1d(
            input=x, 
            weight=self.weight, 
            bias=None, 
            stride=self.ratio, 
            padding=self.ratio, 
            output_padding=0
        )
        new_time_steps = upsampled.shape[-1]
        upsampled = upsampled.reshape(batch_size, classes_num, new_time_steps)
        upsampled = upsampled.transpose(1, 2)
        return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def do_mixup(x, mixup_lambda):
    out = x[0::2].transpose(0, -1) * mixup_lambda[0::2] + \
        x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    return out.transpose(0, -1)


def get_aucpr(predictions, labels):
    predictions, labels = np.array(predictions), np.array(labels, dtype = int)
    idx = np.argsort(predictions)[::-1]
    predictions = predictions[idx]
    labels = labels[idx]
    cumsum = np.cumsum(labels)
    rank = np.arange(len(cumsum)) + 1
    Num = cumsum[-1]
    prec = cumsum / rank
    rec = cumsum / Num
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    pr_auc = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return pr_auc

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    print('model loaded from %s' % checkpoint_path)
    return model

def get_metric(model, dataloaders):
    phase = "test"
    test_dataloader = dataloaders[phase]
    preds = []
    y_true = []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device)
            outputs = list(model(inputs).to("cpu").detach().numpy()[:, 0])
            preds += outputs
            y_true += list(labels.detach().numpy())
    aucpr = get_aucpr(np.array(preds), np.array(y_true))
    auc_roc = roc_auc_score(np.array(y_true), np.array(preds))
    print(f"AUC_PR: {aucpr:.4f}")
    print(f"AUC_ROC: {auc_roc:.4f}")