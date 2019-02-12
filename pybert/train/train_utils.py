#encoding:utf-8
import os
import torch
# gpu
def prepare_device(n_gpu_use,logger):
    """
    setup GPU device if available, move model into configured device
    # 如果n_gpu_use为数字，则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
     """
    if isinstance(n_gpu_use,int):
        n_gpu_use = range(n_gpu_use)
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = range(0)
    if len(n_gpu_use) > n_gpu:
        msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device('cuda:%d'%n_gpu_use[0] if len(n_gpu_use) > 0 else 'cpu')
    list_ids = n_gpu_use
    return device, list_ids

# 加载模型
def restore_checkpoint(resume_path,model = None,optimizer = None):
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    if model:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return [model,optimizer,best,start_epoch]

# 判断环境 cpu还是gpu
def model_device(n_gpu,model,logger):
    device, device_ids = prepare_device(n_gpu,logger)
    if len(device_ids) > 1:
        logger.info("current {} GPUs".format(len(device_ids)))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model,device

