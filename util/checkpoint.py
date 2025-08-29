''' Functions for model checkpointing
'''

import torch as T
from torch.nn.parallel import DistributedDataParallel as DDP

def ensure_extension(filename, extension):
    if not filename.endswith(extension):
        filename += extension
    return filename

def convert_distributed_to_local(model_state_dict):
    return {key.replace('module.', '', 1): value for key, value in model_state_dict.items()}

def make_checkpoint(
        model, 
        optimiser = None, 
        schedular = None, 
        scalar = None, 
        stats = None, 
        logger = None, 
        path: str = 'checkpoint.pt.tar', 
        rank: int = None,
        **kwargs
        ):
    
    if rank is None or rank == 0:
        path = ensure_extension(path, '.pt.tar')
        
        checkpoint = {'model_state_dict' : model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
                      'optimiser_state_dict' : None if not optimiser else optimiser.state_dict(),
                      'scalar_state_dict' : None if not scalar else scalar.state_dict(),
                      'schedular_state_dict' : None if not schedular else schedular.state_dict(),
                      'schedular' : schedular,
                      'stats': stats,
                      'log' : None if not logger else logger.get_log()}
        
        if kwargs:
            checkpoint = {**checkpoint, **kwargs}
        T.save(checkpoint, path)
        
        
def load_checkpoint(
        path, 
        model, 
        optimiser = None, 
        scalar = None, 
        schedular = None, 
        logger = None,
        distributed_to_local: bool = False
        ):
    path = ensure_extension(path, '.pt.tar')
    checkpoint = T.load(path, map_location = 'cpu')
    
    # load model weights
    model_state_dict = checkpoint['model_state_dict'] if not distributed_to_local else convert_distributed_to_local(checkpoint['model_state_dict'])
    model.load_state_dict(model_state_dict)
    model.eval()
    
    if optimiser:
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    if scalar:
        scalar.load_state_dict(checkpoint['scalar_state_dict'])
    if schedular:
        schedular.load_state_dict(checkpoint['schedular_state_dict'])
    if logger:
        logger.load_log(checkpoint['log'])
    elif checkpoint['schedular']:
        schedular = checkpoint['schedular']
        schedular.load_state_dict(checkpoint['schedular_state_dict'])
    
    return model, optimiser, scalar, schedular, logger
        