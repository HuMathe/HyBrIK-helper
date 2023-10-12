import os

def load_retrieve_funcs(fmt_dir, name: str):

    fmt_mod = '.'.join([c for c in os.path.split(fmt_dir) if c != ''])

    import importlib

    assert os.path.exists(os.path.join(fmt_dir, name + '.py')), \
        f'{name} module does not exist.'
    data_funcs = importlib.import_module('.', f'{fmt_mod}.{name}')
    
    assert hasattr(data_funcs, 'GetSubList')
    assert hasattr(data_funcs, 'GetCamList')
    assert hasattr(data_funcs, 'GetFrameList')
    assert hasattr(data_funcs, 'GetImagePath')

    return data_funcs.GetSubList, data_funcs.GetCamList, data_funcs.GetFrameList, data_funcs.GetImagePath 

__all__ = ['load_retrieve_funcs']