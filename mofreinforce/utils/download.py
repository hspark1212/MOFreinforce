import os
import wget
import tarfile
from pathlib import Path
from mofreinforce import __root_dir__


DEFAULT_PATH = Path(__root_dir__)

class DownloadError(Exception):
    pass


def _remove_tmp_file(direc:Path):
    tmp_list = direc.parent.glob('*.tmp')
    for tmp in tmp_list:
        if tmp.exists():
            os.remove(tmp)


def _download_file(link, direc, name='target'):
    if direc.exists():
        print (f'{name} already exists.')
        return
    try:
        print(f'\n====Download {name} =============================================\n')
        filename = wget.download(link, out=str(direc))
    except KeyboardInterrupt:
        _remove_tmp_file(direc)
        raise
    except Exception as e:
        _remove_tmp_file(direc)
        raise DownloadError(e)
    else:
        print (f'\n====Successfully download : {filename}=======================================\n')


def download_default(direc=None, remove_tarfile=False):
    """
    downlaod data and pre-trained models including a generator, predictors for DAC
    """
    if not direc:
        direc = Path(DEFAULT_PATH)
        if not direc.exists():
            direc.mkdir(parents=True, exist_ok=True)
        direc = direc/'default.tar.gz'
    else:
        direc = Path(direc)
        if direc.is_dir():
            if not direc.exists():
                direc.mkdir(parents=True, exist_ok=True)
            direc = direc / 'default.tar.gz'
        else:
            raise ValueError(f'direc must be path for directory, not {direc}')

    link = 'https://figshare.com/ndownloader/files/39472138'
    name = 'basic data and pretrained models'
    _download_file(link, direc, name)

    print(f'\n====Unzip : {name}===============================================\n')
    with tarfile.open(direc) as f:
        f.extractall(path=direc.parent)

    print(f'\n====Unzip successfully: {name}===============================================\n')

    if remove_tarfile:
        os.remove(direc)
