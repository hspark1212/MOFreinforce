class CLICommand:
    """
    Download data and pre-trained model including a generator, predictors for DAC
    """

    @staticmethod
    def add_arguments(parser):
        from mofreinforce.utils.download import DEFAULT_PATH
        add = parser.add_argument
        add ('target', nargs='+', help='download data and pretrained models including a generator and predictors for DAC')
        add ('--outdir', '-o', help=f'The Path where downloaded data will be stored. \n'
                                    f'default : (default) {DEFAULT_PATH} \n')
        add ('--remove_tarfile', '-r', action='store_true', help='remove tar.gz file for download database')

    @staticmethod
    def run(args):
        from mofreinforce.utils.download import (
            download_default,
        )

        func_dic = {'default': download_default,
                    }

        for stuff in args.target:
            if stuff not in func_dic.keys():
                raise ValueError(f'target must be {", ".join(func_dic.keys())}, not {stuff}')

        for stuff in args.target:
            func = func_dic[stuff]
            if func.__code__.co_argcount == 1:
                func(args.outdir)
            else:
                func(args.outdir, args.remove_tarfile)