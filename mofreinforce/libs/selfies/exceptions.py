# This code is adapted from the "selfies" repository:
# https://github.com/aspuru-guzik-group/selfies.git

# The code in this file is licensed under the Apache License, Version 2.0:
# https://github.com/aspuru-guzik-group/selfies/blob/master/LICENSE

# Adapted by Hyunsoo Park, 2022

class SMILESParserError(ValueError):
    """Exception raised when a SMILES fails to be parsed.
    """

    def __init__(self, smiles, reason="N/A", idx=-1):
        self.smiles = smiles
        self.idx = idx
        self.reason = reason

    def __str__(self):
        err_msg = "\n" \
                  "\tSMILES: {smiles}\n" \
                  "\t        {pointer}\n" \
                  "\tIndex:  {index}\n" \
                  "\tReason: {reason}"

        return err_msg.format(
            smiles=self.smiles,
            pointer=(" " * self.idx + "^"),
            index=self.idx,
            reason=self.reason
        )


class EncoderError(Exception):
    """Exception raised by :func:`selfies.encoder`.
    """

    pass


class DecoderError(Exception):
    """Exception raised by :func:`selfies.decoder`.
    """

    pass
