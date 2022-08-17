from copy import deepcopy

from vault.models.tombert import TomBertDatasetForTMSC


class TomViltDatasetFotTMSC(TomBertDatasetForTMSC):
    """TomVilt dataset, for now same as Tombert"""

    argparse_args = deepcopy(TomBertDatasetForTMSC.argparse_args)
    argparse_args["max_total_length"]["default"] = 40
