# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import sys

import prettyprinter

from koine.exceptions import KoineError, fetch_error_string
from koine.semantic import fetch_semantic_model


def main():
    prettyprinter.install_extras(include=('attrs',))
    prettyprinter.set_default_config(width=180, ribbon_width=180)

    if len(sys.argv) < 2:
        print('File is not specified', file=sys.stderr)
        exit(2)

    try:
        model = fetch_semantic_model(sys.argv[1])
    except KoineError as ex:
        print(fetch_error_string(ex), file=sys.stderr)
        exit(1)
    # else:
    #     breakpoint()


if __name__ == '__main__':
    main()
