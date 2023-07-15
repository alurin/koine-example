# Copyright (C) 2021 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
import os

# Compiler name. This value is used in packages/plugins/cli and e.t.c
compiler_name = "koine"

# Compiler version
version = "0.0.0"

# Where is stored stdlib?
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
