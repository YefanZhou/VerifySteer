#!/bin/bash
# .so files must already be present (run `python download_so.py` from VerifySteer root first)
SETUPTOOLS_SCM_PRETEND_VERSION="0.1.dev10893+g68e6ee516" \
VLLM_USE_PRECOMPILED=1 pip install --editable .