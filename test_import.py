#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
try:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    print("PI0Pytorch import OK")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
