"""
Just to Experiment with System Commans to interface Python with SPTK.

"""

from os import system as sys

raw = "wav2raw +f a.wav"
dump1 = "x2x +fa <a.raw >a.txt"
mcep = "x2x +ff <a.raw | frame -l 256 -p 64 | window -l 256 | mcep -a 0.42 -m 24 -l 256 >a.mcep"  # todo: -w, -m
pitch = "x2x +ff <a.raw | pitch -p 64 >a.pitch"
mlsa = "excite -p 64 -n a.pitch | mlsadf -m 24 -a 0.42 -p 64 a.mcep >out.raw"  # todo: -m
# mlsa = "x2x +fs <a.pitch | excite -p 64 -n | mlsadf -m 24 -a 0.42 -p 64 a.mcep >out.raw"
dump2 = "x2x +fa <out.raw >out.txt"
op = "raw2wav -s 16 +f +f out.raw"

# raw = "wav2raw +f a.wav"
# mcep = "x2x +ff <a.raw | frame -l 256 -p 64 | window -l 256 | gcep  -m 20 -l 256 >a.mcep"  # todo: -w, -m
# pitch = "x2x +ff <a.raw | pitch -p 64 >a.pitch"
# mlsa = "excite -p 64 a.pitch | mlsadf -m 20 -a 0.42 -p 64 a.mcep >a.raw"  # todo: -m
# op = "raw2wav -s 16 +f +f a.raw"

sys(raw)
sys(dump1)
sys(mcep)
sys(pitch)
sys(mlsa)
sys(dump2)
sys(op)
