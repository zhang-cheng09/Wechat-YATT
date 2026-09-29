RCDIR=".."
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$PYTHONPATH"
export PYTHONPATH="$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

pkill -9 -f py
python tests/test_perf/test_linear_ce_kernels.py
