set -x
set -e

LOG="experiments/logs/detection/e2e_DeepPest.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/train_net.py --cfg experiments/cfgs/e2e_DeepPest.yaml