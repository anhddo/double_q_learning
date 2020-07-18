#xvfb-run -s "0 1400x900x24" 
DIRPATH=${HOME}/tmp/BreakoutDeterministic-v4/8
python -m june60.deep_rl.run_atari --record \
    --record-path ${DIRPATH}/videos \
    --load-model-path ${DIRPATH}/model/80.ckpt \
    --env BreakoutDeterministic-v4

scp ${DIRPATH}/videos/*.mp4 vultr:~/image/breakout
