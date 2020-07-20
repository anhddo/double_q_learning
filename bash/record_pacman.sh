#xvfb-run -s "0 1400x900x24" 
DIRPATH=${HOME}/tmp/MsPacmanDeterministic-v4/1
python -m june60.deep_rl.run_atari --record \
    --record-path ${DIRPATH}/videos \
    --load-model-path ${DIRPATH}/model/350.ckpt \
    --env MsPacmanDeterministic-v4

scp ${DIRPATH}/videos/*.mp4 vultr:~/image/pacman
