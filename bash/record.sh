echo $1 $2 $3
DIRPATH=${HOME}/tmp/$1/$2
python -m june60.deep_rl.run_atari --record \
    --record-path ${DIRPATH}/videos \
    --load-model-path ${DIRPATH}/model/$3.ckpt \
    --env $1

ssh root@vultr "mkdir -p ~/image/$1"
scp ${DIRPATH}/videos/*.mp4 vultr:~/image/$1
