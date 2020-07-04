python -m june60.deep_rl.atari --tmp-dir tmp --buffer 250000 --step 2000000 \
    --start 1000 --fraction 0.1 --max-epsilon 1 --min-epsilon 0.1 \
    --train-step 2 --update 10000 --env Pong-v0 --batch 32 --ddqn \
    --vi --mse --rms --lr 0.000625 --tau 0.001 --discount 0.99 --tboard
