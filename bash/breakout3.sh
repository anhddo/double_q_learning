python -m june60.deep_rl.run_atari --tmp-dir ~/tmp --buffer 1000000 --step 100000000 \
    --start 1000 --fraction 0.1 --max-epsilon 0.1 --min-epsilon 0.1 \
    --train-step 1 --update 100 --env BreakoutDeterministic-v4 --batch 32 --ddqn \
    --vi --huber --rms --lr 0.00025 --tau 0.001 --discount 0.99 \
    --load-model-path ~/tmp/BreakoutDeterministic-v4/8/model/82.ckpt
