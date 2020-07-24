python -m june60.deep_rl.run_atari --tmp-dir ~/tmp --buffer 1000000 \
    --replay-start-frame 5000 --final-exploration-frame 10000 --frame-each-epoch 2500 --validation-frame 1000\
    --update-step 100 --env BreakoutDeterministic-v4 --batch 32 --ddqn \
    --vi --huber --rms --lr 0.00025 --tau 0.001 --discount 0.99 \
