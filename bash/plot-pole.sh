python -m june60.plot_result --save-dir ~/tmp/CartPole-v1/1 ~/tmp/CartPole-v1/2 \
    --label dqn ddqn --plot-name compare.pdf --avg
ssh vultr "mkdir -p ~/server/CartPole-v1"
scp compare.pdf vultr:~/server/CartPole-v1
