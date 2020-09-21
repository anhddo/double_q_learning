python -m rl.plot_result --save-dir tmp\CartPole-v1\1 tmp\CartPole-v1\2 tmp\CartPole-v1\3  tmp\CartPole-v1\5 `
    --label "No exploration" "epsilon-greedy" "Optimistic, beta=5" "Optimistic buffer 150k" `
    --avg --eval --title CartPole-v1 --plot-name cartpole-ddqn.png
