python -m rl.plot_result --save-dir tmp\Acrobot-v1\3 tmp\Acrobot-v1\5 tmp\Acrobot-v1\6 `
    --label ovi-b10 std-vi egreedy `
    --avg --plot-name bot.png

python -m rl.plot_result --save-dir tmp\CartPole-v1\1  tmp\CartPole-v1\5 tmp\CartPole-v1\6 `
    --label ovi-b1 std-vi egreedy `
    --avg --plot-name cart-pole.png

python -m rl.plot_result --save-dir tmp\MountainCar-v0\1 tmp\MountainCar-v0\5 tmp\MountainCar-v0\6 `
    --label ovi-b1 std-vi egreedy `
    --avg --plot-name car.png
##############
#python -m rl.plot_result --save-dir tmp\Acrobot-v1\1 tmp\Acrobot-v1\2 tmp\Acrobot-v1\3 tmp\Acrobot-v1\4 tmp\Acrobot-v1\5 tmp\Acrobot-v1\6 `
#    --label ovi-b1 ovi-b5 ovi-b10 ovi-b15 std-vi egreedy `
#    --avg --plot-name bot.png
#
#python -m rl.plot_result --save-dir tmp\CartPole-v1\1 tmp\CartPole-v1\2 tmp\CartPole-v1\3 tmp\CartPole-v1\4 tmp\CartPole-v1\5 tmp\CartPole-v1\6 `
#    --label ovi-b1 ovi-b5 ovi-b10 ovi-b15 std-vi egreedy `
#    --avg --plot-name cart-pole.png
#
#python -m rl.plot_result --save-dir tmp\MountainCar-v0\1 tmp\MountainCar-v0\2 tmp\MountainCar-v0\3 tmp\MountainCar-v0\4 tmp\MountainCar-v0\5 tmp\MountainCar-v0\6 `
#    --label ovi-b1 ovi-b5 ovi-b10 ovi-b15 std-vi egreedy `
#    --avg --plot-name car.png
