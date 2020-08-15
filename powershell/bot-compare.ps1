#python -m june60.plot_result --save-dir tmp\Acrobot-v1\1 tmp\Acrobot-v1\2 tmp\Acrobot-v1\3 tmp\Acrobot-v1\4 tmp\Acrobot-v1\6 `
#    tmp\Acrobot-v1\10 tmp\Acrobot-v1\11 `
#    --label greedy  optimistic5 opt-1 opt-1-buf-5000 opt-new opt-new-beta-0.5 greedy-new `
#    --avg --plot-name bot-compare.png


#python -m june60.plot_result --save-dir tmp\Acrobot-v1\13 tmp\Acrobot-v1\14 `
#    --label no-skip opt-new-beta-0.5  `
#    --avg --plot-name bot-compare.png
#

#python -m june60.plot_result --save-dir tmp\Acrobot-v1\1 tmp\Acrobot-v1\2 tmp\Acrobot-v1\3 tmp\Acrobot-v1\4 tmp\Acrobot-v1\5 `
#    tmp\Acrobot-v1\6 tmp\Acrobot-v1\7 tmp\Acrobot-v1\8 tmp\Acrobot-v1\9 `
#    --label 1 2 3 4 5 6 7 8 9 `
#    --avg --plot-name bot-compare.png

python -m june60.plot_result --save-dir tmp\Acrobot-v1\10 tmp\Acrobot-v1\11 tmp\Acrobot-v1\12 `
    --label 10-ovi 11-greedy 12-vi `
    --avg --plot-name bot-compare2.png
.\bot-compare2.png
