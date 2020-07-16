while :
do
    python -m june60.plot_result --log-path ~/tmp/BreakoutDeterministic-v4/4/logs/1.json \
        --width 10 --height 5

    scp ~/tmp/BreakoutDeterministic-v4/4/logs/*.pdf vultr:~/image/breakout
    sleep 10m
done
