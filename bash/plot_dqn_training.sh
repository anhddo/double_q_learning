while :
do
    python -m june60.plot_result --log-path ~/tmp/BreakoutDeterministic-v4/2/logs/1.json 
    scp ~/tmp/BreakoutDeterministic-v4/2/logs/*.pdf vultr:~/image/breakout
    sleep 1h
done
