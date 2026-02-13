for i in `seq 1 15`; do
    echo "$i"
    # skip 5, 8, 9
    if [ "$i" -eq 5 ] || [ "$i" -eq 8 ] || [ "$i" -eq 9 ]; then
        continue
    fi
    python3 main_experiment.py --config experiment_configs/exp_${i}.json

done