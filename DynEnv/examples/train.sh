for seed in 42 # 658 94846 341 35710 4380 5646 83465 97 4162
do
    NOTE="baseline with seed=${seed}"
    ARGS=""

    for arg in "$@"
    do
        if [ "$arg" == "--partial" ]
        then
            ARGS+=" --observationType ObservationType.PARTIAL"
            NOTE+=" using partial observations"
        elif [ "$arg" == "--full" ]
        then
            ARGS+=" --observationType ObservationType.FULL"
            NOTE+=" using full observations"
        fi
        
        if [ "$arg" == "--reconstruction" ]
        then
            ARGS+=" --use-reconstruction True --recon-pretrained True"
            NOTE+=" using reconstruction"
        fi
        
        
        if [ "$arg" == "--longterm" ]
        then
            ARGS+=" --long-horizon-coeff 1e-3"
            NOTE+=" using long-term prediction"
        else
            ARGS+=" --long-horizon-coeff 0.0"
        fi
    done
    echo "$NOTE"
    echo "$ARGS"
    python ./main.py --seed $seed --note $NOTE $ARGS &
done