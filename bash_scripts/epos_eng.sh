#! /bin/bash

for mv in can could may must should
do
    for b in balance unbalance
    do
        for c in classifier1 classifier2
        do
            python main.py --corpus ${mv}_${b}_${c}
        done
    done
done

for mv in can could may must should
do
    python main.py --corpus ${mv}_balance_classifier3
done

for g in blog court-transcript debate-transcript email essays face-to-face ficlets fiction govt-docs jokes journal letters movie-script newspaper non-fiction technical telephone travel-guides twitter
do
    for mv in can could may must should
    do
        python main.py -mode eval --corpus ${mv}_balance_classifier2 --test_corpus ${g}__${mv}
    done
done

