

for seed in 0 1 2;
do
    cuda_id=`expr $seed + 1`
    CUDA_VISIBLE_DEVICES=$cuda_id python main.py --optimizer sgd --seed $seed &
done
wait