for lambda in 1.0 0.46415888 0.21544347 0.1 0.04641589 0.02154435 0.01 0.00464159 0.00215443 0.001
do
   echo "Lambda $lambda training"
   python train.py --load_perplexity_reg --embedding_dim 256 --batch_size 64 --lr 0.0001 --modeldir /home/czestoch/workspace/emb2emb/tmp/sentence-summarization_lstm/lstmae0.0p010 --data_fraction 1.0 --n_epochs 10 --n_layers 1 --print_outputs --dataset_path /home/czestoch/workspace/emb2emb/data/gigaword --mapping offsetnet --hidden_layer_size 1024 --loss cosine --adversarial_regularization --adversarial_lambda $lambda --outputdir /home/czestoch/workspace/emb2emb/tmp/sentence-summarization --perplexity_regressor_path /home/czestoch/workspace/emb2emb/tmp/sentence-summarization/model1638459574phi_perplexity_regressor.pickle --output_file emnlp_gigaword_offset_cosine.csv --real_data_path input --max_prints 10 --unaligned --validation_frequency -1 --validate
done
