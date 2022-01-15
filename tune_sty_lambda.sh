for lambda in 0.1 0.5 0.9 0.95 0.99
do
   echo "Lambda style: $lambda training"
   python train.py --load_perplexity_reg --embedding_dim 256 --batch_size 64 --lr 0.0001 --modeldir /home/czestoch/workspace/emb2emb/tmp/sentence-summarization_lstm/lstmae0.0p010 --data_fraction 1.0 --n_epochs 10 --n_layers 1 --print_outputs --dataset_path /home/czestoch/workspace/emb2emb/data/gigaword --mapping offsetnet --hidden_layer_size 1024 --loss summaryloss --lambda_regloss $lambda --adversarial_regularization --adversarial_lambda 0.004642 --outputdir /home/czestoch/workspace/emb2emb/tmp/sentence-summarization --perplexity_regressor_path /home/czestoch/workspace/emb2emb/tmp/sentence-summarization/model1638459574phi_perplexity_regressor.pickle --output_file emnlp_gigaword_offset_cosine.csv --real_data_path input --max_prints 10 --unaligned --validation_frequency -1 --device cuda:1 --validate
done
