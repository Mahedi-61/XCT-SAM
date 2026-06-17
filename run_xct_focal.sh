CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_2.py --dataset_dir datasets/gan-generated --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --output_dir  output/gan_adapt_focal/test1_c2 --data_name  test1_class2 --num_gpus 1 --batch_size 4 --rank 2 --eval

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_2.py --dataset_dir datasets/gan-generated --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --output_dir  output/gan_adapt_focal/test2_c2 --data_name  test2_class2 --num_gpus 1 --batch_size 4 --rank 2 --eval

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_2.py --dataset_dir datasets/gan-generated --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --output_dir  output/gan_adapt_focal/test4_c2 --data_name  test4_class2 --num_gpus 1 --batch_size 4 --rank 2 --eval

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_2.py --dataset_dir datasets/gan-generated --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --output_dir  output/gan_adapt_focal/test5_c2 --data_name  test5_class2 --num_gpus 1 --batch_size 4 --rank 2 --eval --threshold 0.50 --morph none --morph_size 1 --min_area 5 --edge_margin 3

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_2.py --dataset_dir datasets/gan-generated --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --output_dir  output/gan_adapt_focal/test6_c2 --data_name test6_class2 --num_gpus 1 --batch_size 4 --rank 2 --eval

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_3.py --dataset_dir datasets/gan-generated --ckpt_path AutogluonModels/alloy_to_xct_focal/class_3 --output_dir  output/gan_adapt_focal/test6_c3 --data_name test6_class3 --num_gpus 1 --batch_size 4 --rank 2 --eval 

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_3.py --dataset_dir datasets/gan-generated --ckpt_path AutogluonModels/alloy_to_xct_focal/class_3 --output_dir  output/gan_adapt_focal/test2_c3 --data_name test2_class3 --num_gpus 1 --batch_size 4 --rank 2 --eval

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_3.py --dataset_dir datasets/gan-generated --ckpt_path AutogluonModels/alloy_to_xct_focal/class_3 --output_dir  output/gan_adapt_focal/test1_c3 --data_name test1_class3 --num_gpus 1 --batch_size 4 --rank 2 --eval

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt_alloy_class_2.py --dataset_dir ./datasets --output_dir ./output/israt_adapt_focal/test2_c2 --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --data_name test2_class1 --batch_size 4 --threshold 0.40 --morph none --morph_size 0 --min_area 0

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt_alloy_class_2.py --dataset_dir ./datasets --output_dir ./output/israt_adapt_focal/test3_c2 --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --data_name test3_class1 --batch_size 4 --threshold 0.40 --morph close --morph_size 11 --min_area 0

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt_alloy_class_2.py --dataset_dir ./datasets --output_dir ./output/israt_adapt_focal/test4_c2 --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --data_name test4_class1 --batch_size 4 --threshold 0.70 --morph open --morph_size 5 --min_area 75

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt_alloy_class_2.py --dataset_dir ./datasets --output_dir ./output/israt_adapt_focal/test5_c2 --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --data_name test5_class1 --batch_size 4 --threshold 0.50 --morph dilate --morph_size 3 --min_area 0

CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt_alloy_class_2.py --dataset_dir ./datasets --output_dir ./output/israt_adapt_focal/test6_c2 --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 --data_name test6_class1 --batch_size 4 --threshold 0.50 --morph none --min_area 0 --invert_output
