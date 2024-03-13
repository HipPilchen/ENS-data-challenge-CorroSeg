# Experiments 

# all tests on Unet
dict_exp=   {
    # Focal test on gamma # not conclusive
    1:{"experiment_name": "1",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 0,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},  
    2:{"experiment_name": "2",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 0.5,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    3:{"experiment_name": "3",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    4:{"experiment_name": "4",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 2,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    5:{"experiment_name": "5",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 5,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # Focal test on alpha, alpha = 1 is best to explore for smaller alphas
    6:{"experiment_name": "6",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    7:{"experiment_name": "7",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 10,"pretrained": False,"dropout": False, "p_dropout": None},
    8:{"experiment_name": "8",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 5,"pretrained": False,"dropout": False, "p_dropout": None},
    9:{"experiment_name": "9",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 1,"pretrained": False,"dropout": False, "p_dropout": None},
    # Focal test with dropout, no great diff
    10:{"experiment_name": "10",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": None},
    # Focal test with pretrained backbone, better perf on the training 
    11:{"experiment_name": "11",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
    # Focal test with dropout and pretrained backbone, even better but not on validation
    12:{"experiment_name": "12",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": None},
    # BCE test without dropout, close to with
    13:{"experiment_name": "13",'num_epochs':200, "criterion": "bce","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # BCE test with dropout, close to without dropout
    14:{"experiment_name": "14",'num_epochs':200, "criterion": "bce","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": None},
    # BCE test with pretrained backbone, does not work well
    15:{"experiment_name": "15",'num_epochs':200, "criterion": "bce","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
    # BCE test with dropout and pretrained backbone, does not work well
    16:{"experiment_name": "16",'num_epochs':200, "criterion": "bce","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": None},
    # IOU test on learning rate, 1e-4 is best (then 5e-5)
    17:{"experiment_name": "17",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 1e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    18:{"experiment_name": "18",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    19:{"experiment_name": "19",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    20:{"experiment_name": "20",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # IOU test on batch size #À refaire avec la best learning rate et dropout, 32 seems best but they are not ordered
    21:{"experiment_name": "21",'num_epochs':200, "criterion": "iou","batch_size": 32,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    22:{"experiment_name": "22",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    23:{"experiment_name": "23",'num_epochs':200, "criterion": "iou","batch_size": 128,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    24:{"experiment_name": "24",'num_epochs':200, "criterion": "iou","batch_size": 180,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # IOU test on dropout, no big diff just learn slowly
    25:{"experiment_name": "25",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": None},
    26:{"experiment_name": "26",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.2},
    # IOU test on pretrained backbone, best is mid dropout, without dropout plateau really high, to mention
    27:{"experiment_name": "27",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
    28:{"experiment_name": "28",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": None},
    29:{"experiment_name": "29",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": 0.2},
    # IOU test on the number of transforms, more overfitting but best valid score for the 3 first transforms 
    30:{"experiment_name": "30",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 1},
    31:{"experiment_name": "31",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 2},
    32:{"experiment_name": "32",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    33:{"experiment_name": "33",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 4},
    # IOU test on the number of transforms with pretrained, best with 3 transforms to avoid a high plateau
    34:{"experiment_name": "34",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 1},
    35:{"experiment_name": "35",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 2},
    36:{"experiment_name": "36",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    # IOU test on weight decay with pretrained, weight decay of 1e-3 is game changer
    37:{"experiment_name": "37",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
    38:{"experiment_name": "38",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.1,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
}

new_experiments = {
    # Focal for small alphas
    39:{"experiment_name": "39",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 1,"pretrained": False,"dropout": False, "p_dropout": None},
    40:{"experiment_name": "40",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 0.1,"pretrained": False,"dropout": False, "p_dropout": None},
    41:{"experiment_name": "41",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 0.01,"pretrained": False,"dropout": False, "p_dropout": None},
    # IoU test on batchsize for optimal learning rate, with 3 transforms not pretrained no dropout
    42:{"experiment_name": "42",'num_epochs':200, "criterion": "iou","batch_size": 32,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    43:{"experiment_name": "43",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    44:{"experiment_name": "44",'num_epochs':200, "criterion": "iou","batch_size": 128,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    45:{"experiment_name": "45",'num_epochs':200, "criterion": "iou","batch_size": 180,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    46:{"experiment_name": "46",'num_epochs':200, "criterion": "iou","batch_size": 200,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    # IoU test with pretrained and optimal weight decay
    47:{"experiment_name": "47",'num_epochs':200, "criterion": "iou","batch_size": 32,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    48:{"experiment_name": "48",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    49:{"experiment_name": "49",'num_epochs':200, "criterion": "iou","batch_size": 128,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    50:{"experiment_name": "50",'num_epochs':200, "criterion": "iou","batch_size": 180,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    51:{"experiment_name": "51",'num_epochs':200, "criterion": "iou","batch_size": 200,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    # IoU test with pretrained on balance weight decay, dropout for 4 transforms
    52:{"experiment_name": "52",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": None, "n_transforms": 4},
    53:{"experiment_name": "53",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": 0.2, "n_transforms": 4},
    # IoU test models on same params 
    54:{"experiment_name": "54",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "jacard_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.1, "n_transforms": 4},
    55:{"experiment_name": "55",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "baseline_cnn","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.1, "n_transforms": 4},
    56:{"experiment_name": "56",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "seg_model","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.1, "n_transforms": 4},
    57:{'experiment_name': '57', 'num_epochs': 200, 'criterion': 'iou', 'batch_size': 64, 'model_name': 'unet', 'learning_rate': 1e-4, 'weight_decay': 0.001, 'gamma': 1, 'alpha': 15, 'pretrained': False, 'dropout': True, 'p_dropout': 0.1, 'n_transforms': 4},
    # Cuda out of memory
    58:{'experiment_name': '58', 'num_epochs': 200, 'criterion': 'iou', 'batch_size': 64, 'model_name': 'cnn', 'learning_rate': 1e-4, 'weight_decay': 0.001, 'gamma': 1, 'alpha': 15, 'pretrained': False, 'dropout': True, 'p_dropout': 0.1, 'n_transforms': 4},
    59:{'experiment_name': '59', 'num_epo chs': 200, 'criterion': 'iou', 'batch_size': 64, 'model_name': 'first_model', 'learning_rate': 1e-4, 'weight_decay': 0.001, 'gamma': 1, 'alpha': 15, 'pretrained': False, 'dropout': True, 'p_dropout': 0.1, 'n_transforms': 4},
}

# Experiments on Unet with batchnorm and different skip connection
# Add to REPORT
newest_exp_0 = {
    # Add Unet with Batch Normalization same number as above # moins bien en valid mais pareil en train
    # 17:{"experiment_name": "batchnorm_17",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 1e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # 18:{"experiment_name": "batchnorm_18",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # 19:{"experiment_name": "batchnorm_19",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # 20:{"experiment_name": "batchnorm_20",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # Pas de différence significative quand pretrained 
    # 47:{"experiment_name": "batchnorm_47",'num_epochs':200, "criterion": "iou","batch_size": 32,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    # 48:{"experiment_name": "batchnorm_48",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    # 49:{"experiment_name": "batchnorm_49",'num_epochs':200, "criterion": "iou","batch_size": 128,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    # Aucune diff pour les 2
    42 :{"experiment_name": "batchnorm_42",'num_epochs':200, "criterion": "iou","batch_size": 32,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    43 :{"experiment_name": "batchnorm_43",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    44 :{"experiment_name": "batchnorm_44",'num_epochs':200, "criterion": "iou","batch_size": 128,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    # New experiments with concat skip connections
    # Classis à comparer avec 43, mieux que pas pretrained sur la valid et sur 
    60:{"experiment_name": "60",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 3},
    # To do again
    61:{"experiment_name": "61",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.1, "n_transforms": 3},
    # pretrained # Mieux aussi 
    61:{"experiment_name": "61",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": 0.1, "n_transforms": 3},
    62:{"experiment_name": "62",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": 0.1, "n_transforms": 3},
    63:{"experiment_name": "63",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": 0.2, "n_transforms": 3},
    # Batch size # Impact the speed of convergence but not the final result for both scores. 
    64:{"experiment_name": "62",'num_epochs':200, "criterion": "iou","batch_size": 32,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 4},
    65:{"experiment_name": "65",'num_epochs':200, "criterion": "iou","batch_size": 128,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 4},
    66:{"experiment_name": "66",'num_epochs':200, "criterion": "iou","batch_size": 180,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 4},
    
}


newest_exp = {
    # Num of transforms with concat, not pretrained (mieux dans le cas concat)
    67:{"experiment_name": "67",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 3},
    68:{"experiment_name": "68",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 4},
    69:{"experiment_name": "69",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 5},
    # Num of transforms with concat, pretrained
    # 70:{"experiment_name": "70",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": 0.1, "n_transforms": 3},
    # 71:{"experiment_name": "71",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": 0.1, "n_transforms": 4},
    # 72:{"experiment_name": "72",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": 0.1, "n_transforms": 5},
    #  Concat, pretrained, dropout # Trop dropout nuit à la valid
    # 73:{"experiment_name": "73",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": 0.1, "n_transforms": 4},
    # 74:{"experiment_name": "74",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": 0.2, "n_transforms": 4},
    # Concat without batchnorm  # Moins bien qu'avec batchnorm
    75:{"experiment_name": "75",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 4, "batchnorm": False},
    # 76:{"experiment_name": "76",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": 0.1, "n_transforms": 4, "batchnorm": False},
    # Concat, not pretrained, dropout
    77:{"experiment_name": "77",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.1, "n_transforms": 4},
    78:{"experiment_name": "78",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.2, "n_transforms": 4},
    
}
# Tester les bests avec 4 transforms 
to_submit = {
    73:{"experiment_name": "to_submit_drop_npre",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.1, "n_transforms": 4,'valid_ratio':0.01},
    74:{'experiment_name': 'to_submit_no_drop_npre', 'num_epochs': 200, 'criterion': 'iou', 'batch_size': 64, 'model_name': 'cat_unet', 'learning_rate': 1e-4, 'weight_decay': 0.001, 'gamma': 1, 'alpha': 15, 'pretrained': False, 'dropout': False, 'p_dropout': 0.1, 'n_transforms': 4,'valid_ratio':0.01},
    75:{"experiment_name": "to_submit_no_drop_less_transfo_npre",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": 0.1, "n_transforms": 3,'valid_ratio':0.01},
    76:{"experiment_name": "to_submit_drop_less_transfo_npre",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "cat_unet","learning_rate": 1e-4,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.1, "n_transforms": 3,'valid_ratio':0.01},
    77:{'experiment_name': 'to_submit_best_score1', 'num_epochs': 200, 'criterion': 'iou', 'batch_size': 32, 'model_name': 'cat_unet', 'learning_rate': 5e-5, 'weight_decay': 0.001, 'gamma': 1, 'alpha': 15, 'pretrained': True, 'dropout': False, 'p_dropout': 0.1,'valid_ratio':0.01},
    78:{'experiment_name': 'to_submit_best_score2', 'num_epochs': 200, 'criterion': 'iou', 'batch_size': 64, 'model_name': 'cat_unet', 'learning_rate': 5e-5, 'weight_decay': 0.001, 'gamma': 1, 'alpha': 15, 'pretrained': True, 'dropout': False, 'p_dropout': 0.1,'valid_ratio':0.01},
    79:{'experiment_name': 'to_submit_best_score3', 'num_epochs': 200, 'criterion': 'iou', 'batch_size': 128, 'model_name': 'cat_unet', 'learning_rate': 5e-5, 'weight_decay': 0.001, 'gamma': 1, 'alpha': 15, 'pretrained': True, 'dropout': False, 'p_dropout': 0.1,'valid_ratio':0.01},
    80:{'experiment_name': 'to_submit_best_score4', 'num_epochs': 200, 'criterion': 'iou', 'batch_size': 180, 'model_name': 'cat_unet', 'learning_rate': 5e-5, 'weight_decay': 0.001, 'gamma': 1, 'alpha': 15, 'pretrained': True, 'dropout': False, 'p_dropout': 0.1,'valid_ratio':0.01},
}
