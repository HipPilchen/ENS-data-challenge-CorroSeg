# Experiments 

# all tests on Unet
dict_exp=   {
    # Focal test on gamma
    1:{"experiment_name": "1",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 0,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},  
    2:{"experiment_name": "2",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 0.5,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    3:{"experiment_name": "3",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    4:{"experiment_name": "4",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 2,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    5:{"experiment_name": "5",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 5,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # Focal test on alpha
    6:{"experiment_name": "6",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    7:{"experiment_name": "7",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 10,"pretrained": False,"dropout": False, "p_dropout": None},
    8:{"experiment_name": "8",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 5,"pretrained": False,"dropout": False, "p_dropout": None},
    9:{"experiment_name": "9",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 1,"pretrained": False,"dropout": False, "p_dropout": None},
    # Focal test with dropout
    10:{"experiment_name": "10",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": None},
    # Focal test with pretrained backbone
    11:{"experiment_name": "11",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
    # Focal test with dropout and pretrained backbone
    12:{"experiment_name": "12",'num_epochs':200, "criterion": "focal","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": None},
    # BCE test without dropout
    13:{"experiment_name": "13",'num_epochs':200, "criterion": "bce","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # BCE test with dropout
    14:{"experiment_name": "14",'num_epochs':200, "criterion": "bce","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": None},
    # BCE test with pretrained backbone
    15:{"experiment_name": "15",'num_epochs':200, "criterion": "bce","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
    # BCE test with dropout and pretrained backbone
    16:{"experiment_name": "16",'num_epochs':200, "criterion": "bce","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": None},
    # IOU test on learning rate
    17:{"experiment_name": "17",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 1e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    18:{"experiment_name": "18",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    19:{"experiment_name": "19",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 1e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    20:{"experiment_name": "20",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-4,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # IOU test on batch size
    21:{"experiment_name": "21",'num_epochs':200, "criterion": "iou","batch_size": 32,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    22:{"experiment_name": "22",'num_epochs':200, "criterion": "iou","batch_size": 64,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    23:{"experiment_name": "23",'num_epochs':200, "criterion": "iou","batch_size": 128,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    24:{"experiment_name": "24",'num_epochs':200, "criterion": "iou","batch_size": 180,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None},
    # IOU test on dropout
    25:{"experiment_name": "25",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": None},
    26:{"experiment_name": "26",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": True, "p_dropout": 0.2},
    # IOU test on pretrained backbone
    27:{"experiment_name": "27",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
    28:{"experiment_name": "28",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": None},
    29:{"experiment_name": "29",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": True, "p_dropout": 0.2},
    # IOU test on the number of transforms 
    30:{"experiment_name": "30",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 1},
    31:{"experiment_name": "31",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 2},
    32:{"experiment_name": "32",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 3},
    33:{"experiment_name": "33",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": False,"dropout": False, "p_dropout": None, "n_transforms": 4},
    # IOU test on the number of transforms with pretrained
    34:{"experiment_name": "34",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 1},
    35:{"experiment_name": "35",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 2},
    36:{"experiment_name": "36",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.01,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None, "n_transforms": 3},
    # IOU test on weight decay with pretrained
    37:{"experiment_name": "37",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.001,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
    38:{"experiment_name": "38",'num_epochs':200, "criterion": "iou","batch_size": 160,"model_name": "unet","learning_rate": 5e-5,"weight_decay": 0.1,"gamma": 1,"alpha": 15,"pretrained": True,"dropout": False, "p_dropout": None},
}