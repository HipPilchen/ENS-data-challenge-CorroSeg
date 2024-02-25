import numpy as np
from tqdm import tqdm

from segnet.keras_segmentation.data_utils.data_loader import get_pairs_from_paths, get_segmentation_array


def evaluate(inp_images_pred, annotations, n_classes):

    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    n_pixels = np.zeros(n_classes)

    pr = inp_images_pred

    width, height = pr.shape # c'est bien le bon sens ?

    gt = get_segmentation_array(
        annotations, n_classes, width, height, 
        no_reshape=True, 
        #read_image_type=read_image_type,
        )

    gt = gt.argmax(-1)
    pr = pr.flatten()
    gt = gt.flatten()

    for cl_i in range(n_classes):

        tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
        fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
        fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
        n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score
    }



# Inspired from segnet.keras_segmentation.predict.evaluate
def evaluate_dir(inp_images_dir, annotations_dir, n_classes):

    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    n_pixels = np.zeros(n_classes)

    paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
    paths = list(zip(*paths))
    inp_images = list(paths[0])
    annotations = list(paths[1])

    for inp, ann in tqdm(zip(inp_images, annotations)):
        # pr = predict(model, inp, read_image_type=read_image_type)
        # gt = get_segmentation_array(ann, model.n_classes,
        #                             model.output_width, model.output_height,
        #                             no_reshape=True, read_image_type=read_image_type)

        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(n_classes):

            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score
    }