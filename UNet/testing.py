import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import cv2

from UNet.model import build_unet


def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (256, 256, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (256, 256, 3)
    return mask


def segment(image_lq, image_hq):
    H = 256
    W = 256
    size = (W, H)
    checkpoint_path = "UNet/files/checkpoint.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    image_lq = image_lq[:, :, ::-1]
    image_hq = image_hq[:, :, ::-1]

    x_lq = np.transpose(image_lq, (2, 0, 1))  # (3, 512, 512)
    x_lq = np.expand_dims(x_lq, axis=0)  # (1, 3, 512, 512)
    x_lq = x_lq.astype(np.float32)
    x_lq = torch.from_numpy(x_lq)
    x_lq = x_lq.to(device)

    x_hq = np.transpose(image_hq, (2, 0, 1))  # (3, 512, 512)
    x_hq = np.expand_dims(x_hq, axis=0)  # (1, 3, 512, 512)
    x_hq = x_hq.astype(np.float32)
    x_hq = torch.from_numpy(x_hq)
    x_hq = x_hq.to(device)

    with torch.no_grad():
        pred_y_lq = model(x_lq)
        pred_y_lq = torch.sigmoid(pred_y_lq)

        pred_y_hq = model(x_hq)
        pred_y_hq = torch.sigmoid(pred_y_hq)

        pred_y_lq = pred_y_lq[0].cpu().numpy()  # (1, 512, 512)
        pred_y_lq = np.squeeze(pred_y_lq, axis=0)  # (512, 512)
        pred_y_lq = pred_y_lq > 0.5
        pred_y_lq = np.array(pred_y_lq, dtype=np.uint8)

        pred_y_hq = pred_y_hq[0].cpu().numpy()  # (1, 512, 512)
        pred_y_hq = np.squeeze(pred_y_hq, axis=0)  # (512, 512)
        pred_y_hq = pred_y_hq > 0.5
        pred_y_hq = np.array(pred_y_hq, dtype=np.uint8)

    line = np.ones((size[1], 10, 3)) * (-127.5)
    pred_y_lq = mask_parse(pred_y_lq)
    pred_y_hq = mask_parse(pred_y_hq)

    cat_images = np.vstack((np.hstack((image_lq * 255, line, pred_y_lq * 255)),
                            np.hstack((image_hq * 255, line, pred_y_hq * 255)),
                            np.ones((50, size[1]*2+10, 3)) * (-127.5)))

    return cat_images