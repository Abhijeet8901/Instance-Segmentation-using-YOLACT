import torch.nn.functional as F
import cv2
from box_utils import decode, jaccard
from box_utils import crop_ as crop
import torch
import numpy as np
from config import COLORS
import pdb


def fast_nms(box_thre, coef_thre, class_thre, cfg, second_threshold=False):
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.size()
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = jaccard(box_thre, box_thre)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # We should also only keep detections over the confidence threshold, but at the cost of
    # maxing out your detection count for every image, you can just not do that. Because we
    # have such a minimal amount of computation per detection (matrix mulitplication only),
    # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
    # However, when you implement this in your method, you should do this second threshold.
    if second_threshold:
        keep *= (class_thre > cfg.nms_score_thresh)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

    class_ids = class_ids[keep]

    box_nms = box_thre[keep]
    coef_nms = coef_thre[keep]
    class_nms = class_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    class_nms, idx = class_nms.sort(0, descending=True)

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]

    return box_nms, coef_nms, class_ids, class_nms

def nms(class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    class_thre = class_p[:, keep]
    box_thre = decode(box_p[keep, :], anchors[keep, :])
    coef_thre = coef_p[keep, :]

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        box_thre, coef_thre, class_ids, class_thre = fast_nms(box_thre, coef_thre, class_thre, cfg)

        return class_ids, class_thre, box_thre, coef_thre, proto_p


def after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None, img_name=None):
    if ids_p is None:
        return None, None, None, None

    if cfg and cfg.visual_thre > 0:
        keep = class_p >= cfg.visual_thre
        if not keep.any():
            return None, None, None, None

        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    if cfg and cfg.save_lincomb:
        draw_lincomb(proto_p, coef_p, img_name)

    masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t()))

    if not cfg or not cfg.no_crop:  # Crop masks by box_p
        masks = crop(masks, box_p)

    masks = masks.permute(2, 0, 1).contiguous()

    ori_size = max(img_h, img_w)
    masks = F.interpolate(masks.unsqueeze(0), (ori_size, ori_size), mode='bilinear', align_corners=False).squeeze(0)
    masks.gt_(0.5)  # Binarize the masks because of interpolation.
    masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

    box_p *= ori_size
    box_p = box_p.long()

    return ids_p, class_p, box_p, masks


def draw_lincomb(proto_data, masks, img_name):
    for kdx in range(1):
        jdx = kdx + 0
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))

        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4, 8)
        p_h, p_w, _ = proto_data.size()
        arr_img = np.zeros([p_h * arr_h, p_w * arr_w])
        arr_run = np.zeros([p_h * arr_h, p_w * arr_w])

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = (1 / (1 + np.exp(-running_total)))

                arr_img[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (proto_data[:, :, idx[i]] / torch.max(
                    proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (running_total_nonlin > 0.5).astype(np.float)

        arr_img = ((arr_img + 1) * 127.5).astype('uint8')
        arr_img = cv2.applyColorMap(arr_img, cv2.COLORMAP_WINTER)
        cv2.imwrite(f'results/images/lincomb_{img_name}', arr_img)


def draw_img(ids_p, class_p, box_p, mask_p, img_origin, cfg, img_name=None, fps=None):
    if ids_p is None:
        return img_origin

    ids_p = ids_p.cpu().numpy()
    class_p = class_p.cpu().numpy()
    box_p = box_p.cpu().numpy()
    mask_p = mask_p.cpu().numpy()

    num_detected = ids_p.shape[0]

    img_fused = img_origin
    if not cfg.hide_mask:
        masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
        # The color of the overlap area is different because of the '%' operation.
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
        color_masks = COLORS[masks_semantic].astype('uint8')
        img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

        if cfg.cutout:
            for i in range(num_detected):
                one_obj = np.tile(mask_p[i], (3, 1, 1)).transpose((1, 2, 0))
                one_obj = one_obj * img_origin
                new_mask = mask_p[i] == 0
                new_mask = np.tile(new_mask * 255, (3, 1, 1)).transpose((1, 2, 0))
                x1, y1, x2, y2 = box_p[i, :]
                img_matting = (one_obj + new_mask)[y1:y2, x1:x2, :]
                cv2.imwrite(f'results/images/{img_name}_{i}.jpg', img_matting)

    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    if not cfg.hide_bbox:
        for i in reversed(range(num_detected)):
            x1, y1, x2, y2 = box_p[i, :]

            color = COLORS[ids_p[i] + 1].tolist()
            cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

            class_name = cfg.class_names[ids_p[i]]
            text_str = f'{class_name}: {class_p[i]:.2f}' if not cfg.hide_score else class_name

            text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
            cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
            cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if cfg.real_time:
        fps_str = f'fps: {fps:.2f}'
        text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
        # Create a shadow to show the fps more clearly
        img_fused = img_fused.astype(np.float32)
        img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
        img_fused = img_fused.astype(np.uint8)
        cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_fused