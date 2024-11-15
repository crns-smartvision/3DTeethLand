import numpy as np
# import matplotlib.pyplot as plt

def voc_ar(dist_thresh_list, recall, keypoint_cat):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.array(dist_thresh_list[::-1])
    mpre = np.array(recall[keypoint_cat][::-1])
    mrec = np.concatenate(([0.], mrec, [1.]))
    mpre = np.concatenate(([0.], mpre, [0.]))

    # compute the recall envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ar = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    # plt.plot(mrec, mpre, label='AR Curve')
    # plt.fill_between(mrec, mpre, color='skyblue', alpha=0.4)
    #
    # # Highlight segments contributing to AR
    # for j in range(len(i)):
    #     start_idx = i[j]
    #     end_idx = i[j + 1] + 1 if j + 1 < len(i) else len(mrec) - 1
    #     plt.plot(mrec[start_idx:end_idx], mpre[start_idx:end_idx], 'r-', alpha=0.7)
    #
    # plt.xlabel('dist threshold')
    # plt.ylabel('Recall')
    # plt.title(f'AR Curve')
    # plt.title(f'{keypoint_cat}: Distance-recall Curve (AR={ar:.4f})')
    # plt.ylim([0, 1])
    # plt.xlim([0, 1])
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return ar

def voc_ap(rec, prec, threshold, keypoint_cat):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    # Plot precision recall curve
    # plt.plot(mrec, mpre, label='Precision-Recall Curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'{keypoint_cat}: Precision-Recall Curve (AP@{threshold:.1f}={ap:.4f})')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return ap


def eval_det_cls_map(pred, gt, dist_thresh, classname):
    # construct gt objects
    class_recs = {}  # {mesh name: {'kp': kp list, 'det': matched list}}
    npos = 0
    for mesh_name in gt.keys():
        keypoints = np.array(gt[mesh_name])
        det = [False] * len(keypoints)
        npos += len(keypoints)
        class_recs[mesh_name] = {'kp': keypoints, 'det': det}
    # pad empty list to all other imgids
    for mesh_name in pred.keys():
        if mesh_name not in gt:
            class_recs[mesh_name] = {'kp': np.array([]), 'det': []}

    # construct dets
    mesh_names = []
    confidence = []
    KP = []
    for mesh_name in pred.keys():
        for kp, score in pred[mesh_name]:
            mesh_names.append(mesh_name)
            confidence.append(score)
            KP.append(kp)
    confidence = np.array(confidence)
    KP = np.array(KP)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    KP = KP[sorted_ind, ...]
    mesh_names = [mesh_names[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(mesh_names)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[mesh_names[d]]
        kp = KP[d]
        dmin = np.inf
        KPGT = R['kp']

        if KPGT.size > 0:
            distance = np.linalg.norm(np.array(kp).reshape(-1, 3) - KPGT, axis=1)
            dmin = min(distance)
            jmin = np.argmin(distance)

        # print dmin
        if dmin < dist_thresh:
            if not R['det'][jmin]:
                tp[d] = 1.
                R['det'][jmin] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # print('NPOS: ' + str(npos))
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, dist_thresh, classname)
    return rec, prec, ap


def eval_map(pred_all, gt_all, dist_thresh=0.1):
    """ Generic functions to compute precision/recall for keypoint detection
        for multiple classes.
        Input:
            pred_all: map of {classname: {meshname: [(kp, score)]}}
            gt_all: map of {classname: {meshname: [kp]}}
            dist_thresh: scalar, iou threshold
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """

    rec = {}
    prec = {}
    ap = {}
    for classname in gt_all.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls_map(pred_all[classname], gt_all[classname],
                                                                          dist_thresh, classname)

    return rec, prec, ap


def filter_scan(all_scans, scan_name):
    all_filtered = {}
    for category, data_dict in all_scans.items():
        all_filtered[category] = {}
        try:
            all_filtered[category][scan_name] = all_scans[category][scan_name]
        except:
            all_filtered[category][scan_name] = []
    return all_filtered


def calculate_metrics_per_scan(pred_submission, gt_all):
    pred_all_map = {
        "Mesial": {},
        "Distal": {},
        "Cusp": {},
        "InnerPoint": {},
        "OuterPoint": {},
        "FacialPoint": {}
    }
    scans_list = set()
    for _, row in pred_submission.iterrows():
        class_name = row['class']
        key = row['key']
        scans_list.add(key)
        coord = [row['coord_x'], row['coord_y'], row['coord_z']]
        prob = row['score']
        if key not in pred_all_map[class_name]:
            pred_all_map[class_name][key] = [[coord, prob]]
        else:
            pred_all_map[class_name][key].append([coord, prob])

    all_metrics = {}
    scans_list = list(gt_all['Cusp'].keys())
    for scan in scans_list:
        recall = {
            "Mesial": [],
            "Distal": [],
            "Cusp": [],
            "InnerPoint": [],
            "OuterPoint": [],
            "FacialPoint": []
        }
        dist_thresh_list = []
        metrics = {}
        gt = filter_scan(gt_all, scan)
        pred = filter_scan(pred_all_map, scan)
        # define distance threshold
        for i in range(0, 30):
            dist_thresh = 0.1 * i
            rec, prec, ap = eval_map(pred, gt, dist_thresh=dist_thresh)
            metrics[str(i)] = ap
            dist_thresh_list.append(dist_thresh)
            for cat in rec.keys():
                try:
                    recall[cat].append(rec[cat][-1])
                except:
                    recall[cat].append(0)
        # Collect all values for each class
        class_values = {'Mesial': [], 'Distal': [], 'Cusp': [], 'InnerPoint': [], 'OuterPoint': [], 'FacialPoint': []}
        for threshold in metrics.values():
            for class_name in class_values.keys():
                class_values[class_name].append(threshold[class_name])

        # Calculate the mean for each class
        map = {class_name: sum(values) / len(values) for class_name, values in class_values.items()}

        # Calculate mAR
        mar = {}
        for cat in recall.keys():
            ar = voc_ar(np.exp(-np.asarray(dist_thresh_list)), recall, cat)
            mar[cat] = ar

        all_metrics[scan] = {}
        all_metrics[scan] = {"mAP": map, "mAR": mar}
    return all_metrics
