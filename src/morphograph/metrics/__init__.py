"""Evaluation metrics: mIoU, BF1, clDice, ConnR, graph metrics.

Metrics tracked (from sibling project baseline results):
    Segmentation: mIoU_fg, mIoU_all, per-class IoU, Dice, pixel_acc
    Boundary:     BF1 (2px tolerance)
    Topology:     clDice, ConnR (connectivity recall)
    Graph:        endpoint F1, junction F1, edge F1, GED, path recall
    Engineering:  crack length error, width MAE
"""
