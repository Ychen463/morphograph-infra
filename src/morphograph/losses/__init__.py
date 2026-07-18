"""Loss functions for multi-task training.

Each baseline configuration has a specific loss set:
    B0:   CE(w=[0.2,2.0,3.0]) + FG-only Dice
    B1a:  B0 + clDice (scheduled from epoch 40, w=0.15)
    B1b:  B0 + SRL
    B2:   B0 + skeleton BCE+Dice (pos_weight=50)
    B3:   B2 + endpoint BCE (pos_weight=100) + junction BCE (pos_weight=100)
    B4:   B3 + edge connectivity loss
    B5:   B4 + width Smooth L1 (masked to skeleton)

Tversky and boundary F1 losses are ablation variants, not defaults.
"""
