# Original hand poses
hand_poses = {
    "indexFl": [[0,0,0,0,0,0], [120, 7, 3, 0, 0, 0]], 
    "indexEx": [[120, 7, 3, 0, 0, 0], [0,0,0,0,0,0]],
    "thumbFl": [[0,0,0,0,0,0], [1, 0, 0, 0, 103, -103]], 
    "thumbEx": [[1, 0, 0, 0, 103, -103],[0,0,0,0,0,0]], 
    "thumbAb": [[0,0,0,0,0,0], [0, 0, 0, 0, 85, 0]], 
    "thumbAd": [[0, 0, 0, 0, 85, -5],[0,0,0,0,0,0]], 
    "mrpFl": [[0,0,0,0,0,0], [5, 120, 120, 120, 0, 0]], 
    "mrpEx": [[5, 120, 120, 120, 0, 0],[0,0,0,0,0,0]], 
    "fingersFl": [[0,0,0,0,0,0], [120, 120, 120, 120, 0, 0]],
    "fingersEx": [[120, 120, 120, 120, 0, 0], [0,0,0,0,0,0]],
    "handCl": [[0,0,0,0,0,0], [50, 50, 50, 50, 30, -30], [80, 80, 80, 80, 40, -40], [90, 90, 90, 90, 40, -60], [100, 100, 100, 100, 50, -80], [120, 120, 120, 120, 60, -100]],
    "handOp": [[120, 120, 120, 120, 60, -100], [100, 100, 100, 100, 40, -60], [90, 90, 90, 90, 30, -30], [0,0,0,0,0,0]],
    "pinchCl": [[0,0,0,0,0,0], [60, 5, 0, 0, 59, -59]],
    "pinchOp": [[60, 0, 0, 0, 59, -59],[0,0,0,0,0,0]],
    "pointCl": [[0,0,0,0,0,0], [50, 0, 0, 0, 0, 0]], 
    "pointOp": [[50, 0, 0, 0, 0, 0],[0,0,0,0,0,0]], 
    "indexFlDigits": [[63, 0, 0, 0, 59, -59], [0, 120, 120, 120, 0, -5]],
    "indexDigitsEx": [[0, 120, 120, 120, 0, 0], [63, 0, 0, 0, 59, -59]],
    "thumbCl": [[120, 120, 120, 120, 85, -10], [120, 120, 120, 120, 0, 0]],
    "thumbOp": [[120, 120, 120, 120, 0, 0], [120, 120, 120, 120, 85, -10]],
    "indexCl": [[120, 120, 120, 120, 60, -110], [0, 120, 120, 120, 60, -110]],
    "indexOp": [[0, 120, 120, 120, 60, -110], [120, 120, 120, 120, 60, -110]],
    "indexFlEx": ["indexFl", "indexEx"],
    "indexClOp": ["indexCl", "indexOp"],
    "thumbFlEx": ["thumbFl", "thumbEx"],
    "thumbAbAd": ["thumbAb", "thumbAd"], 
    "thumbClOp": ["thumbCl", "thumbOp"],
    "mrpFlEx": ["mrpFl", "mrpEx"],
    "fingersFlEx": ["fingersFl", "fingersEx"],
    "handClOp": ["handCl", "handOp"],
    "pinchClOp": ["pinchCl", "pinchOp"],
    "pointClOp": ["pointCl", "pointOp"],
    "indexDigitsFlEx": ["indexFlDigits", "indexDigitsEx"],
    
    # Comprehensive test poses
    "comp_fist": [120, 120, 120, 120, 60, -100],
    "comp_fist_step1": [50, 50, 50, 50, 30, -30],
    "comp_fist_step2": [80, 80, 80, 80, 40, -40],
    "comp_fist_step3": [90, 90, 90, 90, 40, -60],
    "comp_fist_step4": [100, 100, 100, 100, 50, -80],
    "comp_fist_final": [120, 120, 120, 120, 60, -100],
    
    "comp_pinch": [60, 0, 0, 0, 59, -59],
    "comp_precision_pinch": [30, 0, 0, 0, 40, -40],
    "comp_power_grip": [90, 90, 90, 90, 50, -80],
    "comp_key_grip": [60, 0, 0, 0, 85, 0],
    
    "comp_point": [0, 120, 120, 120, 0, 0],
    "comp_peace": [0, 0, 120, 120, 0, 0],
    "comp_thumbs_up": [120, 120, 120, 120, 85, 0],
    
    "comp_index_only": [120, 0, 0, 0, 0, 0],
    "comp_middle_only": [0, 120, 0, 0, 0, 0],
    "comp_ring_only": [0, 0, 120, 0, 0, 0],
    "comp_pinky_only": [0, 0, 0, 120, 0, 0],
    "comp_thumb_flex": [0, 0, 0, 0, 103, -103],
    "comp_thumb_abduct": [0, 0, 0, 0, 85, 0],
    
    "comp_half_fist": [60, 60, 60, 60, 30, -30],
    "comp_claw": [40, 40, 40, 40, 20, -20],
    "comp_relaxed": [20, 20, 20, 20, 10, -10],
    "comp_spread": [0, 0, 0, 0, 0, 85],
}

# Comprehensive test sequence that runs the full test
# This is treated as a special "movement" that executes the comprehensive test
COMPREHENSIVE_TEST_SEQUENCE = [
    # 1. Basic poses
    [0, 0, 0, 0, 0, 0],  # rest
    # [50, 50, 50, 50, 30, -30],  # fist_step1 (anti-collision)
    # [80, 80, 80, 80, 40, -40],  # fist_step2
    # [90, 90, 90, 90, 40, -60],  # fist_step3
    # [100, 100, 100, 100, 50, -80],  # fist_step4
    [120, 120, 120, 120, 60, -100],  # fist_final
    [0, 0, 0, 0, 0, 0],  # rest
    
    # 2. Pinch
    [60, 0, 0, 0, 59, -59],  # pinch
    [0, 0, 0, 0, 0, 0],  # rest
    
    # 3. Point
    [0, 120, 120, 120, 0, 0],  # point
    [0, 0, 0, 0, 0, 0],  # rest

    # 4. Rock
    [0, 120, 120, 0, 0, 0],  # point
    [0, 0, 0, 0, 0, 0],  # rest
    
    # 5. Thumbs up
    [120, 120, 120, 120, 85, 0],  # thumbs_up
    [0, 0, 0, 0, 0, 0],  # rest
    
    # 6. Peace
    [0, 0, 120, 120, 0, 0],  # peace
    [0, 0, 0, 0, 0, 0],  # rest
    
    # 7. Individual joints test
    [120, 0, 0, 0, 0, 0],  # index_only
    [0, 0, 0, 0, 0, 0],  # rest
    [0, 120, 0, 0, 0, 0],  # middle_only  
    [0, 0, 0, 0, 0, 0],  # rest
    [0, 0, 120, 0, 0, 0],  # ring_only
    [0, 0, 0, 0, 0, 0],  # rest
    [0, 0, 0, 120, 0, 0],  # pinky_only
    [0, 0, 0, 0, 0, 0],  # rest
    [0, 0, 0, 0, 103, -103],  # thumb_flex
    [0, 0, 0, 0, 0, 0],  # rest
    [0, 0, 0, 0, 85, 0],  # thumb_abduct
    [0, 0, 0, 0, 0, 0],  # rest
    
    # 8. Functional grips
    [30, 0, 0, 0, 40, -40],  # precision_pinch
    [0, 0, 0, 0, 0, 0],  # rest
    [60, 0, 0, 0, 85, 0],  # key_grip
    [0, 0, 0, 0, 0, 0],  # rest
    [90, 90, 90, 90, 50, -80],  # power_grip
    [0, 0, 0, 0, 0, 0],  # rest
    
    # 8. Complex positions
    [60, 60, 60, 60, 30, -30],  # half_fist
    [0, 0, 0, 0, 0, 0],  # rest
    [40, 40, 40, 40, 20, -20],  # claw
    [0, 0, 0, 0, 0, 0],  # rest
    [20, 20, 20, 20, 10, -10],  # relaxed
    [0, 0, 0, 0, 0, 0],  # rest (final)
]

hand_poses["Comp"] = COMPREHENSIVE_TEST_SEQUENCE