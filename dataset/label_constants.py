'''Label file for different datasets.'''

TRUCKSCENES_LABELS_TO_IDX = {"animal": 0 ,
                            "human.pedestrian.adult": 7,
                            "human.pedestrian.child": 7 ,
                            "human.pedestrian.construction_worker": 7 ,
                            "human.pedestrian.personal_mobility": 7 ,
                            "human.pedestrian.police_officer": 7 ,
                            "human.pedestrian.stroller": 255 ,
                            "human.pedestrian.wheelchair": 255 ,
                            "movable_object.barrier": 1 ,
                            "movable_object.debris": 255 ,
                            "movable_object.pushable_pullable": 255,
                            "movable_object.trafficcone": 8,
                            "static_object.bicycle_rack": 255,
                            "static_object.traffic_sign": 9,
                            "vehicle.bicycle": 2,
                            "vehicle.bus.bendy": 3,
                            "vehicle.bus.rigid": 3,
                            "vehicle.car": 4,
                            "vehicle.construction": 6,
                            "vehicle.emergency.ambulance": 255,
                            "vehicle.emergency.police": 255,
                            "vehicle.motorcycle": 5,
                            "vehicle.trailer": 10,
                            "vehicle.truck": 11,
                            "vehicle.train": 255,
                            "vehicle.other": 255,
                            "vehicle.ego_trailer": 255,
                            "unlabeled": 255
}


SCANNET_LABELS_20 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture','counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
                     'toilet', 'sink', 'bathtub', 'otherfurniture')


MATTERPORT_LABELS_21 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                    'shower curtain', 'toilet', 'sink', 'bathtub', 'other', 'ceiling')

MATTERPORT_LABELS_40 = ('wall', 'door', 'ceiling', 'floor', 'picture', 'window', 'chair',
                'pillow', 'lamp', 'cabinet', 'curtain', 'table', 'plant', 'mirror',
                'towel', 'sink', 'shelves', 'sofa', 'bed', 'night stand', 'toilet', 'column',
                'banister', 'stairs', 'stool', 'vase', 'television', 'pot', 'desk', 'box',
                'coffee table', 'counter', 'bench', 'garbage bin', 'fireplace',
                'clothes', 'bathtub', 'book', 'air vent', 'faucet')

MATTERPORT_LABELS_80 = ('wall', 'door', 'ceiling', 'floor', 'picture', 'window', 'chair', 'pillow', 'lamp',
                           'cabinet', 'curtain', 'table', 'plant', 'mirror', 'towel', 'sink', 'shelves', 'sofa', 
                           'bed', 'night stand', 'toilet', 'column', 'banister', 'stairs', 'stool', 'vase', 
                           'television', 'pot', 'desk', 'box', 'coffee table', 'counter', 'bench', 'garbage bin', 
                           'fireplace', 'clothes', 'bathtub', 'book', 'air vent', 'faucet', 'photo', 'toilet paper', 
                           'fan', 'railing', 'sculpture', 'dresser', 'rug', 'ottoman', 'bottle', 'refridgerator', 
                           'bookshelf', 'wardrobe', 'pipe', 'monitor', 'stand', 'drawer', 'container', 'light switch', 
                           'purse', 'door way', 'basket', 'chandelier', 'oven', 'clock', 'stove', 'washing machine', 
                           'shower curtain', 'fire alarm', 'bin', 'chest', 'microwave', 'blinds', 'bowl', 'tissue box', 
                           'plate', 'tv stand', 'shoe', 'heater', 'headboard', 'bucket')

MATTERPORT_LABELS_160 = ('wall', 'door', 'ceiling', 'floor', 'picture', 'window', 'chair', 'pillow', 'lamp', 
                            'cabinet', 'curtain', 'table', 'plant', 'mirror', 'towel', 'sink', 'shelves', 'sofa', 
                            'bed', 'night stand', 'toilet', 'column', 'banister', 'stairs', 'stool', 'vase', 
                            'television', 'pot', 'desk', 'box', 'coffee table', 'counter', 'bench', 'garbage bin', 
                            'fireplace', 'clothes', 'bathtub', 'book', 'air vent', 'faucet', 'photo', 'toilet paper', 
                            'fan', 'railing', 'sculpture', 'dresser', 'rug', 'ottoman', 'bottle', 'refridgerator', 
                            'bookshelf', 'wardrobe', 'pipe', 'monitor', 'stand', 'drawer', 'container', 'light switch', 
                            'purse', 'door way', 'basket', 'chandelier', 'oven', 'clock', 'stove', 'washing machine', 
                            'shower curtain', 'fire alarm', 'bin', 'chest', 'microwave', 'blinds', 'bowl', 'tissue box', 
                            'plate', 'tv stand', 'shoe', 'heater', 'headboard', 'bucket', 'candle', 'flower pot', 
                            'speaker', 'furniture', 'sign', 'air conditioner', 'fire extinguisher', 'curtain rod', 
                            'floor mat', 'printer', 'telephone', 'blanket', 'handle', 'shower head', 'soap', 'keyboard', 
                            'thermostat', 'radiator', 'kitchen island', 'paper towel', 'sheet', 'glass', 'dishwasher', 
                            'cup', 'ladder', 'garage door', 'hat', 'exit sign', 'piano', 'board', 'rope', 'ball', 
                            'excercise equipment', 'hanger', 'candlestick', 'light', 'scale', 'bag', 'laptop', 'treadmill', 
                            'guitar', 'display case', 'toilet paper holder', 'bar', 'tray', 'urn', 'decorative plate', 'pool table', 
                            'jacket', 'bottle of soap', 'water cooler', 'utensil', 'tea pot', 'stuffed animal', 'paper towel dispenser', 
                            'lamp shade', 'car', 'toilet brush', 'doll', 'drum', 'whiteboard', 'range hood', 'candelabra', 'toy', 
                            'foot rest', 'soap dish', 'placemat', 'cleaner', 'computer', 'knob', 'paper', 'projector', 'coat hanger', 
                            'case', 'pan', 'luggage', 'trinket', 'chimney', 'person', 'alarm')

NUSCENES_LABELS_16 = ('barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'person', 'traffic cone',
                      'trailer', 'truck', 'drivable surface', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')

TRUCKSCENES_LABELS_12 = ('animal', 'barrier', 'bicycle', 'bus', 'car', 'motorcycle', 'construction vehicle', 'person', 'traffic cone',
                         'traffic sign', 'trailer', 'truck')

NUSCENES_LABELS_DETAILS = ('barrier', 'barricade', 'bicycle', 'bus', 'car', 'bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck',
                           'motorcycle', 'person', 'pedestrian','traffic cone', 'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
                           'truck', 'road', 'curb', 'traffic island', 'traffic median', 'sidewalk', 'grass', 'grassland', 'lawn', 'meadow', 'turf', 'sod',
                           'building', 'wall', 'pole', 'awning', 'tree', 'trunk', 'tree trunk', 'bush', 'shrub', 'plant', 'flower', 'woods')


MAPPING_NUSCENES_DETAILS = (0, 0, 1, 2, 3, 4, 4, 4, 4, 4,
                            5, 6, 6, 7, 8, 8, 8, 8, 8,
                            9, 10, 11, 11, 11, 12, 13, 13, 13, 13, 13, 13,
                            14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15 # last 15 here added by myself to fix visualization  
                            )
TRUCKSCENES12_COLORMAP = {
    1: (160, 82, 45),      # animal 
    2: (220, 220, 0),      # barrier
    3: (119, 11, 32),      # bicycle
    4: (0, 60, 100),       # bus
    5: (0, 0, 250),        # car
    6: (0, 0, 230),        # motorcycle
    7: (230, 230, 250),    # construction vehicle
    8: (220, 20, 60),      # person
    9: (250, 170, 30),     # traffic cone
    10: (200, 150, 0),     # traffic sign
    11: (0, 0, 110),       # trailer
    12: (128, 64, 128),    # truck
    13: (0, 0, 0),         # unknown / unlabeled
}


NUSCENES16_COLORMAP = {
    1: (220,220,  0), # barrier
    2: (119, 11, 32), # bicycle
    3: (0, 60, 100), # bus
    4: (0, 0, 250), # car
    5: (230,230,250), # construction vehicle
    6: (0, 0, 230), # motorcycle
    7: (220, 20, 60), # person
    8: (250, 170, 30), # traffic cone
    9: (200, 150, 0), # trailer
    10: (0, 0, 110) , # truck
    11: (128, 64, 128), # road
    12: (0,250, 250), # other flat
    13: (244, 35, 232), # sidewalk
    14: (152, 251, 152), # terrain
    15: (70, 70, 70), # manmade
    16: (107,142, 35), # vegetation
    17: (0, 0, 0), # unknown
    }

SCANNET_COLOR_MAP_20 = {
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    16: (219., 219., 141.),
    24: (255., 127., 14.),
    28: (158., 218., 229.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    36: (227., 119., 194.),
    39: (82., 84., 163.),
    0: (0., 0., 0.), # unlabel/unknown
}

MATTERPORT_COLOR_MAP_21 = {
    1: (174., 199., 232.), # wall
    2: (152., 223., 138.), # floor
    3: (31., 119., 180.), # cabinet
    4: (255., 187., 120.), # bed
    5: (188., 189., 34.), # chair
    6: (140., 86., 75.), # sofa
    7: (255., 152., 150.), # table
    8: (214., 39., 40.), # door
    9: (197., 176., 213.), # window
    10: (148., 103., 189.), # bookshelf
    11: (196., 156., 148.), # picture
    12: (23., 190., 207.), # counter
    14: (247., 182., 210.), # desk
    16: (219., 219., 141.), # curtain
    24: (255., 127., 14.), # refrigerator
    28: (158., 218., 229.), # shower curtain
    33: (44., 160., 44.), # toilet
    34: (112., 128., 144.), # sink
    36: (227., 119., 194.), # bathtub
    39: (82., 84., 163.), # other
    # 41: (186., 197., 62.), # ceiling
    41: (58., 98., 26.), # ceiling
    0: (0., 0., 0.), # unlabel/unknown
}

MATTERPORT_COLOR_MAP_160 = {
    1: (174., 199., 232.), # wall
    2: (214., 39., 40.), # door
    3: (186., 197., 62.), # ceiling
    4: (152., 223., 138.), # floor
    5: (196., 156., 148.), # picture
    6: (197., 176., 213.), # window
    7: (188., 189., 34.), # chair
    8: (141., 91., 229.), # pillow
    9: (237.0, 204.0, 37.0), # lamp
    10: (31., 119., 180.), # cabinet
    11: (219., 219., 141.), # curtain
    12: (255., 152., 150.), # table
    13: (150.0, 53.0, 56.0), # plant
    14: (162.0, 62.0, 60.0), # mirror
    15: (62.0, 143.0, 148.0), # towel
    16: (112., 128., 144.), # sink
    17: (229.0, 91.0, 104.0), # shelves
    18: (140., 86., 75.), # sofa
    19: (255., 187., 120.), # bed
    20: (137.0, 63.0, 14.0), # night stand
    21: (44., 160., 44.), # toilet
    22: (39.0, 19.0, 208.0), # column
    23: (64.0, 158.0, 70.0), # banister
    24: (208.0, 49.0, 84.0), # stairs
    25: (90.0, 119.0, 201.0), # stool
    26: (118., 174., 76.), # vase
    27: (143.0, 45.0, 115.0), # television
    28: (153., 108., 234.), # pot
    29: (247., 182., 210.), # desk
    30: (177.0, 82.0, 239.0), # box
    31: (58.0, 98.0, 137.0), # coffee table
    32: (23., 190., 207.), # counter
    33: (17.0, 242.0, 171.0), # bench
    34: (79.0, 55.0, 137.0), # garbage bin
    35: (127.0, 63.0, 52.0), # fireplace
    36: (34.0, 14.0, 130.0), # clothes
    37: (227., 119., 194.), # bathtub
    38: (192.0, 229.0, 91.0), # book
    39: (49.0, 206.0, 87.0), # air vent
    40: (250., 253., 26.), # faucet
    41: (0., 0., 0.), # unlabel/unknown
    80: (82., 75., 227.),
    82: (253., 59., 222.),
    84: (240., 130., 89.),
    86: (123., 172., 47.),
    87: (71., 194., 133.),
    88: (24., 94., 205.),
    89: (134., 16., 179.),
    90: (159., 32., 52.),
    93: (213., 208., 88.),
    95: (64., 158., 70.),
    96: (18., 163., 194.),
    97: (65., 29., 153.),
    98: (177., 10., 109.),
    99: (152., 83., 7.),
    100: (83., 175., 30.),
    101: (18., 199., 153.),
    102: (61., 81., 208.),
    103: (213., 85., 216.),
    104: (170., 53., 42.),
    105: (161., 192., 38.),
    106: (23., 241., 91.),
    107: (12., 103., 170.),
    110: (151., 41., 245.),
    112: (133., 51., 80.),
    115: (184., 162., 91.),
    116: (50., 138., 38.),
    118: (31., 237., 236.),
    120: (39., 19., 208.),
    121: (223., 27., 180.),
    122: (254., 141., 85.),
    125: (97., 144., 39.),
    128: (106., 231., 176.),
    130: (12., 61., 162.),
    131: (124., 66., 140.),
    132: (137., 66., 73.),
    134: (250., 253., 26.),
    136: (55., 191., 73.),
    138: (60., 126., 146.),
    139: (153., 108., 234.),
    140: (184., 58., 125.),
    141: (135., 84., 14.),
    145: (139., 248., 91.),
    148: (53., 200., 172.),
    154: (63., 69., 134.),
    155: (190., 75., 186.),
    156: (127., 63., 52.),
    157: (141., 182., 25.),
    159: (56., 144., 89.),
    161: (64., 160., 250.),
    163: (182., 86., 245.),
    165: (139., 18., 53.),
    166: (134., 120., 54.),
    168: (49., 165., 42.),
    169: (51., 128., 133.),
    170: (44., 21., 163.),
    177: (232., 93., 193.),
    180: (176., 102., 54.),
    185: (116., 217., 17.),
    188: (54., 209., 150.),
    191: (60., 99., 204.),
    193: (129., 43., 144.),
    195: (252., 100., 106.),
    202: (187., 196., 73.),
    208: (13., 158., 40.),
    213: (52., 122., 152.),
    214: (128., 76., 202.),
    221: (187., 50., 115.),
    229: (180., 141., 71.),
    230: (77., 208., 35.),
    232: (72., 183., 168.),
    233: (97., 99., 203.),
    242: (172., 22., 158.),
    250: (155., 64., 40.),
    261: (118., 159., 30.),
    264: (69., 252., 148.),
    276: (45., 103., 173.),
    283: (111., 38., 149.),
    286: (184., 9., 49.),
    300: (188., 174., 67.),
    304: (53., 206., 53.),
    312: (97., 235., 252.),
    323: (66., 32., 182.),
    325: (236., 114., 195.),
    331: (241., 154., 83.),
    342: (133., 240., 52.),
    356: (16., 205., 144.),
    370: (75., 101., 198.),
    392: (237., 95., 251.),
    395: (191., 52., 49.),
    399: (227., 254., 54.),
    408: (49., 206., 87.),
    417: (48., 113., 150.),
    488: (125., 73., 182.),
    540: (229., 32., 114.),
    562: (158., 119., 28.),
    570: (60., 205., 27.),
    572: (18., 215., 201.),
    581: (79., 76., 153.),
    609: (134., 13., 116.),
    748: (192., 97., 63.),
    776: (108., 163., 18.),
    1156: (95., 220., 156.),
    1163: (98., 141., 208.),
    1164: (144., 19., 193.),
    1165: (166., 36., 57.),
    1166: (212., 202., 34.),
    1167: (23., 206., 34.),
    1168: (91., 211., 236.),
    1169: (79., 55., 137.),
    1170: (182., 19., 117.),
    1171: (134., 76., 14.),
    1172: (87., 185., 28.),
    1173: (82., 224., 187.),
    1174: (92., 110., 214.),
    1175: (168., 80., 171.),
    1176: (197., 63., 51.),
    1178: (175., 199., 77.),
    1179: (62., 180., 98.),
    1180: (8., 91., 150.),
    1181: (77., 15., 130.),
    1182: (154., 65., 96.),
    1183: (197., 152., 11.),
    1184: (59., 155., 45.),
    1185: (12., 147., 145.),
    1186: (54., 35., 219.),
    1187: (210., 73., 181.),
    1188: (221., 124., 77.),
    1189: (149., 214., 66.),
    1190: (72., 185., 134.),
    1191: (42., 94., 198.),
    1200: (0, 0, 0)
}
