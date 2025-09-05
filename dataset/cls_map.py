cls_map = ['Children Cabinet', 'Nightstand', 'Bookcase / jewelry Armoire', 'Wardrobe', 'Coffee Table',
                'Corner/Side Table', 'Sideboard / Side Cabinet / Console Table', 'Wine Cabinet', 'TV Stand', 'Drawer Chest / Corner cabinet',
                'Shelf', 'Round End Table', 'King-size Bed', 'Bunk Bed', 'Bed Frame',
                'Single bed', 'Kids Bed', 'Dining Chair', 'Lounge Chair / Cafe Chair / Office Chair', 'Dressing Chair',
                'Classic Chinese Chair', 'Barstool', 'Dressing Table', 'Dining Table', 'Desk',
                'Three-Seat / Multi-seat Sofa', 'armchair', 'Loveseat Sofa', 'L-shaped Sofa', 'Lazy Sofa',
                'Chaise Longue Sofa', 'Footstool / Sofastool / Bed End Stool / Stool', 'Pendant Lamp', 'Ceiling Lamp', 'Back',
                'Flue', 'CustomizedFixedFurniture', 'WallInner', 'CustomizedCeiling', 'Cabinet',
                'LightBand', 'SmartCustomizedCeiling', 'Floor', 'CustomizedPlatform', 'CustomizedFurniture',
                'Customized_wainscot', 'Window', 'CustomizedPersonalizedModel', 'Column', 'clipMesh',
                'WallOuter', 'Front', 'Hole', 'SewerPipe', 'BayWindow',
                'SlabSide', 'Pocket', 'SlabBottom', 'Beam', 'Cornice',
                'Baseboard', 'SlabTop', 'WallTop', 'CustomizedBackgroundModel', 'Door',
                'WallBottom', 'Cabinet/LighBand', 'Ceiling', 'CustomizedFeatureWall', 'ExtrusionCustomizedCeilingModel',
                'ExtrusionCustomizedBackgroundWall']
print(cls_map[13])


# import numpy as np

# from tqdm import tqdm
# import glob
# import os
# import sys

# data_path = '/cluster/sc_download/zhuwanru/density1250'
# data_list = sorted(glob.glob(os.path.join(data_path, '*.npy')))
# for i in range(71):
#     if not os.path.exists('cls/'+str(i)):
#         os.makedirs('cls/'+str(i))
# class_label = 23
# for file in tqdm(data_list):
#     file_name = file.split('/')[-1].split('.')[0]
#     data = np.load(file)
#     pc = data[:, 0:3]
#     label = data[:, 6]
#     if class_label in np.unique(label):
#         np.savetxt('cls/'+str(int(class_label))+'/'+file_name+'.txt', pc[label==class_label])
#     # for i in np.unique(label):
#     # np.savetxt('cls/'+str(int(i))+'/'+file_name+'.txt', pc[label==9])

