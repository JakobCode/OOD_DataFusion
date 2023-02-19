"""
Dataset class for the multimodal BigEarthNet data set. The implementations build
up on the data loader provided here:

https://git.tu-berlin.de/rsim/BigEarthNet-MM_tools (MIT License)
"""
import numpy as np

BAND_STATS = {
            "S2":{
                "mean": {
                    "B01": 340.76769064,
                    "B02": 429.9430203,
                    "B03": 614.21682446,
                    "B04": 590.23569706,
                    "B05": 950.68368468,
                    "B06": 1792.46290469,
                    "B07": 2075.46795189,
                    "B08": 2218.94553375,
                    "B8A": 2266.46036911,
                    "B09": 2246.0605464,
                    "B11": 1594.42694882,
                    "B12": 1009.32729131
                },
                "std": {
                    "B01": 554.81258967,
                    "B02": 572.41639287,
                    "B03": 582.87945694,
                    "B04": 675.88746967,
                    "B05": 729.89827633,
                    "B06": 1096.01480586,
                    "B07": 1273.45393088,
                    "B08": 1365.45589904,
                    "B8A": 1356.13789355,
                    "B09": 1302.3292881,
                    "B11": 1079.19066363,
                    "B12": 818.86747235
                }
            },
            "S1": {
                "mean": { # we ordered them by naming --> first VH then VV
                    "VH": -19.29044597721542,
                    "VV": -12.619993741972035,
                    "VV/VH": 0.6525036195871579,
                },
                "std": {
                    "VH": 5.464428464912864,
                    "VV": 5.115911777546365,
                    "VV/VH": 30.75264076801808,
                },
                "min": {
                    "VH": -75.11137390136719,
                    "VV": -74.33214569091797,
                    "R": 3.21E-2
                },
                "max": {
                    "VH": 33.59768295288086,
                    "VV": 34.60696029663086,
                    "R": 1.08
                }
            }
        }

CLASS_NAMES = [ "Urban Fabric",
                "Industrial or commercial units",
                "Arable Land", 
                "Permannent Crops",
                "Pasture",
                "Complex cultivation patterns",
                "mostly agriculture",
                "agro-forestry",
                "broad-leaved forest",
                "Coniferous fores",
                "Mixed forest",
                "natural grassland",
                "moors",
                "transitional woodland-shrub",
                "beaches, dunes, sand",
                "inland wetlands",
                "coastal wetlands",
                "inland water",
                "marine water"]

MEAN = np.reshape(np.concatenate([list(BAND_STATS["S1"]["mean"].values())[:-1], \
                                list(BAND_STATS["S2"]["mean"].values())]), (-1,1,1))
STD = np.reshape(np.concatenate([list(BAND_STATS["S1"]["std"].values())[:-1], \
                                list(BAND_STATS["S2"]["std"].values())]), (-1,1,1))

def get_class_names(class_filter=None):
    """Returns class names for list of class indices."""
    if class_filter is None:
        class_filter = []
    return [CLASS_NAMES[i] for i in range(19) if i in class_filter]

def normalize(img_raw):
    """Normalize input image with mean and std of trainingn set."""
    return (img_raw-MEAN) / STD
