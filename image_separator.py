import csv
from shutil import copy
import os

with open('ISIC_2020_Training_GroundTruth.csv') as csv_file:
    
    root = os.path.abspath(os.curdir)
    original_folder = root + "/data/melanoma-original/jpeg/train/"
    destination_folder = root + "/data/melanoma-particionada/" 
    print(original_folder)
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    benigno = 0
    maligno = 0
    for row in csv_reader:
        image_name = row[0]
        original_path = original_folder + image_name + ".jpg"
        destination_path = ""
        if row[6] == "benign":
            destination_path = destination_folder + 'benigno/' + image_name + ".jpg"
            copy(original_path, destination_path)
            benigno = benigno + 1
        elif row[6] == "malignant":
            destination_path = destination_folder + 'maligno/' + image_name + ".jpg"
            copy(original_path, destination_path)
            maligno = maligno + 1
    print("Benigno = {} , maligno = {}".format(benigno,maligno))
    
