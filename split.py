import splitfolders  
import os
import pathlib


folder = 'data/malignas/output'
splitfolders.ratio(folder, output='data/malignaaugmentated', seed=1337, ratio=(0.75,0.25))