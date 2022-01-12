import glob
import os
import random
import string

folder_path = 'C:/Users/ADMIN/Downloads/*.jpeg'
org_path = 'C:/Users/ADMIN/Downloads/'
letters = string.ascii_lowercase


def get_random_string(length):
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


for file_path in glob.glob(folder_path):
    new_path = org_path + get_random_string(10) + '.jpg'
    os.rename(file_path, new_path)
