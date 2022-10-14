import multiprocessing as mp
import os
import time
from itertools import product
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def lbp_process_image(config):
    start_t = time.perf_counter()
    inpath, method, R, scale = config
    gray = cv2.imread(inpath, 0)
    lbp = local_binary_pattern(gray, R * scale, R, method).astype(np.uint8)
    output_name = os.path.join('test', f'{inpath.split("/")[-2]}',
                               f'{inpath.split("_")[1]}', method, f"R{R}.P{R * scale}.tiff")
    cv2.imwrite(output_name, lbp)
    end_t = time.perf_counter()
    return inpath, method, R, R * scale, end_t - start_t


def main():
    white_pic = [os.path.join(white_folder, pic) for pic in os.listdir(white_folder) if pic.endswith('Tripod.tiff')]
    black_pic = [os.path.join(black_folder, pic) for pic in os.listdir(black_folder) if pic.endswith('Tripod.tiff')]
    pics = list()
    pics.extend(white_pic)
    pics.extend(black_pic)
    # print(pics)
    configs = product(pics, lbp_methods, Rs, P_scale)
    if not os.path.isdir('test'):
        os.mkdir('test')
    if not os.path.isdir(os.path.join('test', 'White')):
        os.mkdir('test/White')
    if not os.path.isdir(os.path.join('test', 'Black')):
        os.mkdir('test/Black')
    for name, method in product(white_pic, lbp_methods):
        if not os.path.isdir(os.path.join('test', 'White', f'{name.split("_")[1]}')):
            os.mkdir(os.path.join('test', 'White', f'{name.split("_")[1]}'))
        if not os.path.isdir(os.path.join('test', 'White', f'{name.split("_")[1]}', method)):
            os.mkdir(os.path.join('test', 'White', f'{name.split("_")[1]}', method))
    for name, method in product(black_pic, lbp_methods):
        if not os.path.isdir(os.path.join('test', 'Black', f'{name.split("_")[1]}')):
            os.mkdir(os.path.join('test', 'Black', f'{name.split("_")[1]}'))
        if not os.path.isdir(os.path.join('test', 'black', f'{name.split("_")[1]}', method)):
            os.mkdir(os.path.join('test', 'Black', f'{name.split("_")[1]}', method))
    start_t = time.perf_counter()
    with mp.Pool() as pool:
        results = pool.imap_unordered(lbp_process_image, configs)
        for name, method, R, P, duration in results:
            print(f"{name.split('/')[-1]} with R: {R} P: {P} on method: {method} took {duration:.2f}s")
    end_t = time.perf_counter()
    print(f"processing all the images took {end_t - start_t: .2f}s")


if __name__ == '__main__':
    white_folder = '/Users/tangyuanzhe/Downloads/Testing Photos - Canon EOS 6D DSLR/White'
    # white folder name has been renamed from "White v2" to "White" for simplifying code
    black_folder = '/Users/tangyuanzhe/Downloads/Testing Photos - Canon EOS 6D DSLR/Black'

    Rs = [1, 2, 3, 4, 5]
    P_scale = [4, 8, 16, 32]
    lbp_methods = ['default', 'ror', 'uniform']
    main()

    # for config in configs:
    #     t = threading.Thread(target=lbp_process_image, args=(config, ))
    #     t.start()

    # for pic in white_pic:
    #     file = cv2.imread(os.path.join(white_folder, pic), cv2.IMREAD_GRAYSCALE)
    #     print(pic)
    #     for method, R, scale in configs:
    #         if not os.path.isdir(os.path.join('test', 'white', f'{pic.split("_")[1]}')):
    #             os.mkdir(os.path.join('test', 'white', f'{pic.split("_")[1]}'))
    #         if not os.path.isdir(os.path.join('test', 'white', f'{pic.split("_")[1]}', method)):
    #             os.mkdir(os.path.join('test', 'white', f'{pic.split("_")[1]}', method))
    #         print(f"R: {R}, P: {R * scale}, method: {method}")
    #         lbp = local_binary_pattern(file, R * scale, R, method).astype(np.uint8)
    #         cv2.imwrite(os.path.join('test', 'white', f'{pic.split("_")[1]}', method, f"R{R}.P{R * scale}.tiff"), lbp)
    # print('end')

    # blurred = cv2.medianBlur(grey, 7)
    # lbp = local_binary_pattern(file, 16, 2, 'default').astype(np.uint8)
    # cv2.imwrite('White_test_lbp_6.tiff', lbp)
