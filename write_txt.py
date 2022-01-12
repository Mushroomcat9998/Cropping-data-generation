import glob


def export_txt(path, mode):
    # img_folder = 'datasets/{}/img'.format(mode)
    gt_folder = '{}/{}/gt'.format(path, mode)

    with open('{}/{}.txt'.format(path, mode), 'w') as f:
        f.write('')

    for file in glob.glob(gt_folder + '/*.txt'):
        file_name = file.split('\\')[-1][:-4]

        img_path = '/content/{}/{}/img'.format('10k_dataset_new', mode) + '/' + file_name + '.jpg'
        gt_path = '/content/{}/{}/gt'.format('10k_dataset_new', mode) + '/' + file_name + '.txt'

        with open('{}/{}.txt'.format(path, mode), 'a') as f:
            content = img_path + '\t' + gt_path + '\n'
            f.write(content)


export_txt(r'D:\OCR\Preprocessing\Transform', 'test')
# export_txt(r'D:\OCR', '10k_dataset', 'test')
