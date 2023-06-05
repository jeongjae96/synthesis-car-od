import os

def train2images(data_root):
    os.rename(
        os.path.join(data_root, 'train'),
        os.path.join(data_root, 'images')
    )
    print("renamed 'train' folder to 'images' folder.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_root',
        dest='data_root',
        default='../../data/',
        help='input data root directory.'
    )
    args = parser.parse_args()

    train2images(args.data_root)