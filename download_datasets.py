import argparse
import urllib
import zipfile
import os

DEFAULT_DATASETS_DIR = "datasets/"

DATASETS_URL = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/"

DATASETS_LIST = [
    "ae_photos", "apple2orange", "summer2winter_yosemite", "horse2zebra",
    "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "maps",
    "cityscapes", "facades", "iphone2dslr_flower"
]

def parse_args():
    parser = argparse.ArgumentParser("Download datasets")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="",
        help="Set to download only specified dataset")
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=DEFAULT_DATASETS_DIR,
        help="Set to override default datasets directory")
    return parser.parse_args()

def download_dataset(name, directory):
    url = DATASETS_URL + name + ".zip"
    tmp_zip_file = directory + name + ".zip"
    print("Downloading %s" % (name))
    url_opener = urllib.URLopener()
    url_opener.retrieve(url, tmp_zip_file)
    print("Downloading %s completed, unzipping" % (name))
    zip_ref = zipfile.ZipFile(tmp_zip_file, 'r')
    zip_ref.extractall(directory)
    zip_ref.close()
    print("Unzip %s completed" % (name))
    os.remove(tmp_zip_file)
    print("Removed tmp zip file, dataset %s is ready" % (name))

def main():
    args = parse_args()
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    if not args.file:
        dlen = len(DATASETS_LIST)
        for i in range(dlen):
             print("Downloading %d / %d datasets" % (i + 1, dlen))
             download_dataset(DATASETS_LIST[i], args.dir)
        return
    if args.file in DATASETS_LIST:
        download_dataset(args.file, args.dir)
        return
    print("Error: unavailable dataset file name")

if __name__ == '__main__':
    main()

