import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)\


java_small_test_files_url = "https://nghimodel.s3-ap-southeast-1.amazonaws.com/java-small-graph-testing.zip"
java_med_test_files_url = "https://nghimodel.s3-ap-southeast-1.amazonaws.com/java-med-graph-testing.zip"
java_large_test_files_url = "https://nghimodel.s3-ap-southeast-1.amazonaws.com/java-large-graph-testing.zip"

java_small_test_files_zip = "data/java-small-graph-testing.zip"
java_med_test_files_zip = "data/java-med-graph-testing.zip"
java_large_test_files_zip = "data/java-large-graph-testing.zip"

download_url(java_small_test_files_url, java_small_test_files_zip)
download_url(java_med_test_files_url, java_med_test_files_zip)
download_url(java_large_test_files_url, java_large_test_files_zip)


extracted_java_small_files_path = "data/java-small-graph-transformed"
extracted_java_med_files_path = "data/java-med-graph-transformed"
extracted_java_large_files_path = "data/java-large-graph-transformed"

with zipfile.ZipFile(java_small_test_files_zip) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
        try:
            zf.extract(member, extracted_java_small_files_path)
        except zipfile.error as e:
            pass

with zipfile.ZipFile(java_small_test_files_zip) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
        try:
            zf.extract(member, extracted_java_med_files_path)
        except zipfile.error as e:
            pass
            
with zipfile.ZipFile(java_small_test_files_zip) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
        try:
            zf.extract(member, extracted_java_large_files_path)
        except zipfile.error as e:
            pass
