import argparse
import glob
import json
import os
import shutil
import subprocess
import uuid
import csv
from collections import OrderedDict
from tqdm import tqdm
from joblib import delayed
from joblib import Parallel
import pandas as pd


def create_video_folders(dataset, output_dir, tmp_dir):
    """Creates a directory for each label name in the dataset."""
    if 'label-name' not in dataset.columns:
        this_dir = os.path.join(output_dir, 'test')
        dataset["label-name"] = [0 for x in range(len(dataset))]
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        # I should return a dict but ...
        return this_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    label_to_dir = {}
    for label_name in dataset['label-name'].unique():
        this_dir = os.path.join(output_dir, label_name)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        label_to_dir[label_name] = this_dir
    return label_to_dir


def construct_video_filename(row, label_to_dir, trim_format='%06d'):
    """Given a dataset row, this function constructs the
       output filename for a given video.
    """
    basename = '%s_%s_%s.mp4' % (row['video-id'],
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    if not isinstance(label_to_dir, dict):
        dirname = label_to_dir
    else:
        dirname = label_to_dir[row['label-name']]
    output_filename = os.path.join(dirname, basename)
    return output_filename


def download_clip(video_identifier, output_filename,
                  start_time, end_time, out_height=256, out_width=-1,
                  tmp_dir='/tmp/kinetics',
        # output = subprocess.check_output(command_resize, shell=True,
        #                                  stderr=subprocess.STDOUT)
                  num_attempts=5,
                  url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.

    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'
    assert len(video_identifier) == 11, 'video_identifier must have length 11'

    status = False
    # Construct command line for getting the direct video link.
    tmp_filename = os.path.join(tmp_dir,
                                '%s.%%(ext)s' % uuid.uuid4())
    command = ['youtube-dl',
               '--quiet', '--no-warnings',
               '-f', 'mp4',
               '-o', '"%s"' % tmp_filename,
               '"%s"' % (url_base + video_identifier)]
    command = ' '.join(command)
    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output.decode("utf-8")
        else:
            break

    tmp_filename = glob.glob('%s*' % tmp_filename.split('.')[0])[0]
    # Construct command to trim the videos (ffmpeg required).
    # Example command
    # ffmpeg -i /tmp/kinetics/11d602ad-b859-434a-8770-d120dd96f348.mp4 -ss 0 -t 10  -c:v libx264 -c:a copy -threads 1 Dataset/validate/washing feet/--GkrdYZ9Tc_000000_000010.mp4
    command = ['ffmpeg',
               '-i', '"%s"' % tmp_filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-filter:v scale="trunc(oh*a/2)*2:%s"' % out_height,
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print("Exception!")
        print(output.decode("utf-8"))
        print(command)
        return status, err.output.decode("utf-8")

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    os.remove(tmp_filename)
    return status, 'Downloaded'


def download_clip_wrapper(row, label_to_dir, trim_format, tmp_dir):
    """Wrapper for parallel processing purposes."""
    output_filename = construct_video_filename(row, label_to_dir,
                                               trim_format)
    
    clip_id = os.path.basename(output_filename).split('.mp4')[0]
    if os.path.exists(output_filename):
        status = tuple([clip_id, True, 'Exists', output_filename, row['label-name']])
        return status

    downloaded, log = download_clip(row['video-id'], output_filename,
                                    row['start-time'], row['end-time'],
                                    tmp_dir=tmp_dir)
    status = tuple([clip_id, downloaded, log, output_filename, row['label-name']])
    return status


def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
    """Returns a parsed DataFrame.

    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'

    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
    if 'youtube_id' in df.columns:
        columns = OrderedDict([
            ('youtube_id', 'video-id'),
            ('time_start', 'start-time'),
            ('time_end', 'end-time'),
            ('label', 'label-name')])
        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]
    return df


def main(input_csv, output_dir,
         trim_format='%06d', num_jobs=24, tmp_dir='/tmp/kinetics',
         drop_duplicates=False, total=-1, label_csv=None):

    # Reading and parsing Kinetics.
    dataset = parse_kinetics_annotations(input_csv)
    # if os.path.isfile(drop_duplicates):
    #     print('Attempt to remove duplicates')
    #     old_dataset = parse_kinetics_annotations(drop_duplicates,
    #                                              ignore_is_cc=True)
    #     df = pd.concat([dataset, old_dataset], axis=0, ignore_index=True)
    #     df.drop_duplicates(inplace=True, keep=False)
    #     print(dataset.shape, old_dataset.shape)
    #     dataset = df
    #     print(dataset.shape)
    if total == -1:
        total = len(dataset)
    dataset = dataset[:total]
    if label_csv is None or not os.path.exists(label_csv):
        import requests
        import io
        print("Downloading csv")
        # TODO add input param for csv
        url = 'https://gist.githubusercontent.com/willprice/f19da185c9c5f32847134b87c1960769/raw/9dc94028ecced572f302225c49fcdee2f3d748d8/kinetics_400_labels.csv'
        r = requests.get(url, allow_redirects=True)
        labels_dict = {y:x for x,y in csv.reader(io.StringIO(r.content.decode("utf-8")))}
        # print(labels_dict)
    else:
        with open(label_csv,"r") as labels_file:
            labels_dict = {y:x for x,y in csv.reader(labels_file)}
    labels_dict[0]=0
    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, output_dir, tmp_dir)
    # Download all clips.
    csv_file = open(os.path.join(output_dir, "dataset.csv"), "w")
    if num_jobs == 1:
        status_lst = []
        for i, row in tqdm(dataset[:total].iterrows(),total=total):
            status_lst.append(download_clip_wrapper(row, label_to_dir,
                                                    trim_format, tmp_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(download_clip_wrapper)(
            row, label_to_dir,
            trim_format, tmp_dir) for i, row in tqdm(dataset[:total].iterrows(),total=total))

    csv_writer = csv.DictWriter(csv_file, fieldnames=["path","label"])
    for status in status_lst:
        if status[1]:
            # print("writing row...")
            csv_writer.writerow({"path":os.path.abspath(status[3]), "label":labels_dict[status[4]]})
    csv_file.close()
    # Clean tmp dir.
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # Save download report.
    with open('download_report.json', 'w') as fobj:
        try:
            fobj.write(json.dumps(status_lst))
        except:
            print(status_lst)


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('--label_csv', type=str, default=None,
                   help=('CSV file containing the following format: '
                         'Id, Class label'),)
    p.add_argument('-f', '--trim-format', type=str, default='%06d',
                   help=('This will be the format for the '
                         'filename of trimmed videos: '
                         'videoid_%0xd(start_time)_%0xd(end_time).mp4'))
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/kinetics')
    p.add_argument('--total',type=int, default=-1)

    p.add_argument('--drop-duplicates', type=str, default='non-existent',
                   help='Unavailable at the moment')
                   # help='CSV file of the previous version of Kinetics.')
    main(**vars(p.parse_args()))
