import argparse
import glob
from linesegmentation.detection.detection_rest import LineDetectionRest, LineDetectionSettings
from linesegmentation.detection.callback import LineDetectionCallback, DummyLineDetectionCallback
import tqdm


def glob_all(filenames):
    files = []
    for f in filenames:
        files += glob.glob(f)

    return files


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', ''):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Detects music lines in historical documents')
    parser.add_argument("--gray", type=str, required=True, nargs="+",
                        help="directory name of the grayscale images")
    parser.add_argument("--processes", type=int, default=8,
                        help="Number of processes to use")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Display debug images")
    args = parser.parse_args()

    gray_file_paths = sorted(glob_all(args.gray))
    #print("Loading {} files with character height {}".format(len(gray_file_paths), args.space_height))


    import os

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-427.nrm.png')
    model_line = os.path.join(project_dir, 'demo/models/line/virtual_lines/model')
    model_region = os.path.join(project_dir, 'demo/models/region/model')
    settings_prediction = LineDetectionSettings(debug=True, min_lines_per_system=3, line_number=5, line_space_height=20,
                                                target_line_space_height=10, line_fit_distance=1.0, debug_model=True,
                                                model=model_line)
    t_callback = DummyLineDetectionCallback(total_steps=10, total_pages=1)
    lineDetector = LineDetectionRest(settings_prediction, callback=t_callback)
    for i_ind, i in tqdm.tqdm(enumerate(lineDetector.detect_paths(gray_file_paths)), total = len(gray_file_paths)):
        pass


if __name__ == "__main__":
    main()
