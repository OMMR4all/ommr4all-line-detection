import argparse
import glob
from linesegmentation.detection.detection import LineDetection, LineDetectionSettings
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
    parser.add_argument("--load", type=str,
                        help="Model to load")
    parser.add_argument("--space_height", type=int, default=20,
                        help="Average space between two lines. If set to '0',"
                             " the value will be calculated automatically ...")
    parser.add_argument("--target_line_height", type=int, default=10,
                        help="Scale the data images so that the space height"
                             " matches this value (must be the same as in training)")
    parser.add_argument("--gray", type=str, required=True, nargs="+",
                        help="directory name of the grayscale images")
    parser.add_argument("--num_line", type=int, default=4,
                        help="number of lines in a system. Can also be set to 0")
    parser.add_argument("--min_hrun_lengths", type=int, default=6,
                        help="Minimum allowed horizontal run lengths")
    parser.add_argument("--interpolate", type= str2bool, default=True,
                        help="Line extension through interpolation")
    parser.add_argument("--post_process", type=str2bool, default=True,
                        help="Post processing line systems ")
    parser.add_argument("--smooth_lines", type=int, default=2,
                        help="0 = Off, 1 = basic Smoothing (low pass filter), 2 = Advanced Smoothing (slower)")
    parser.add_argument("--smooth_value_adv", type=int, default=25,
                        help="Advanced smooth value")
    parser.add_argument("--smooth_value_lowpass", type=int, default=5,
                        help="Low_pass_filter smooth value")
    parser.add_argument("--line_fit_distance", type=float, default=1.0,
                        help="Line simplification. Higher values simplifies lines more")
    parser.add_argument("--processes", type=int, default=8,
                        help="Number of processes to use")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Display debug images")
    args = parser.parse_args()

    gray_file_paths = sorted(glob_all(args.gray))
    print("Loading {} files with character height {}".format(len(gray_file_paths), args.space_height))

    settings = LineDetectionSettings(
        line_number=args.num_line,
        horizontal_min_length=args.min_hrun_lengths,
        line_interpolation=args.interpolate,
        debug=args.debug,
        line_space_height=args.space_height,
        target_line_space_height=args.target_line_height,
        model=args.load,
        post_process=args.post_process,
        smooth_lines=args.smooth_lines,
        line_fit_distance=args.line_fit_distance,
        processes=args.processes,
        smooth_value_adv=args.smooth_value_adv

    )
    lineDetector = LineDetection(settings)
    for i_ind, i in tqdm.tqdm(enumerate(lineDetector.detect_paths(gray_file_paths)), total = len(gray_file_paths)):
        pass


if __name__ == "__main__":
    main()
