import argparse

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Tremor Frequency Analysis"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to HDF5 file"
    )


    parser.add_argument(
        "--method",
        default="both",
        choices=["fft", "cwt", "both"],
        help = (
        "Spectral analysis method to use.\n"
        "fft  = Fast Fourier Transform\n"
        "cwt  = Continuous Wavelet Transform\n"
        "both = FFT and CWT"
        )
    )

    parser.add_argument(
        "--freq-min",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--freq-max",
        type=float,
        default=12.0
    )

    parser.add_argument(
        "--axis",
        default="mag",
        choices=["x", "y", "z", "mag"],
        help=(
            "Acceleration signal to analyze.\n"
            "mag = vector magnitude "
        )
    )

    parser.add_argument(
        "--save",
        action="store_true"
    )

    parser.add_argument(
        "--output",
        default="output",
        help=(
            "Output directory")
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Display figures interactively."
        )

    )

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_arguments()

    # TODO run implementation