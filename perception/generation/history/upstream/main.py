from scripts import make_polyhedron_objs
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="assets/generated")
    parser.add_argument("--size", type=float, default=0.08, help="max bbox extent in meters")
    args = parser.parse_args()

    make_polyhedron_objs.generate_all(args.out, args.size)